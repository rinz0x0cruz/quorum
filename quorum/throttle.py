"""Throttle analysis: *what* is rate-limiting a run, and *how hard*.

Reads the per-attempt telemetry recorded by :mod:`quorum.provider` (the
``api_calls`` table) and turns it into an actionable report: per-model 429 rate,
the observed requests-per-minute against the provider ceiling, and concrete
recommendations. Optionally probes a provider's key endpoint (OpenRouter's
``GET /key``) for the remaining daily quota.

Free OpenRouter ``:free`` models are capped by *request count*: 20 requests/min,
and 50/day (<10 credits purchased) or 1000/day (>=10). Those limits are governed
globally across keys, so the lever is making fewer requests -- which this report
helps you see and tune.

The summarising logic is pure (list of rows -> dict), so it is fully
offline-testable; only :func:`key_status` touches the network.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

from .config import api_key, provider_conf

# OpenRouter free-tier ceilings (docs: /docs/api-reference/limits).
FREE_RPM = 20
FREE_RPD_NO_CREDITS = 50
FREE_RPD_WITH_CREDITS = 1000

_UA = "quorum (+https://github.com/rinz0x0cruz/quorum)"


def _minute(ts: str) -> str:
    """Bucket an ISO ``...THH:MM:SSZ`` timestamp to the minute."""
    return ts[:16] if ts else ""


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate api_call rows into per-model and per-provider throttle stats.

    Pure function -- no I/O -- so tests feed it synthetic rows.
    """
    by_model: dict[str, dict[str, Any]] = {}
    prov_minute: dict[str, dict[str, int]] = {}   # provider -> minute -> count
    for r in rows:
        model = r.get("model", "") or "?"
        prov = r.get("provider", "") or "?"
        status = str(r.get("status", ""))
        code = int(r.get("http_code", 0) or 0)
        m = by_model.setdefault(model, {
            "provider": prov, "total": 0, "ok": 0, "throttled": 0, "errors": 0,
            "latency_sum": 0, "latency_n": 0, "last_rl_remaining": None,
        })
        m["total"] += 1
        if status == "ok":
            m["ok"] += 1
            if r.get("latency_ms"):
                m["latency_sum"] += int(r["latency_ms"])
                m["latency_n"] += 1
        elif code == 429:
            m["throttled"] += 1
        else:
            m["errors"] += 1
        rem = r.get("rl_remaining")
        if rem not in (None, 0, ""):
            m["last_rl_remaining"] = int(rem)
        prov_minute.setdefault(prov, {})
        key = _minute(r.get("ts", ""))
        prov_minute[prov][key] = prov_minute[prov].get(key, 0) + 1

    for m in by_model.values():
        m["rate_429"] = round(m["throttled"] / m["total"], 3) if m["total"] else 0.0
        m["avg_latency_ms"] = int(m["latency_sum"] / m["latency_n"]) if m["latency_n"] else 0
        del m["latency_sum"], m["latency_n"]

    peak_rpm = {p: (max(buckets.values()) if buckets else 0) for p, buckets in prov_minute.items()}
    return {
        "total": len(rows),
        "by_model": by_model,
        "peak_rpm": peak_rpm,
        "throttled": sum(m["throttled"] for m in by_model.values()),
    }


def recommendations(summary: dict[str, Any], cfg: dict,
                    key: Optional[dict[str, Any]] = None) -> list[str]:
    """Turn a summary into concrete, config-aware suggestions. Pure function."""
    recs: list[str] = []
    run = cfg.get("run", {}) or {}
    members = (cfg.get("council", {}) or {}).get("members", []) or []

    peak = max(summary.get("peak_rpm", {}).values(), default=0)
    if peak >= FREE_RPM:
        recs.append(f"Peak {peak} req/min hit the free ceiling ({FREE_RPM}/min). "
                    f"Set run.rate_limit_rpm to ~{FREE_RPM - 2} to pace calls under it.")
    elif peak >= FREE_RPM * 0.8:
        recs.append(f"Peak {peak} req/min is close to the {FREE_RPM}/min ceiling; "
                    f"consider run.rate_limit_rpm ~{FREE_RPM - 2}.")

    if summary.get("throttled") and run.get("parallel", True):
        recs.append("Saw 429s with run.parallel on: parallel fan-out bursts trip the "
                    "per-minute limit. Try run.parallel: false to space calls out.")

    hot = [m for m, s in summary.get("by_model", {}).items() if s.get("rate_429", 0) >= 0.1]
    if hot and len(members) > 1:
        recs.append(f"High 429 rate on: {', '.join(hot)}. Spread load across more models, "
                    "reduce members, or add run.fallbacks alternates.")

    if key and isinstance(key, dict) and not key.get("error"):
        if key.get("is_free_tier"):
            cap = FREE_RPD_WITH_CREDITS if (key.get("usage", 0) or 0) > 0 else FREE_RPD_NO_CREDITS
            recs.append(f"Free tier: daily :free cap is ~{cap} requests. "
                        "Purchasing >=10 credits raises it to 1000/day.")
        rem = key.get("limit_remaining")
        if rem is not None:
            recs.append(f"Key credits remaining: {rem}.")

    if not recs:
        recs.append("No throttling detected in the recorded window.")
    return recs


def key_status(cfg: dict, provider: str = "openrouter", *, timeout: int = 15) -> Optional[dict[str, Any]]:
    """Probe a provider's key endpoint for quota/credits (OpenRouter ``GET /key``).

    Returns the provider's ``data`` object, ``{"error": ...}`` on failure, or
    ``None`` if the provider has no base_url/key configured. Network I/O.
    """
    conf = provider_conf(cfg, provider)
    base = (conf.get("base_url", "") or "").rstrip("/")
    key = api_key(cfg, provider)
    if not base or not key:
        return None
    req = urllib.request.Request(base + "/key",
                                 headers={"Authorization": f"Bearer {key}", "User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - configured endpoint
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("data", data) if isinstance(data, dict) else {"raw": data}
    except (urllib.error.URLError, ValueError, TimeoutError) as e:
        return {"error": str(e)}


def run(cfg: dict, store: Any, *, provider: str = "openrouter", limit: int = 5000,
        probe: bool = True) -> int:
    """CLI entry: print a throttle report from recorded telemetry (+ optional quota probe)."""
    rows = store.api_calls_recent(limit) if hasattr(store, "api_calls_recent") else []
    summary = summarize(rows)

    if not rows:
        print("  no API-call telemetry recorded yet. Run a live deliberation first "
              "(mock runs make no HTTP calls).")
        return 0

    print(f"  attempts recorded: {summary['total']}  (429s: {summary['throttled']})")
    print("  by model:")
    print(f"    {'model':<44} {'reqs':>5} {'ok':>4} {'429':>4} {'err':>4} {'429%':>5} {'lat':>7}")
    for model, s in sorted(summary["by_model"].items(), key=lambda kv: -kv[1]["total"]):
        print(f"    {model[:44]:<44} {s['total']:>5} {s['ok']:>4} {s['throttled']:>4} "
              f"{s['errors']:>4} {s['rate_429'] * 100:>4.0f}% {s['avg_latency_ms']:>5}ms")

    print("  peak requests/min per provider:")
    for prov, rpm in sorted(summary["peak_rpm"].items(), key=lambda kv: -kv[1]):
        flag = "  <= at/over free ceiling" if rpm >= FREE_RPM else ""
        print(f"    {prov:<16} {rpm:>3}/min (free limit {FREE_RPM}){flag}")

    key = key_status(cfg, provider) if probe else None
    if key and not key.get("error"):
        print("  quota:")
        for k in ("is_free_tier", "usage", "usage_daily", "limit", "limit_remaining"):
            if k in key:
                print(f"    {k:<16} {key[k]}")
    elif key and key.get("error"):
        print(f"  quota probe failed: {key['error'][:80]}")

    print("  recommendations:")
    for rec in recommendations(summary, cfg, key):
        print(f"    - {rec}")
    return 0
