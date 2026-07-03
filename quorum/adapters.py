"""External -> quorum request/config adapters shared by both integration surfaces.

The two entry points that let a *foreign* caller drive a deliberation -- the embed
API (``api.py``) and the OpenAI-compatible server (``serveapi.py``) -- each have to
translate the caller's vocabulary into quorum's before handing off to the
orchestrator. Those translations used to be duplicated; this module is their single
home so both surfaces stay byte-for-byte consistent and a new host has one place to
look:

- ``host_config`` -- a host tool's ``config.yaml`` (its ``ai:`` block + an optional
  ``quorum:`` block) -> a full, deep-merged quorum config.
- ``split_messages`` -- an OpenAI ``messages`` array -> quorum's
  ``(system, history, last_user)`` triple.
- ``select_strategy`` -- a request's ``model`` field -> the strategy to run (the
  named strategy if known, else the configured default).

Layering: this is an *entry-layer* helper. It may import the domain ``config`` and
the strategy registry's ``available``, but nothing above them, and it must never be
imported by strategies / the orchestrator / reasoning modules -- only ``api.py`` and
``serveapi.py`` import it. That keeps the dependency arrows pointing down (no cycles).
"""
from __future__ import annotations

from typing import Any

from .config import DEFAULT_CONFIG, _deep_merge
from .strategies import available as strategies_available


def host_config(cfg: dict) -> dict:
    """Compose a quorum config from the host tool's ``ai:`` + optional ``quorum:``.

    The host's ``ai:`` block supplies the default provider/model/key (so even a
    single model gains self-refine); an optional ``quorum:`` block overlays the
    strategy, extra council members/providers, and run knobs. The result is a full
    quorum config deep-merged onto ``DEFAULT_CONFIG``.
    """
    ai = cfg.get("ai", {}) or {}
    q = cfg.get("quorum", {}) or {}
    provider = ai.get("provider") or "openai"
    model = ai.get("model") or ""
    role = f"{provider}:{model}"

    providers = {provider: {"base_url": ai.get("base_url", ""),
                            "api_key_env": ai.get("api_key_env", "")}}
    providers.update(q.get("providers") or {})
    members = q.get("members") or [{"name": "m1", "provider": provider, "model": model}]

    overlay = {
        "providers": providers,
        "council": {
            "members": members,
            "judge": q.get("judge") or role,
            "chairman": q.get("chairman") or role,
            "aggregator": q.get("aggregator") or role,
        },
        "run": {
            "strategy": q.get("strategy", "refine"),
            "max_rounds": int(q.get("max_rounds", 2)),
            "target_score": float(q.get("target_score", 85)),
            "temperature": ai.get("temperature", 0.3),
            "max_tokens": ai.get("max_tokens", 700),
            "parallel": bool(q.get("parallel", False)),
        },
        "promptsmith": {"enabled": False},
        "cost": {"budget_usd": float(q.get("budget_usd", 0.0))},
    }
    return _deep_merge(DEFAULT_CONFIG, overlay)


def split_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]], str]:
    """Return ``(system, history, last_user)`` from an OpenAI ``messages`` array.

    ``history`` is every user/assistant turn *before* the final user message, so a
    multi-turn client (e.g. a chatbot) gets conversation memory; ``last_user`` is
    the message to act on. All system messages are concatenated.
    """
    system = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
    convo = [{"role": m.get("role", ""), "content": m.get("content", "")}
             for m in messages if m.get("role") in ("user", "assistant")]
    last_user = -1
    for i, m in enumerate(convo):
        if m["role"] == "user":
            last_user = i
    if last_user < 0:
        return system, convo, ""
    return system, convo[:last_user], convo[last_user]["content"]


def select_strategy(model: str, cfg: dict) -> str:
    """Pick the strategy for a request.

    Use the request's ``model`` field when it names a known strategy (as reported by
    ``strategies.available``), otherwise fall back to the config's ``run.strategy``
    default.
    """
    strategies = set(strategies_available())
    default_strategy = (cfg.get("run", {}) or {}).get("strategy", "refine")
    model = model or ""
    return model if model in strategies else default_strategy
