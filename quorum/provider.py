"""Provider layer: talk to any OpenAI-compatible ``/chat/completions`` endpoint,
or answer fully offline via the built-in ``mock`` provider.

The rest of the engine only ever sees :class:`Completion` objects and never
touches the network directly, which keeps strategies/judge testable offline.
Live calls use the standard library (``urllib``) so the core has no third-party
runtime dependency beyond PyYAML.

Security note (OWASP LLM01): model outputs are fed back into other models' prompts
during deliberation. Callers must frame peer text as *data, not instructions*
(the judge/aggregator system prompts do exactly that). The provider itself never
executes anything a model returns.
"""
from __future__ import annotations

import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from . import cost
from .config import api_key, member_specs, provider_conf, role_spec
from .model import ModelSpec, Turn, content_hash


@dataclass
class Completion:
    text: str
    model: str
    provider: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    ok: bool = True
    error: str = ""
    latency_ms: int = 0


# --------------------------------------------------------------------------
# offline mock
# --------------------------------------------------------------------------
class MockResponder:
    """Deterministic offline responder used by selftest, replay, and ``--dry``.

    It reads sentinels placed in role system prompts (``QUORUM-JUDGE``,
    ``QUORUM-PROMPTSMITH``, ``QUORUM-CHAIRMAN``, ``QUORUM-AGGREGATOR``,
    ``QUORUM-REVIEW``) to decide what shape to return, and ramps judge scores by
    the ``ROUND=N`` marker so stopping logic is exercised deterministically.
    """

    def __init__(self, script: Optional[dict[str, str]] = None):
        self.script = script or {}

    def respond(self, spec: ModelSpec, messages: list[dict[str, str]]) -> str:
        system = _concat(messages, "system")
        user = _last(messages, "user")
        if "QUORUM-JUDGE" in system:
            return self._judge(user)
        if "QUORUM-PROMPTSMITH" in system:
            return self._promptsmith(user)
        if "QUORUM-CHAIRMAN" in system or "QUORUM-AGGREGATOR" in system:
            return self._synthesize(spec, user)
        if "QUORUM-REVIEW" in system:
            return self._review(user)
        return self._propose(spec, user)

    def _judge(self, user: str) -> str:
        m = re.search(r"ROUND=(\d+)", user or "")
        rnd = int(m.group(1)) if m else 1
        score = float(min(55 + 15 * rnd, 96))
        labels = re.findall(r"CANDIDATE (\w+)", user or "")
        best = labels[0] if labels else "A"
        return json.dumps({
            "score": score,
            "sub_scores": {"correctness": score, "completeness": max(0, score - 3),
                           "clarity": min(100, score + 1), "grounding": max(0, score - 2)},
            "best": best,
            "rationale": f"Mock judgment (round {rnd}): candidate {best} is strongest.",
        })

    def _promptsmith(self, user: str) -> str:
        return ("Approach: restate the goal in one line, decompose the problem, state "
                "assumptions explicitly, reason step by step, verify the result against the "
                "requirements, then present a clear and complete final answer.")

    def _synthesize(self, spec: ModelSpec, user: str) -> str:
        return f"[{spec.model}] Synthesized final answer combining the members' best points."

    def _review(self, user: str) -> str:
        labels = re.findall(r"CANDIDATE (\w+)", user or "") or ["A", "B"]
        ranking = ", ".join(labels)
        return f"Ranking (best first): {ranking}. Rationale: mock peer review."

    def _propose(self, spec: ModelSpec, user: str) -> str:
        seed = content_hash(spec.model, user)[:6]
        return f"[{spec.model}] Answer ({seed}): a concrete response to the prompt."


def _concat(messages: list[dict[str, str]], role: str) -> str:
    return "\n".join(m.get("content", "") for m in messages if m.get("role") == role)


def _last(messages: list[dict[str, str]], role: str) -> str:
    for m in reversed(messages):
        if m.get("role") == role:
            return m.get("content", "")
    return ""


def _after(text: str, marker: str) -> str:
    i = (text or "").find(marker)
    return text[i + len(marker):] if i >= 0 else ""


# --------------------------------------------------------------------------
# provider
# --------------------------------------------------------------------------
class Provider:
    """Routes a :class:`ModelSpec` + messages to the right endpoint or the mock."""

    def __init__(self, cfg: dict, *, mock: Optional[MockResponder] = None, timeout: int = 60):
        self.cfg = cfg
        self.mock = mock or MockResponder()
        self.timeout = timeout

    # -- single call ------------------------------------------------------
    def complete(self, spec: ModelSpec, messages: list[dict[str, str]], *,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 store: Any = None, cache: bool = True) -> Completion:
        run = self.cfg.get("run", {}) or {}
        temp = run.get("temperature", 0.5) if temperature is None else temperature
        maxt = run.get("max_tokens", 1200) if max_tokens is None else max_tokens
        key = content_hash(spec.provider, spec.model, temp, json.dumps(messages, sort_keys=True))

        if cache and store is not None:
            hit = store.ai_cache_get(key)
            if hit is not None:
                tin = cost.count_messages(messages, spec.model)
                tout = cost.count_tokens(hit, spec.model)
                return Completion(hit, spec.model, spec.provider, tin, tout,
                                  cost.price(self.cfg, spec.model, tin, tout))

        t0 = time.time()
        if spec.provider == "mock":
            text = self.mock.respond(spec, messages)
            comp = self._finish(spec, messages, text, t0)
        else:
            comp = self._http(spec, messages, temp, maxt, t0)

        if comp.ok and cache and store is not None:
            store.ai_cache_put(key, spec.model, _last(messages, "user")[:2000], comp.text)
        return comp

    def _finish(self, spec: ModelSpec, messages: list[dict[str, str]], text: str,
                t0: float, tokens_out: Optional[int] = None) -> Completion:
        tin = cost.count_messages(messages, spec.model)
        tout = cost.count_tokens(text, spec.model) if tokens_out is None else tokens_out
        return Completion(
            text=text, model=spec.model, provider=spec.provider,
            tokens_in=tin, tokens_out=tout,
            cost_usd=cost.price(self.cfg, spec.model, tin, tout),
            latency_ms=int((time.time() - t0) * 1000),
        )

    def _http(self, spec: ModelSpec, messages: list[dict[str, str]],
              temperature: float, max_tokens: int, t0: float) -> Completion:
        conf = provider_conf(self.cfg, spec.provider)
        base = (conf.get("base_url", "") or "").rstrip("/")
        if not base:
            return Completion("", spec.model, spec.provider, ok=False,
                              error=f"provider '{spec.provider}' has no base_url")
        body = json.dumps({
            "model": spec.model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(base + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        k = api_key(self.cfg, spec.provider)
        if k:
            req.add_header("Authorization", f"Bearer {k}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - configured endpoint
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {}) or {}
            comp = self._finish(spec, messages, text, t0,
                                tokens_out=usage.get("completion_tokens"))
            if usage.get("prompt_tokens"):
                comp.tokens_in = int(usage["prompt_tokens"])
                comp.cost_usd = cost.price(self.cfg, spec.model, comp.tokens_in, comp.tokens_out)
            return comp
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as e:
            return Completion("", spec.model, spec.provider, ok=False,
                              error=str(e), latency_ms=int((time.time() - t0) * 1000))

    # -- fan-out ----------------------------------------------------------
    def complete_many(self, jobs: list[tuple[ModelSpec, list[dict[str, str]]]], *,
                      temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                      store: Any = None, cache: bool = True) -> list[Completion]:
        """Run several completions, in parallel when ``run.parallel`` is set.

        Order is preserved. A store (SQLite) is not thread-safe, so caching is
        disabled during parallel fan-out and callers should pass ``store`` only
        for sequential calls.
        """
        parallel = bool((self.cfg.get("run", {}) or {}).get("parallel", True)) and len(jobs) > 1
        if not parallel:
            return [self.complete(s, m, temperature=temperature, max_tokens=max_tokens,
                                  store=store, cache=cache) for s, m in jobs]
        results: list[Optional[Completion]] = [None] * len(jobs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
            futs = {ex.submit(self.complete, s, m, temperature=temperature,
                              max_tokens=max_tokens, store=None, cache=False): i
                    for i, (s, m) in enumerate(jobs)}
            for fut in concurrent.futures.as_completed(futs):
                results[futs[fut]] = fut.result()
        return [r for r in results if r is not None]


def for_config(cfg: dict, *, mock: Optional[MockResponder] = None) -> Provider:
    return Provider(cfg, mock=mock)


def to_turn(comp: Completion, round_index: int, member: str, kind: str) -> Turn:
    """Build a transcript :class:`Turn` from a :class:`Completion`."""
    return Turn(
        round=round_index, member=member, model=comp.model, kind=kind,
        content=comp.text, tokens_in=comp.tokens_in, tokens_out=comp.tokens_out,
        cost_usd=comp.cost_usd,
    )


# --------------------------------------------------------------------------
# `quorum models`
# --------------------------------------------------------------------------
def list_models(cfg: dict, ping: bool = False) -> int:
    members = member_specs(cfg)
    roles = [role_spec(cfg, r) for r in ("judge", "chairman", "aggregator")]
    print("  council:")
    for m in members:
        print(f"    - {m.name:<8} {m.ref()}")
    print("  roles:")
    for r in roles:
        print(f"    - {r.role:<10} {r.ref()}")
    if not ping:
        return 0

    prov = for_config(cfg)
    print("\n  ping:")
    seen: dict[str, Completion] = {}
    for spec in members + roles:
        if spec.ref() in seen:
            continue
        comp = prov.complete(spec, [{"role": "user", "content": "ping"}], max_tokens=5, cache=False)
        seen[spec.ref()] = comp
        mark = "ok " if comp.ok else "ERR"
        detail = f"{comp.latency_ms}ms" if comp.ok else comp.error[:60]
        print(f"    [{mark}] {spec.ref():<48} {detail}")
    return 0 if all(c.ok for c in seen.values()) else 1
