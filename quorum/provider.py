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
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
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
        if "QUORUM-GRADER" in system:
            return self._grade(user)
        if "QUORUM-USC" in system:
            return self._usc(user)
        if "QUORUM-REFLECT" in system:
            return self._reflect(user)
        if "QUORUM-VERIFY-PLAN" in system:
            return self._verify_plan(user)
        if "QUORUM-VERIFY-ANSWER" in system:
            return self._verify_answer(user)
        if "QUORUM-VERIFY-REVISE" in system:
            return self._verify_revise(spec, user)
        if "QUORUM-SELFDISCOVER-PLAN" in system:
            return self._sd_plan(user)
        if "QUORUM-SELFDISCOVER-SOLVE" in system:
            return self._sd_solve(spec, user)
        if "QUORUM-STEPBACK-ABSTRACT" in system:
            return self._sb_abstract(user)
        if "QUORUM-STEPBACK-SOLVE" in system:
            return self._sb_solve(spec, user)
        if "QUORUM-LTM-DECOMPOSE" in system:
            return self._ltm_decompose(user)
        if "QUORUM-LTM-SOLVE" in system:
            return self._ltm_solve(spec, user)
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

    def _grade(self, user: str) -> str:
        return json.dumps({"score": 90.0, "correct": True,
                           "rationale": "Mock grade: candidate matches the reference."})

    def _usc(self, user: str) -> str:
        labels = re.findall(r"CANDIDATE (\w+)", user or "") or ["A"]
        return json.dumps({"choice": labels[0]})

    def _reflect(self, user: str) -> str:
        return ("Reflection: the previous answer under-specified a step; next time decompose the "
                "problem and verify each part against the requirements before answering.")

    def _verify_plan(self, user: str) -> str:
        return "1. Is the main claim correct?\n2. Are the stated assumptions valid?"

    def _verify_answer(self, user: str) -> str:
        return "1. Yes, the main claim holds.\n2. The stated assumptions are valid."

    def _verify_revise(self, spec: ModelSpec, user: str) -> str:
        return f"[{spec.model}] Verified final answer, corrected against the checks."

    def _sd_plan(self, user: str) -> str:
        return ("1. Restate what is asked.\n2. List the given facts and constraints.\n"
                "3. Reason step by step to the result.\n4. Verify the result is plausible.")

    def _sd_solve(self, spec: ModelSpec, user: str) -> str:
        return f"[{spec.model}] Final answer produced by following the composed reasoning structure."

    def _sb_abstract(self, user: str) -> str:
        return ("Step-back question: what general principle governs this? "
                "Principle: apply the relevant definition/rule, then reason to the specifics.")

    def _sb_solve(self, spec: ModelSpec, user: str) -> str:
        return f"[{spec.model}] Final answer reasoned from the general principle."

    def _ltm_decompose(self, user: str) -> str:
        return ("1. What are the given quantities?\n2. What operation combines them?\n"
                "3. Compute the final result.")

    def _ltm_solve(self, spec: ModelSpec, user: str) -> str:
        return f"[{spec.model}] Sub-answer derived from the prior steps."

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


def _rl_from_headers(headers: Any) -> tuple[int, int, str]:
    """Parse ``X-RateLimit-*`` headers (present on OpenRouter/OpenAI responses)."""
    if not headers:
        return 0, 0, ""

    def _int(name: str) -> int:
        v = headers.get(name)
        try:
            return int(float(v)) if v not in (None, "") else 0
        except (ValueError, TypeError):
            return 0

    return (_int("X-RateLimit-Limit"), _int("X-RateLimit-Remaining"),
            str(headers.get("X-RateLimit-Reset") or ""))


# --------------------------------------------------------------------------
# rate limiting
# --------------------------------------------------------------------------
class RateLimiter:
    """Pace :meth:`acquire` calls to at most ``rpm`` per minute (thread-safe).

    Shared across a provider's parallel fan-out so bursts are spread out instead
    of tripping a per-minute cap (free tiers throttle ~20 req/min). It advances a
    single ``next allowed time`` cursor under a lock, then sleeps outside it, so N
    concurrent threads are spaced by ``60/rpm`` seconds. ``rpm <= 0`` disables it.
    """

    def __init__(self, rpm: float):
        self.rpm = float(rpm or 0)
        self.interval = 60.0 / self.rpm if self.rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> float:
        """Block until a slot is free; return the seconds waited (0 if disabled)."""
        if self.rpm <= 0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            start = now if now >= self._next else self._next
            self._next = start + self.interval
            wait = start - now
        if wait > 0:
            time.sleep(wait)
        return wait


# Process-wide per-provider limiters, so every Provider in this process (bench
# tasks, server requests, a long-running embed host) shares one pacing budget per
# provider -- a fresh Provider per call would otherwise reset pacing and let
# bursts through, defeating the point on free tiers.
_LIMITERS: dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def _shared_limiter(name: str, rpm: float) -> RateLimiter:
    with _LIMITERS_LOCK:
        lim = _LIMITERS.get(name)
        if lim is None or lim.rpm != rpm:
            lim = RateLimiter(rpm)
            _LIMITERS[name] = lim
        return lim


def reset_rate_limiters() -> None:
    """Clear the process-wide limiter registry (test hygiene / reconfiguration)."""
    with _LIMITERS_LOCK:
        _LIMITERS.clear()


# --------------------------------------------------------------------------
# provider
# --------------------------------------------------------------------------
class Provider:
    """Routes a :class:`ModelSpec` + messages to the right endpoint or the mock."""

    def __init__(self, cfg: dict, *, mock: Optional[MockResponder] = None, timeout: int = 60,
                 max_retries: int = 3, backoff: float = 2.5, telemetry: Any = None,
                 retry_429: int = 1):
        self.cfg = cfg
        self.mock = mock or MockResponder()
        self.timeout = timeout
        self.max_retries = max_retries      # retries on 429/5xx (free tiers throttle)
        self.backoff = backoff              # initial backoff seconds (doubles each retry)
        self.retry_429 = retry_429          # fewer retries on 429: a per-minute cap won't clear in seconds -> rotate to a fallback instead
        self.telemetry = telemetry          # Store-like sink for per-attempt throttle logs

    # -- single call ------------------------------------------------------
    def complete(self, spec: ModelSpec, messages: list[dict[str, str]], *,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 response_format: Optional[dict] = None,
                 store: Any = None, cache: bool = True) -> Completion:
        run = self.cfg.get("run", {}) or {}
        temp = run.get("temperature", 0.5) if temperature is None else temperature
        maxt = run.get("max_tokens", 1200) if max_tokens is None else max_tokens
        key = content_hash(spec.provider, spec.model, temp, json.dumps(messages, sort_keys=True))

        if cache and store is not None and hasattr(store, "ai_cache_get"):
            hit = store.ai_cache_get(key)
            if hit is not None:
                tin = cost.count_messages(messages, spec.model)
                tout = cost.count_tokens(hit, spec.model)
                return Completion(hit, spec.model, spec.provider, tin, tout,
                                  cost.price(self.cfg, spec.model, tin, tout))

        comp = self._once(spec, messages, temp, maxt, response_format)
        # On failure, try this spec's configured alternates in order (e.g. a free
        # tier returned 429). The winning model is recorded on the Completion, so
        # it surfaces naturally in the transcript.
        if not comp.ok:
            for alt in spec.fallbacks:
                comp = self._once(alt, messages, temp, maxt, response_format)
                if comp.ok:
                    break

        # Cache under the primary key regardless of which model answered, so a
        # repeated identical call hits the cache (and offline replay is stable).
        if comp.ok and cache and store is not None and hasattr(store, "ai_cache_put"):
            store.ai_cache_put(key, comp.model, _last(messages, "user")[:2000], comp.text)
        return comp

    def _once(self, spec: ModelSpec, messages: list[dict[str, str]], temp: float,
              maxt: int, response_format: Optional[dict] = None) -> Completion:
        """One attempt against a single spec (mock or HTTP) -- no fallback, no cache."""
        t0 = time.time()
        if spec.provider == "mock":
            return self._finish(spec, messages, self.mock.respond(spec, messages), t0)
        return self._http(spec, messages, temp, maxt, t0, response_format)

    def _limiter(self, provider: str) -> RateLimiter:
        """The process-wide, shared rate limiter for a provider."""
        return _shared_limiter(provider, self._rpm_for(provider))

    def _rpm_for(self, provider: str) -> float:
        """Per-minute budget for a provider: ``providers.<p>.rpm`` else ``run.rate_limit_rpm``."""
        pconf = provider_conf(self.cfg, provider)
        if pconf.get("rpm") is not None:
            return float(pconf.get("rpm") or 0)
        return float((self.cfg.get("run", {}) or {}).get("rate_limit_rpm", 0) or 0)

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

    def _record(self, spec: ModelSpec, status: str, *, http_code: int = 0, attempt: int = 0,
                a0: float = 0.0, headers: Any = None, retry_after: float = 0.0,
                tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Best-effort telemetry: log one HTTP attempt for throttle analysis."""
        tel = self.telemetry
        if tel is None or not hasattr(tel, "add_api_call"):
            return
        rl_lim, rl_rem, rl_reset = _rl_from_headers(headers)
        latency = int((time.time() - a0) * 1000) if a0 else 0
        try:
            tel.add_api_call(spec.provider, spec.model, status, http_code=http_code,
                             attempt=attempt, latency_ms=latency, retry_after=retry_after,
                             rl_limit=rl_lim, rl_remaining=rl_rem, rl_reset=rl_reset,
                             tokens_in=tokens_in, tokens_out=tokens_out)
        except Exception:  # noqa: BLE001 - telemetry must never break a run
            pass

    def _http(self, spec: ModelSpec, messages: list[dict[str, str]],
              temperature: float, max_tokens: int, t0: float,
              response_format: Optional[dict] = None) -> Completion:
        conf = provider_conf(self.cfg, spec.provider)
        base = (conf.get("base_url", "") or "").rstrip("/")
        if not base:
            return Completion("", spec.model, spec.provider, ok=False,
                              error=f"provider '{spec.provider}' has no base_url")
        body: dict[str, Any] = {
            "model": spec.model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = response_format
        payload = json.dumps(body).encode("utf-8")
        # Send an explicit User-Agent: some provider CDNs (e.g. Groq behind
        # Cloudflare) 403 the default urllib User-Agent as a suspected bot.
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "quorum (+https://github.com/rinz0x0cruz/quorum)",
        }
        k = api_key(self.cfg, spec.provider)
        if k:
            headers["Authorization"] = f"Bearer {k}"

        delay, last_err = self.backoff, ""
        limiter = self._limiter(spec.provider)
        for attempt in range(self.max_retries + 1):
            limiter.acquire()
            a0 = time.time()
            req = urllib.request.Request(base + "/chat/completions", data=payload,
                                         headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - configured endpoint
                    resp_headers = getattr(resp, "headers", None)
                    data = json.loads(resp.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {}) or {}
                comp = self._finish(spec, messages, text, t0,
                                    tokens_out=usage.get("completion_tokens"))
                if usage.get("prompt_tokens"):
                    comp.tokens_in = int(usage["prompt_tokens"])
                    comp.cost_usd = cost.price(self.cfg, spec.model, comp.tokens_in, comp.tokens_out)
                self._record(spec, "ok", http_code=200, attempt=attempt, a0=a0,
                             headers=resp_headers, tokens_in=comp.tokens_in,
                             tokens_out=comp.tokens_out)
                return comp
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}"
                ra = e.headers.get("Retry-After") if e.headers else None
                ra_s = float(ra) if (ra and str(ra).replace(".", "", 1).isdigit()) else 0.0
                self._record(spec, last_err, http_code=e.code, attempt=attempt, a0=a0,
                             headers=e.headers, retry_after=ra_s)
                # A 429 is a per-minute cap that a few seconds of backoff won't clear,
                # so retry it at most `retry_429` times, then fail so complete() can
                # rotate to a fallback model (which has its own separate limit).
                budget = self.retry_429 if e.code == 429 else self.max_retries
                if e.code in (429, 500, 502, 503, 504) and attempt < budget:
                    time.sleep(min(ra_s or delay, 15))
                    delay *= 2
                    continue
                return Completion("", spec.model, spec.provider, ok=False,
                                  error=last_err, latency_ms=int((time.time() - t0) * 1000))
            except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as e:
                last_err = str(e)
                status = "timeout" if isinstance(e, TimeoutError) else "error"
                self._record(spec, status, http_code=0, attempt=attempt, a0=a0)
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                return Completion("", spec.model, spec.provider, ok=False,
                                  error=last_err, latency_ms=int((time.time() - t0) * 1000))
        return Completion("", spec.model, spec.provider, ok=False, error=last_err)

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


def for_config(cfg: dict, *, mock: Optional[MockResponder] = None, store: Any = None) -> Provider:
    return Provider(cfg, mock=mock, telemetry=store)


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
