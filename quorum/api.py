"""Embeddable API -- use quorum as the AI backend inside another tool.

Instead of a single generic model call, a host tool can route its enrichment
through a quorum deliberation (self-refine by default, or a full council). It is
built to drop into the sibling tools' ``ai.chat(cfg, store, system, user)``
contract and stays **optional**: if disabled, unconfigured, or if quorum is not
installed, the helpers return ``None`` so the host degrades to its own behavior
(honoring the "AI is always optional" rule).

Host config -- add a ``quorum:`` block to the tool's existing ``config.yaml``::

    quorum:
      enabled: true            # off by default -> AI stays optional
      strategy: refine         # refine | debate | council | moa | ensemble
      max_rounds: 2
      # Optional extra council members; if omitted, the tool's single `ai.model`
      # is used (so even one model gains self-refine):
      members:
        - { name: a, provider: openrouter, model: google/gemma-4-31b-it:free }
        - { name: b, provider: openrouter, model: openai/gpt-oss-120b:free }
      providers:
        openrouter: { base_url: https://openrouter.ai/api/v1, api_key_env: TOOL_OPENROUTER_KEY }

The tool's existing ``ai:`` block supplies the default provider/model/key, so in
the simplest case you only add ``quorum: {enabled: true}``.

Usage inside a host tool's ``ai.py``::

    def chat(cfg, store, system, user, *, temperature=None):
        try:
            from quorum.api import chat as _q
            out = _q(cfg, store, system, user, temperature=temperature)
            if out is not None:
                return out
        except ImportError:
            pass
        ...  # existing single-model path, unchanged
"""
from __future__ import annotations

import os
from typing import Any, Optional

from .config import DEFAULT_CONFIG, _deep_merge


def enabled(cfg: dict) -> bool:
    """True when the host opted in AND a usable key/provider is present."""
    q = cfg.get("quorum", {}) or {}
    if not q.get("enabled"):
        return False
    ai = cfg.get("ai", {}) or {}
    prov = (ai.get("provider") or "").lower()
    if prov in ("ollama", "mock"):
        return True
    if q.get("members") or q.get("providers"):
        return True  # explicit council -> trust the host to have set its keys
    env = ai.get("api_key_env", "")
    return bool(env and os.environ.get(env))


def build_config(cfg: dict) -> dict:
    """Compose a quorum config from the host tool's ``ai:`` + optional ``quorum:``."""
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


def deliberate(task: str, *, system: Optional[str] = None, cfg: Optional[dict] = None,
               store: Any = None, strategy: Optional[str] = None) -> Optional[str]:
    """Run a deliberation and return the final answer text (or ``None`` on failure).

    ``system`` is used verbatim as the solve-instruction (promptsmith is skipped),
    so the host's own prompt fully drives the deliberation.
    """
    from . import orchestrator
    qcfg = build_config(cfg or {})
    if strategy:
        qcfg["run"]["strategy"] = strategy
    sess = orchestrator.run_session(qcfg, task, store=store, solve_prompt=system or "",
                                    promptsmith_on=False, verbose=False)
    if sess.status != "ok" or not (sess.final or "").strip():
        return None
    return sess.final


def chat(cfg: dict, store: Any, system: str, user: str, *,
         temperature: Optional[float] = None) -> Optional[str]:
    """Drop-in for a sibling tool's ``ai.chat`` -- deliberate, or ``None`` if off.

    Signature-compatible so a host's ``ai.chat`` can delegate here first and fall
    back to its own single-model path when this returns ``None``.
    """
    if not enabled(cfg):
        return None
    if temperature is not None:
        ai = dict(cfg.get("ai") or {})
        ai["temperature"] = temperature
        cfg = {**cfg, "ai": ai}
    return deliberate(user, system=system, cfg=cfg, store=store)
