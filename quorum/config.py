"""Configuration loading for quorum.

A single ``DEFAULT_CONFIG`` dict is deep-merged with an optional user file
(``config.yaml`` or ``config.json``). Secrets never live in the config file --
each provider names an environment variable that holds its key (optionally loaded
from a local ``.env``). Mirrors the claudebudget/jobscope/learnscope config
system so the whole tool family behaves the same way.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .model import ModelSpec

DEFAULT_CONFIG: dict[str, Any] = {
    # The council: any number of models, across any providers below. By default a
    # trio of *free* OpenRouter models -- edit freely (add a local Ollama model,
    # OpenAI, Groq, ...). These are examples; you own the roster.
    "council": {
        "members": [
            {"name": "alice", "provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free"},
            {"name": "bob",   "provider": "openrouter", "model": "qwen/qwen3-next-80b-a3b-instruct:free"},
            {"name": "carol", "provider": "openrouter", "model": "google/gemma-4-31b-it:free"},
        ],
        # Single-model roles (referenced as "provider:model"). Empty -> first member.
        "judge": "openrouter:openai/gpt-oss-120b:free",
        "chairman": "openrouter:openai/gpt-oss-120b:free",
        "aggregator": "openrouter:openai/gpt-oss-120b:free",
    },

    # Provider profiles: any OpenAI-compatible /chat/completions endpoint. The
    # built-in "mock" provider is fully offline (used by selftest + replay).
    "providers": {
        "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "QUORUM_OPENROUTER_KEY"},
        "openai":     {"base_url": "https://api.openai.com/v1",    "api_key_env": "QUORUM_OPENAI_KEY"},
        "groq":       {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "QUORUM_GROQ_KEY"},
        "ollama":     {"base_url": "http://localhost:11434/v1",    "api_key_env": ""},  # local, keyless
        "mock":       {"base_url": "", "api_key_env": ""},                              # offline
    },

    "run": {
        "strategy": "refine",       # debate | council | moa | refine | ensemble
        "max_rounds": 4,            # hard cap on deliberation rounds
        "target_score": 85,         # 0-100 "good enough" threshold
        "plateau_delta": 2,         # stop if the best score gains < this ...
        "plateau_patience": 2,      # ... for this many consecutive rounds
        "consensus": False,         # also stop when members converge on one answer
        "moa_layers": 2,            # layers for the mixture-of-agents strategy
        "samples": 3,               # samples for the ensemble baseline
        "temperature": 0.5,
        "max_tokens": 1200,
        "anonymize": True,          # hide model identities during peer review
        "parallel": True,           # fan proposer calls out concurrently
    },

    # Phase 1: design + refine the solve-prompt before the council answers.
    "promptsmith": {"enabled": True, "rounds": 2},

    "judge": {
        # Weighted 0-100 rubric; weights need not sum to 1 (they are normalised).
        "rubric": {"correctness": 0.40, "completeness": 0.25, "clarity": 0.20, "grounding": 0.15},
        "cross_family_guard": True,  # prefer a judge from a different vendor than the candidate
    },

    "cost": {
        "budget_usd": 0.50,          # abort a run if projected spend exceeds this (0 = no cap)
        # Per-model price overrides, USD per 1M tokens, e.g.:
        #   "openai/gpt-4o": {"input": 2.5, "output": 10}
        "pricing": {},
    },

    "output": {
        "db_path": "data/quorum.db",
        "dashboard_path": "data/dashboard.html",
    },
}

CONFIG_CANDIDATES = ("config.yaml", "config.yml", "config.json")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    import copy
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency). Existing env vars win."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _parse_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith(".json"):
            return json.load(fh)
        try:
            import yaml  # lazy: only needed for YAML configs
        except ImportError as exc:  # pragma: no cover - guidance path
            raise SystemExit(
                "PyYAML is required to read YAML config. Install with "
                "`pip install pyyaml`, or use config.json instead."
            ) from exc
        return yaml.safe_load(fh) or {}


def load_config(path: str | None = None) -> dict[str, Any]:
    """Return the effective config (defaults deep-merged with a user file)."""
    _load_dotenv()
    if path is None:
        for candidate in CONFIG_CANDIDATES:
            if os.path.exists(candidate):
                path = candidate
                break
    if path and os.path.exists(path):
        return _deep_merge(DEFAULT_CONFIG, _parse_file(path))
    return json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------
def provider_conf(cfg: dict, name: str) -> dict[str, Any]:
    return (cfg.get("providers", {}) or {}).get(name, {}) or {}


def api_key(cfg: dict, provider: str) -> str:
    """Resolve a provider's API key from its configured environment variable."""
    env = provider_conf(cfg, provider).get("api_key_env", "")
    return os.environ.get(env, "") if env else ""


def parse_ref(ref: str) -> tuple[str, str]:
    """Split a ``provider:model`` reference (only the first colon separates)."""
    provider, _, model = (ref or "").partition(":")
    return provider, model


def member_specs(cfg: dict) -> list[ModelSpec]:
    members = (cfg.get("council", {}) or {}).get("members", []) or []
    specs: list[ModelSpec] = []
    for i, m in enumerate(members):
        specs.append(ModelSpec(
            name=m.get("name") or f"m{i + 1}",
            provider=m.get("provider") or "mock",
            model=m.get("model") or "",
            role="proposer",
        ))
    return specs


def role_spec(cfg: dict, role: str) -> ModelSpec:
    """Resolve a single-model role (judge/chairman/aggregator).

    Falls back to the first council member when the role is unset.
    """
    ref = (cfg.get("council", {}) or {}).get(role, "") or ""
    if ref:
        provider, model = parse_ref(ref)
        return ModelSpec(name=role, provider=provider, model=model, role=role)
    members = member_specs(cfg)
    if members:
        first = members[0]
        return ModelSpec(name=role, provider=first.provider, model=first.model, role=role)
    return ModelSpec(name=role, provider="mock", model="mock-model", role=role)
