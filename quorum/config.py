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
import sys
from typing import Any

from .model import ModelSpec

DEFAULT_CONFIG: dict[str, Any] = {
    # The council: any number of models, across any providers below. By default a
    # trio of *free* OpenRouter models -- edit freely (add a local Ollama model,
    # OpenAI, Groq, ...). These are examples; you own the roster.
    "council": {
        "members": [
            {"name": "alice", "provider": "openrouter", "model": "nvidia/nemotron-3-ultra-550b-a55b:free"},
            {"name": "bob",   "provider": "openrouter", "model": "google/gemma-4-31b-it:free"},
            {"name": "carol", "provider": "openrouter", "model": "openai/gpt-oss-20b:free"},
        ],
        # Single-model roles (referenced as "provider:model"). Empty -> first member.
        "judge": "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free",
        "chairman": "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free",
        "aggregator": "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free",
    },

    # Provider profiles: any OpenAI-compatible /chat/completions endpoint. The
    # built-in "mock" provider is fully offline (used by selftest + replay).
    "providers": {
        "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "QUORUM_OPENROUTER_KEY"},
        "openai":     {"base_url": "https://api.openai.com/v1",    "api_key_env": "QUORUM_OPENAI_KEY"},
        "groq":       {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "QUORUM_GROQ_KEY",
                   "model_options": {}},
        "ollama":     {"base_url": "http://localhost:11434/v1",    "api_key_env": ""},  # local, keyless
        "mock":       {"base_url": "", "api_key_env": ""},                              # offline
    },

    # Optional provider-catalog discovery. Syncing is always explicit; normal
    # runs consume only the pinned model ids configured above.
    "catalog": {
        "enabled": False,
        "cache_dir": "data/catalogs",
        "providers": {},
        "allow_preview": False,
        "expiry_horizon_days": 14,
        "access_classes": ["zero_price", "free_quota", "local"],
    },

    # Reproducible model/profile evaluation. The existing ``bench`` command
    # remains strategy-centric; this section is inert until the eval workflow is
    # explicitly invoked.
    "evaluation": {
        "enabled": False,
        "packs_dir": "evals",
        "repeats": 1,
        "min_promotion_samples": 30,
        "min_served_rate": 0.95,
        "bootstrap_samples": 2000,
    },

    # Named, pinned use-case profiles. Definitions are user-owned overlays and
    # are never selected automatically while profiles/routing are disabled.
    "profiles": {
        "enabled": False,
        "active": "",
        "paths": [],
        "definitions": {},
    },

    # Selection precedence will be explicit hint -> deterministic rules -> an
    # optional learned router -> the unchanged run.strategy/model defaults.
    "routing": {
        "enabled": False,
        "rules": [],
        "learned": False,
        "artifact": "",
        "min_confidence": 0.8,
    },

    # ``single`` exactly preserves today's one-judge behavior. Jury and peer
    # modes are opt-in and will use family-balanced ballots when implemented.
    "decision": {
        "mode": "single",
        "jurors": [],
        "min_ballots": 3,
        "min_families": 3,
        "preserve_dissent": True,
    },

    # Training remains an isolated optional workflow. The core process never
    # imports GPU/training libraries; named backends describe subprocess/plugin
    # runners and are disabled by default.
    "tune": {
        "enabled": False,
        "backend": "mock",
        "backends": {},
        "output_dir": "data/tuning",
        "python": "",
    },

    "run": {
        "strategy": "refine",       # debate | council | moa | refine | ensemble
        "max_rounds": 4,            # hard cap on deliberation rounds
        "target_score": 85,         # 0-100 "good enough" threshold
        "plateau_delta": 2,         # stop if the best score gains < this ...
        "plateau_patience": 2,      # ... for this many consecutive rounds
        "consensus": False,         # also stop when members converge on one answer
        "moa_layers": 2,            # layers for the mixture-of-agents strategy
        "samples": 3,               # samples for the ensemble baseline (max, when adaptive)
        "samples_min": 2,           # ensemble: min samples before an adaptive early-stop
        "adaptive_samples": False,  # ensemble: sample incrementally + stop on a confident majority
        "temperature": 0.5,
        "max_tokens": 1200,
        "judge_every": 1,          # judge every N rounds (1=every round; >1 saves judge calls in debate/council/refine)
        "anonymize": True,          # hide model identities during peer review
        "parallel": True,           # fan proposer calls out concurrently
        "rate_limit_rpm": 0,        # pace HTTP calls to <= this many/min per provider (0 = off; ~18 for free OpenRouter)
        "fallbacks": [],            # default alternates ("provider:model") tried when a call fails (e.g. 429)
        "top_k": 0,                 # if >0, fuse only the top-K peer-ranked candidates (council/moa)
        "devils_advocate": False,   # debate: have one member argue the counter-case from round 2 on
        "cascade": [],              # strategy=cascade: ordered stages, cheapest first (default [refine,debate,council])
    },

    # Phase 1: design + refine the solve-prompt before the council answers.
    #   bootstrap -> seed the prompt engineer with instructions from past high-scoring sessions.
    "promptsmith": {"enabled": True, "rounds": 2, "bootstrap": False},

    "judge": {
        # Weighted 0-100 rubric; weights need not sum to 1 (they are normalised).
        "rubric": {"correctness": 0.40, "completeness": 0.25, "clarity": 0.20, "grounding": 0.15},
        "cross_family_guard": True,  # prefer a judge from a different vendor than the candidate
        "json_mode": False,          # ask the judge/grader endpoint for OpenAI JSON mode (opt-in)
        "shuffle_candidates": True,  # randomise candidate order to curb LLM-judge position bias
    },

    "cost": {
        "budget_usd": 0.50,          # abort a run if projected spend exceeds this (0 = no cap)
        # Per-model price overrides, USD per 1M tokens, e.g.:
        #   "openai/gpt-4o": {"input": 2.5, "output": 10}
        "pricing": {},
    },

    # Optional grounding for callers that need memory (chatbot conversation
    # history, or feeding prior documents back in). Inert unless a caller passes
    # history/docs; these knobs just bound how much gets injected.
    "context": {
        "budget_tokens": 4000,       # max tokens of injected history + docs
        "history_turns": 8,          # max prior conversation messages kept (most recent)
        "top_k": 5,                  # max grounding docs the lexical selector keeps
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


def load_config(path: str | None = None, *, warn: bool = False) -> dict[str, Any]:
    """Return the effective config (defaults deep-merged with a user file).

    With ``warn=True``, unknown/mistyped keys in the user file are printed to
    stderr (never fatal) -- a light guard so a typo'd knob does not silently no-op.
    """
    _load_dotenv()
    if path is None:
        for candidate in CONFIG_CANDIDATES:
            if os.path.exists(candidate):
                path = candidate
                break
    if path and os.path.exists(path):
        user = _parse_file(path)
        if warn:
            for key in validate_config(user):
                print(f"  warning: unknown config key '{key}' (ignored)", file=sys.stderr)
        return _deep_merge(DEFAULT_CONFIG, user)
    return json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy


# Subtrees whose keys are user-defined (provider/profile/backend names, model
# prices, rubric criteria) and so must not be flagged as unknown.
_OPEN_SUBTREES = (
    "providers",
    "catalog.providers",
    "profiles.definitions",
    "tune.backends",
    "cost.pricing",
    "judge.rubric",
)


def _validate_walk(user: Any, default: Any, prefix: str, out: list[str]) -> None:
    if not isinstance(user, dict):
        return
    for k, v in user.items():
        path = f"{prefix}.{k}" if prefix else k
        if path in _OPEN_SUBTREES or any(path.startswith(s + ".") for s in _OPEN_SUBTREES):
            continue
        if not isinstance(default, dict) or k not in default:
            out.append(path)
            continue
        if isinstance(v, dict) and isinstance(default.get(k), dict):
            _validate_walk(v, default[k], path, out)


def validate_config(user: dict) -> list[str]:
    """Return the unknown key-paths in a user config (typos). Never raises.

    Open subtrees (``providers.*``, ``cost.pricing.*``, ``judge.rubric.*``) allow
    arbitrary user-defined keys and are not reported.
    """
    out: list[str] = []
    _validate_walk(user, DEFAULT_CONFIG, "", out)
    return out


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


def _parse_fallbacks(refs: Any) -> list[ModelSpec]:
    """Build leaf :class:`ModelSpec` alternates from ``provider:model`` refs."""
    out: list[ModelSpec] = []
    for ref in (refs or []):
        provider, model = parse_ref(ref)
        if model:
            out.append(ModelSpec(name=f"fallback:{model}", provider=provider, model=model))
    return out


def member_specs(cfg: dict) -> list[ModelSpec]:
    members = (cfg.get("council", {}) or {}).get("members", []) or []
    default_fb = (cfg.get("run", {}) or {}).get("fallbacks", []) or []
    specs: list[ModelSpec] = []
    for i, m in enumerate(members):
        own_fb = m.get("fallbacks") if isinstance(m, dict) else None
        specs.append(ModelSpec(
            name=m.get("name") or f"m{i + 1}",
            provider=m.get("provider") or "mock",
            model=m.get("model") or "",
            role="proposer",
            fallbacks=_parse_fallbacks(own_fb if own_fb is not None else default_fb),
        ))
    return specs


def role_spec(cfg: dict, role: str) -> ModelSpec:
    """Resolve a single-model role (judge/chairman/aggregator).

    Falls back to the first council member when the role is unset. Operational
    alternates come from ``council.<role>_fallbacks`` or, failing that, the global
    ``run.fallbacks`` default.
    """
    council = cfg.get("council", {}) or {}
    default_fb = (cfg.get("run", {}) or {}).get("fallbacks", []) or []
    role_fb = council.get(f"{role}_fallbacks")
    fb = _parse_fallbacks(role_fb if role_fb is not None else default_fb)
    ref = council.get(role, "") or ""
    if ref:
        provider, model = parse_ref(ref)
        return ModelSpec(name=role, provider=provider, model=model, role=role, fallbacks=fb)
    members = member_specs(cfg)
    if members:
        first = members[0]
        return ModelSpec(name=role, provider=first.provider, model=first.model, role=role, fallbacks=fb)
    return ModelSpec(name=role, provider="mock", model="mock-model", role=role, fallbacks=fb)
