"""Token + cost accounting.

Token counts use ``tiktoken`` when installed (``pip install quorum[tokens]``) and
otherwise fall back to a ~4-chars-per-token heuristic, so the engine has zero hard
dependency on it. Prices are USD per 1M tokens; a small built-in table covers a
few common models and is overridden/extended by ``cost.pricing`` in config. An
unknown model prices at 0 (so free/local models don't inflate the budget).
"""
from __future__ import annotations

from typing import Any

# USD per 1M tokens. Override or extend via cfg["cost"]["pricing"].
_DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "openai/gpt-4o": {"input": 2.5, "output": 10.0},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "openai/o1": {"input": 15.0, "output": 60.0},
    "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "google/gemini-1.5-pro": {"input": 1.25, "output": 5.0},
    "google/gemini-1.5-flash": {"input": 0.075, "output": 0.3},
}


def count_tokens(text: str, model: str = "") -> int:
    if not text:
        return 0
    try:  # optional accurate path
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model.split("/")[-1])
        except Exception:  # noqa: BLE001 - unknown model -> generic encoding
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001 - tiktoken absent -> heuristic
        return max(1, len(text) // 4)


def count_messages(messages: list[dict[str, Any]], model: str = "") -> int:
    return sum(count_tokens(m.get("content", ""), model) for m in messages)


def _lookup(cfg: dict, model: str) -> dict[str, float] | None:
    table = dict(_DEFAULT_PRICES)
    table.update((cfg.get("cost", {}) or {}).get("pricing", {}) or {})
    if model in table:
        return table[model]
    for k, v in table.items():  # loose match (handles vendor prefixes / suffixes)
        if k and (k in model or model in k):
            return v
    return None


def price(cfg: dict, model: str, tokens_in: int, tokens_out: int) -> float:
    p = _lookup(cfg, model or "")
    if not p:
        return 0.0
    return (tokens_in / 1e6) * p.get("input", 0.0) + (tokens_out / 1e6) * p.get("output", 0.0)


def budget_usd(cfg: dict) -> float:
    return float((cfg.get("cost", {}) or {}).get("budget_usd", 0) or 0)


def over_budget(cfg: dict, spent: float) -> bool:
    cap = budget_usd(cfg)
    return cap > 0 and spent > cap
