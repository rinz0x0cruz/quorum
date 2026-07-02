"""Core data models for quorum.

Plain dataclasses shared across the provider layer, judge, strategies, store,
renderer, and benchmark. Everything is JSON-serialisable (``to_dict``) so a whole
deliberation ``Session`` can be persisted as one blob and replayed offline.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# --------------------------------------------------------------------------
# time + id helpers
# --------------------------------------------------------------------------
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def iso_from_epoch(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def content_hash(*parts: Any) -> str:
    """Stable short hash of the given parts (used for cache keys + ids)."""
    h = hashlib.sha256()
    h.update("\x00".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:32]


def session_id(task: str, strategy: str, epoch: Optional[float] = None) -> str:
    return "s-" + content_hash(task, strategy, epoch if epoch is not None else time.time())[:12]


# --------------------------------------------------------------------------
# model-vendor mapping (for the cross-family judge guard)
# --------------------------------------------------------------------------
_VENDOR_HINTS: dict[str, tuple[str, ...]] = {
    "openai": ("gpt", "o1", "o3", "o4", "davinci", "chatgpt"),
    "anthropic": ("claude",),
    "google": ("gemini", "gemma", "palm"),
    "meta": ("llama",),
    "mistral": ("mistral", "mixtral", "codestral"),
    "alibaba": ("qwen",),
    "xai": ("grok",),
    "deepseek": ("deepseek",),
    "microsoft": ("phi",),
    "cohere": ("command",),
}


def model_vendor(model: str) -> str:
    """Coarse vendor for a model id, e.g. ``anthropic/claude-3.5`` -> ``anthropic``.

    Used only to let the judge avoid scoring a candidate produced by its own
    vendor when ``judge.cross_family_guard`` is on (a fairness caveat raised by
    the multi-agent-debate literature).
    """
    m = (model or "").lower()
    for vendor, hints in _VENDOR_HINTS.items():
        if any(h in m for h in hints):
            return vendor
    return "other"


# --------------------------------------------------------------------------
# dataclasses
# --------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """One configured council member (or role) and where its API lives."""

    name: str                 # friendly label, e.g. "alice" / "judge"
    provider: str             # provider profile key (openrouter, ollama, mock, ...)
    model: str                # model id sent to the endpoint
    role: str = "proposer"    # proposer | judge | chairman | aggregator

    def ref(self) -> str:
        return f"{self.provider}:{self.model}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Turn:
    """A single model utterance within a round."""

    round: int
    member: str               # ModelSpec.name, or a role name
    model: str
    kind: str                 # propose|critique|revise|review|synthesize|aggregate|judge|promptsmith
    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    ts: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    """The judge's assessment of a round's best candidate."""

    round: int
    score: float                                  # 0-100
    sub_scores: dict[str, float] = field(default_factory=dict)
    best_label: str = ""                          # which candidate won
    best_content: str = ""
    rationale: str = ""
    stop: bool = False
    reason: str = ""                              # why we stopped (or "")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Round:
    index: int
    turns: list[Turn] = field(default_factory=list)
    verdict: Optional[Verdict] = None
    best_content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "turns": [t.to_dict() for t in self.turns],
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "best_content": self.best_content,
        }


@dataclass
class Session:
    """A full deliberation: the task, the refined prompt, every round, the final."""

    id: str
    task: str
    strategy: str
    prompt: str = ""                              # refined solve-prompt (phase 1 output)
    rounds: list[Round] = field(default_factory=list)
    final: str = ""
    final_score: float = 0.0
    status: str = "ok"                            # ok | aborted | error
    stop_reason: str = ""
    created: str = field(default_factory=now_iso)
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "strategy": self.strategy,
            "prompt": self.prompt,
            "rounds": [r.to_dict() for r in self.rounds],
            "final": self.final,
            "final_score": self.final_score,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "created": self.created,
            "cost_usd": round(self.cost_usd, 6),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }

    def metrics(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "score": round(self.final_score, 2),
            "rounds": len(self.rounds),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens": self.tokens_in + self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
        }

    def account(self, turn: Turn) -> None:
        """Fold a turn's token/cost usage into the session totals."""
        self.tokens_in += turn.tokens_in
        self.tokens_out += turn.tokens_out
        self.cost_usd += turn.cost_usd
