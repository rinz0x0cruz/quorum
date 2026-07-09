"""Deliberation strategies.

Each strategy is a callable ``run(ctx: Context) -> Session`` that fills
``ctx.session`` with rounds and a final answer. Built-ins are resolved lazily;
third-party strategies can register under the ``quorum.strategies`` entry-point
group (see pyproject) and are picked up automatically when installed.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

from ..model import ModelSpec, Session

_BUILTIN = {
    "debate": "quorum.strategies.debate",
    "council": "quorum.strategies.council",
    "moa": "quorum.strategies.moa",
    "refine": "quorum.strategies.refine",
    "ensemble": "quorum.strategies.ensemble",
    "selfconsistency": "quorum.strategies.selfconsistency",
    "reflexion": "quorum.strategies.reflexion",
    "verify": "quorum.strategies.verify",
    "cascade": "quorum.strategies.cascade",
}


@dataclass
class RunOptions:
    """The ``run.*`` knobs resolved once, so strategies read fields, not a dict.

    Parsed in the orchestrator from the effective config (after any CLI/arg
    overrides) and hung on :class:`Context`. Add a new run-level knob here and in
    ``from_cfg``, and every strategy can use it without re-parsing config.
    """

    strategy: str = "refine"
    max_rounds: int = 4
    target_score: float = 85.0
    plateau_delta: float = 2.0
    plateau_patience: int = 2
    consensus: bool = False
    moa_layers: int = 2
    samples: int = 3
    samples_min: int = 2
    adaptive_samples: bool = False
    temperature: float = 0.5
    max_tokens: int = 1200
    judge_every: int = 1
    anonymize: bool = True
    parallel: bool = True
    top_k: int = 0
    devils_advocate: bool = False
    cascade: list = field(default_factory=list)  # ordered strategies for the cascade escalation

    @classmethod
    def from_cfg(cls, cfg: dict) -> "RunOptions":
        r = cfg.get("run", {}) or {}
        return cls(
            strategy=r.get("strategy", "refine"),
            max_rounds=int(r.get("max_rounds", 4)),
            target_score=float(r.get("target_score", 85)),
            plateau_delta=float(r.get("plateau_delta", 2)),
            plateau_patience=int(r.get("plateau_patience", 2)),
            consensus=bool(r.get("consensus", False)),
            moa_layers=max(1, int(r.get("moa_layers", 2))),
            samples=max(1, int(r.get("samples", 3))),
            samples_min=max(1, int(r.get("samples_min", 2))),
            adaptive_samples=bool(r.get("adaptive_samples", False)),
            temperature=float(r.get("temperature", 0.5)),
            max_tokens=int(r.get("max_tokens", 1200)),
            judge_every=max(1, int(r.get("judge_every", 1))),
            anonymize=bool(r.get("anonymize", True)),
            parallel=bool(r.get("parallel", True)),
            top_k=int(r.get("top_k", 0) or 0),
            devils_advocate=bool(r.get("devils_advocate", False)),
            cascade=list(r.get("cascade", []) or []),
        )


@dataclass
class Context:
    cfg: dict
    prov: Any                      # provider.Provider
    store: Any                     # store.Store | None
    task: str
    prompt: str                    # refined solve-prompt (phase 1 output)
    members: list[ModelSpec]
    session: Session
    emit: Callable[[Any], None]    # accepts a str (log line) or an events.Event
    opts: RunOptions = field(default_factory=RunOptions)  # resolved run.* knobs

    def event(self, kind: str, message: str = "", *, round: int = 0, **data: Any) -> None:
        """Emit a structured :class:`~quorum.events.Event` (also rendered to the CLI)."""
        from ..events import Event
        self.emit(Event(kind, message, round=round, data=data))


def available() -> list[str]:
    names = set(_BUILTIN)
    names.update(_entry_points().keys())
    return sorted(names)


def get(name: str) -> Callable[[Context], Session]:
    eps = _entry_points()
    if name in eps:
        return eps[name].load()
    if name in _BUILTIN:
        return importlib.import_module(_BUILTIN[name]).run
    raise KeyError(f"unknown strategy '{name}' (have: {', '.join(available())})")


def _entry_points() -> dict[str, Any]:
    try:
        from importlib.metadata import entry_points
        eps = entry_points()
        group = eps.select(group="quorum.strategies") if hasattr(eps, "select") \
            else eps.get("quorum.strategies", [])  # py<3.10 shape
        return {ep.name: ep for ep in group}
    except Exception:  # noqa: BLE001 - entry points are optional
        return {}
