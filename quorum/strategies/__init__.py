"""Deliberation strategies.

Each strategy is a callable ``run(ctx: Context) -> Session`` that fills
``ctx.session`` with rounds and a final answer. Built-ins are resolved lazily;
third-party strategies can register under the ``quorum.strategies`` entry-point
group (see pyproject) and are picked up automatically when installed.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

from ..model import ModelSpec, Session

_BUILTIN = {
    "debate": "quorum.strategies.debate",
    "council": "quorum.strategies.council",
    "moa": "quorum.strategies.moa",
    "refine": "quorum.strategies.refine",
    "ensemble": "quorum.strategies.ensemble",
}


@dataclass
class Context:
    cfg: dict
    prov: Any                      # provider.Provider
    store: Any                     # store.Store | None
    task: str
    prompt: str                    # refined solve-prompt (phase 1 output)
    members: list[ModelSpec]
    session: Session
    emit: Callable[[str], None]


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
