"""Text scoring: shared lexical primitives + a pluggable ``Scorer`` registry.

quorum measures text several *different* ways in a few places: lexical overlap to
rank grounding docs (:mod:`quorum.contextwindow`), token Jaccard to detect member
convergence (:func:`quorum.judge.consensus_reached`), and -- via the LLM -- rubric
scoring (:mod:`quorum.judge`) and reference grading (:mod:`quorum.grade`). This
package unifies the *dependency-free lexical* primitives so the two distinct
measures (overlap coefficient vs Jaccard) are named once, tested once, and never
conflated.

A :class:`Scorer` protocol plus a small registry -- mirroring the
``quorum.strategies`` plugin pattern, discovering the ``quorum.scorers``
entry-point group -- lets richer scorers (the rubric-LLM judge, the
reference-numeric grader, or a future embedding scorer) register under a name
later without touching callers.

Layering: this is a *leaf* reasoning helper. It imports only the stdlib and must
not import strategies, the orchestrator, or the provider. Deterministic and
offline.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .text import LexicalScorer, jaccard, overlap_coeff, tokens

__all__ = [
    "Scorer",
    "LexicalScorer",
    "tokens",
    "overlap_coeff",
    "jaccard",
    "register",
    "get",
    "available",
]


@runtime_checkable
class Scorer(Protocol):
    """Scores how well ``text`` answers ``query`` on a ``0..1`` scale."""

    def score(self, query: str, text: str) -> float:  # pragma: no cover - protocol
        ...


_REGISTRY: dict[str, Scorer] = {}


def register(name: str, scorer: Scorer) -> None:
    """Register ``scorer`` under ``name`` (a later registration replaces an earlier)."""
    _REGISTRY[name] = scorer


def get(name: str) -> Scorer:
    """Return the scorer registered as ``name``.

    Checks in-process registrations first (the built-in ``lexical`` and anything a
    host registered), then installed ``quorum.scorers`` entry-point plugins. An
    entry point resolving to a class is instantiated; one resolving to an instance
    is used as-is.
    """
    if name in _REGISTRY:
        return _REGISTRY[name]
    eps = _entry_points()
    if name in eps:
        obj = eps[name].load()
        return obj() if isinstance(obj, type) else obj
    raise KeyError(f"unknown scorer '{name}' (have: {', '.join(available())})")


def available() -> list[str]:
    """Sorted names of every registered + entry-point-discoverable scorer."""
    names = set(_REGISTRY)
    names.update(_entry_points().keys())
    return sorted(names)


def _entry_points() -> dict[str, Any]:
    """Discover third-party scorers in the ``quorum.scorers`` group.

    Mirrors ``quorum.strategies._entry_points``; entry points are optional, so any
    failure degrades cleanly to the in-process registry.
    """
    try:
        from importlib.metadata import entry_points
        eps = entry_points()
        group = eps.select(group="quorum.scorers") if hasattr(eps, "select") \
            else eps.get("quorum.scorers", [])  # py<3.10 shape
        return {ep.name: ep for ep in group}
    except Exception:  # noqa: BLE001 - entry points are optional
        return {}


# Built-in lexical scorer, always available offline (no install/entry point needed).
register("lexical", LexicalScorer())
