"""Dependency-free lexical text-scoring primitives.

The exact math previously inlined in the callers, centralised so the two
*different* similarity measures stay named and impossible to conflate:

* :func:`overlap_coeff` -- asymmetric ``|q & d| / |q|`` (the fraction of the
  query's terms the document covers); how :func:`quorum.contextwindow.select`
  ranks grounding docs.
* :func:`jaccard` -- symmetric ``|a & b| / |a | b|``; how
  :func:`quorum.judge.consensus_reached` measures member convergence.

Both share one tokenizer (:func:`tokens`). Everything here is pure stdlib and
deterministic -- this module never touches the network and is a *leaf* helper: it
imports nothing from quorum above it.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")


def tokens(s: str) -> set[str]:
    """Lowercase ``[a-z0-9]+`` token *set* -- the tokenizer used across quorum."""
    return set(_WORD.findall((s or "").lower()))


def overlap_coeff(query: set[str], doc: set[str]) -> float:
    """Overlap coefficient ``|query & doc| / |query|`` in ``0..1``.

    Asymmetric: the fraction of the query's terms the document covers; ``0.0``
    when ``query`` is empty. (Identical math to the former ``contextwindow._overlap``.)
    """
    return (len(query & doc) / len(query)) if query else 0.0


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity ``|a & b| / |a | b|`` in ``0..1``.

    Symmetric; ``0.0`` when both sets are empty. Deliberately *distinct* from
    :func:`overlap_coeff` -- do not collapse the two. (The formula
    :func:`quorum.judge.consensus_reached` averages pairwise.)
    """
    union = a | b
    return (len(a & b) / len(union)) if union else 0.0


class LexicalScorer:
    """A :class:`~quorum.scoring.Scorer` that scores ``text`` by lexical
    :func:`overlap_coeff` with ``query`` -- registered under the name ``"lexical"``.

    Structurally implements the ``Scorer`` protocol (a ``score`` method); it does
    not import it, keeping this module a dependency-free leaf.
    """

    name = "lexical"

    def score(self, query: str, text: str) -> float:
        return overlap_coeff(tokens(query), tokens(text))
