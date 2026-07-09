"""Answer clustering + adaptive sampling stop (the self-consistency family).

Groups sampled answers into equivalence buckets so a majority vote can pick the
consensus, and offers a lightweight stopping rule so sampling can halt early once
a confident majority emerges -- Adaptive-Consistency (Aggarwal et al., EMNLP
2023), which reaches the same answer as a fixed-N vote with far fewer samples.

Numeric answers bucket by their final number (exact); free-form answers bucket by
lexical similarity (token Jaccard). Both are dependency-free and offline, so the
mock provider exercises them in selftest. A stronger free-form selector (USC --
ask a model to pick the most consistent answer) can layer on top later.
"""
from __future__ import annotations

from typing import Optional

from . import grade, scoring

_SIM = 0.8          # token Jaccard >= this => same free-form answer bucket
_THRESHOLD = 0.6    # stop when the top bucket holds >= this share of votes ...
_MARGIN = 2         # ... and leads the runner-up by at least this many votes


def _key(text: str) -> str:
    """A bucket key from a numeric final answer, or '' for free-form."""
    n = grade.final_number(text)
    return f"#{n}" if n is not None else ""


def assign(clusters: list[dict], text: str, *, sim_threshold: float = _SIM) -> list[dict]:
    """Add ``text`` to a matching bucket (or open a new one). Mutates + returns clusters."""
    if not text:
        return clusters
    key = _key(text)
    if key:  # numeric: exact bucket by the final number
        for c in clusters:
            if c.get("key") == key:
                c["members"].append(text)
                c["count"] += 1
                return clusters
        clusters.append({"key": key, "rep": text, "members": [text], "count": 1, "_toks": set()})
        return clusters
    toks = scoring.tokens(text)  # free-form: bucket by lexical similarity
    for c in clusters:
        if not c.get("key") and scoring.jaccard(toks, c.get("_toks", set())) >= sim_threshold:
            c["members"].append(text)
            c["count"] += 1
            return clusters
    clusters.append({"key": "", "rep": text, "members": [text], "count": 1, "_toks": toks})
    return clusters


def counts(clusters: list[dict]) -> list[int]:
    return [c["count"] for c in clusters]


def leader(clusters: list[dict]) -> Optional[dict]:
    """The bucket with the most votes (the consensus answer), or None."""
    return max(clusters, key=lambda c: c["count"]) if clusters else None


def confident(clusters: list[dict], *, threshold: float = _THRESHOLD, margin: int = _MARGIN) -> bool:
    """True when the top bucket holds a >= ``threshold`` share AND leads by >= ``margin``.

    A cheap, dependency-free stand-in for Adaptive-Consistency's Dirichlet rule
    that captures most of the sample savings.
    """
    cs = sorted(counts(clusters), reverse=True)
    if not cs:
        return False
    total = sum(cs)
    top = cs[0]
    second = cs[1] if len(cs) > 1 else 0
    return total > 0 and (top / total) >= threshold and (top - second) >= margin


def cluster(texts: list[str], *, sim_threshold: float = _SIM) -> list[dict]:
    """Cluster a full list of answers at once (for non-incremental callers)."""
    clusters: list[dict] = []
    for t in texts:
        assign(clusters, t, sim_threshold=sim_threshold)
    return clusters
