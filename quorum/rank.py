"""Rank candidates from peer reviews so the fuser sees only the best few.

LLM-Blender's insight is to *separate ranking from fusion*: rank the candidates
first, then fuse only the top-K rather than every response. quorum derives the
ranking from the peer-review turns the council already produces -- each reviewer
is asked to list the candidates best-first -- so no extra model call is needed
(the council reuses its reviews; the MoA strategy adds one lightweight review
only when top-K is enabled).

We read the order in which candidate labels (``CANDIDATE A`` / bare capitals used
as rankings, e.g. ``A, C, B``) first appear in each review, score them by
position (a Borda count), and aggregate across reviewers. Fully deterministic and
offline-testable; this module never touches the network.
"""
from __future__ import annotations

import re

_CAND_RE = re.compile(r"candidate\s+([a-z])", re.IGNORECASE)
_LONE_CAP_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")


def _order_in(review: str, n: int) -> list[int]:
    """First-appearance order of candidate indices (0..n-1) in one review.

    Prefers explicit ``candidate X`` mentions; otherwise falls back to lone
    capital letters used as ranking tokens (``A, B, C``). Unmentioned candidates
    are appended in their original order so every candidate is scored.
    """
    text = review or ""
    letters = _CAND_RE.findall(text) or _LONE_CAP_RE.findall(text)
    seen: list[int] = []
    for ch in letters:
        idx = ord(ch.upper()) - 65
        if 0 <= idx < n and idx not in seen:
            seen.append(idx)
    for i in range(n):
        if i not in seen:
            seen.append(i)
    return seen


def consensus_order(n: int, reviews: list[str]) -> list[int]:
    """Aggregate reviewer orderings into one ranking (Borda count), best-first.

    With no usable reviews, returns the original order ``0..n-1`` so callers
    degrade gracefully.
    """
    if n <= 0:
        return []
    points = [0.0] * n
    counted = 0
    for review in reviews or []:
        if not review:
            continue
        counted += 1
        for position, idx in enumerate(_order_in(review, n)):
            points[idx] += (n - position)   # best-ranked gets the most points
    if not counted:
        return list(range(n))
    return sorted(range(n), key=lambda i: (-points[i], i))


def top_k_indices(n: int, reviews: list[str], k: int) -> list[int]:
    """Indices of the top-``k`` candidates, best-first. ``k <= 0`` returns all."""
    order = consensus_order(n, reviews)
    return order[:k] if k and k > 0 else order
