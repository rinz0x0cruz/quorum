"""Unit tests for the ``quorum.scoring`` package.

These lock the shared lexical primitives and the scorer registry. They also serve
as the behavior-preservation proof for the refactor: ``overlap_coeff`` is exactly
what ``contextwindow.select`` computes and ``jaccard`` is exactly what
``judge.consensus_reached`` averages -- two *different* formulas that must not be
collapsed. All offline, deterministic, stdlib-only.
"""
import math

import pytest

from quorum import scoring
from quorum.scoring import text as scoring_text


# --- tokenizer -----------------------------------------------------------
def test_tokens_lowercases_and_splits_on_non_alnum():
    assert scoring.tokens("Hello, WORLD 42!") == {"hello", "world", "42"}


def test_tokens_empty_and_none_safe():
    assert scoring.tokens("") == set()
    assert scoring.tokens(None) == set()          # type: ignore[arg-type]
    assert scoring.tokens("!!! --- ???") == set()  # no [a-z0-9] runs


# --- overlap coefficient (rank / contextwindow) --------------------------
def test_overlap_coeff_is_fraction_of_query_covered():
    assert scoring.overlap_coeff({"a", "b"}, {"a", "b", "c", "d"}) == 1.0
    assert scoring.overlap_coeff({"a", "b", "c", "d"}, {"a", "b"}) == 0.5


def test_overlap_coeff_empty_query_is_zero():
    assert scoring.overlap_coeff(set(), {"a"}) == 0.0
    assert scoring.overlap_coeff(set(), set()) == 0.0


# --- jaccard (judge.consensus_reached) -----------------------------------
def test_jaccard_is_intersection_over_union():
    assert scoring.jaccard({"a", "b"}, {"a", "b", "c", "d"}) == 0.5
    assert scoring.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert scoring.jaccard({"a"}, {"a"}) == 1.0


def test_jaccard_empty_union_is_zero():
    assert scoring.jaccard(set(), set()) == 0.0


def test_overlap_and_jaccard_are_different_formulas():
    # Same inputs, different results -> the two measures are not interchangeable.
    q, d = {"a", "b"}, {"a", "b", "c", "d"}
    assert scoring.overlap_coeff(q, d) == 1.0
    assert scoring.jaccard(q, d) == 0.5
    assert scoring.overlap_coeff(q, d) != scoring.jaccard(q, d)


# --- LexicalScorer + registry --------------------------------------------
def test_lexical_scorer_scores_via_overlap_coeff():
    scorer = scoring.LexicalScorer()
    # score(query, text) == overlap_coeff(tokens(query), tokens(text))
    assert scorer.score("a b", "a b c d") == 1.0
    assert scorer.score("a b c d", "a b") == 0.5
    assert scorer.score("", "anything") == 0.0


def test_lexical_scorer_satisfies_scorer_protocol():
    assert isinstance(scoring.LexicalScorer(), scoring.Scorer)


def test_registry_has_builtin_lexical():
    assert "lexical" in scoring.available()
    got = scoring.get("lexical")
    assert isinstance(got, scoring.Scorer)
    assert got.score("shared terms", "these shared terms") == 1.0


def test_register_and_get_roundtrip():
    class Constant:
        def score(self, query: str, text: str) -> float:
            return 0.5

    scoring.register("t_const", Constant())
    try:
        assert "t_const" in scoring.available()
        assert scoring.get("t_const").score("x", "y") == 0.5
    finally:
        scoring._REGISTRY.pop("t_const", None)   # keep the registry clean for other tests


def test_get_unknown_scorer_raises_keyerror():
    with pytest.raises(KeyError):
        scoring.get("no_such_scorer")


# --- the primitives live in scoring.text and are re-exported -------------
def test_reexports_match_text_module():
    assert scoring.tokens is scoring_text.tokens
    assert scoring.overlap_coeff is scoring_text.overlap_coeff
    assert scoring.jaccard is scoring_text.jaccard


def test_scores_are_bounded_0_to_1():
    for a, b in [({"a"}, {"b"}), ({"a", "b"}, {"a"}), ({"a"}, {"a", "b"})]:
        assert 0.0 <= scoring.overlap_coeff(a, b) <= 1.0
        assert 0.0 <= scoring.jaccard(a, b) <= 1.0
        assert not math.isnan(scoring.jaccard(a, b))
