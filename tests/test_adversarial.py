"""Adversarial / degenerate inputs -- probe robustness of the new code paths."""
from quorum import consistency, judge, provider, throttle
from quorum.config import validate_config
from tests.helpers import mock_cfg


def test_rate_limiter_edge_rpm():
    assert provider.RateLimiter(-5).acquire() == 0.0      # negative -> disabled
    assert provider.RateLimiter(0).acquire() == 0.0
    assert provider.RateLimiter(0.5).acquire() == 0.0     # first slot is always free


def test_consistency_degenerate_inputs():
    assert consistency.cluster([]) == []
    assert consistency.cluster(["", "   ", "\n\t"]) == []   # blank/whitespace ignored
    assert consistency.leader([]) is None
    assert consistency.confident([]) is False
    assert consistency.confident(consistency.cluster(["only one"])) is False  # no margin


def test_summarize_tolerates_sparse_rows():
    rows = [{"model": "m", "provider": "openrouter", "status": "ok"},   # no latency/code/rl
            {"status": "HTTP 429", "http_code": 429}]                    # no model/provider/ts
    s = throttle.summarize(rows)
    assert s["total"] == 2 and s["throttled"] == 1
    assert "m" in s["by_model"] and "?" in s["by_model"]                 # missing model -> "?"


def test_validate_handles_lists_and_nested_typos():
    assert validate_config({"run": {"cascade": ["refine", "debate"], "max_rounds": 3}}) == []
    assert "run.bogus" in validate_config({"run": {"bogus": {"x": 1}}})  # dict typo flagged by leaf


def test_judge_single_candidate_shuffle_is_noop(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))                 # shuffle_candidates default on
    prov = provider.for_config(cfg)
    v, _ = judge.evaluate(cfg, prov, 1, "t", "p", [("solo", "the answer")],
                          candidate_models=["m"])
    assert v.best_label == "solo" and v.best_content == "the answer"


def test_judge_empty_best_letter_defaults_to_first(tmp_path, monkeypatch):
    cfg = mock_cfg(str(tmp_path / "t.db"), judge={"shuffle_candidates": False})
    prov = provider.for_config(cfg)
    # judge returns no usable "best" -> falls back to the first shown candidate
    monkeypatch.setattr(provider.MockResponder, "respond",
                        lambda self, spec, msgs: '{"score": 50}')
    v, _ = judge.evaluate(cfg, prov, 1, "t", "p", [("a", "one"), ("b", "two")],
                          candidate_models=["m1", "m2"])
    assert v.best_label == "a" and v.score == 50.0
