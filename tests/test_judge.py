from quorum import judge, provider
from quorum.model import Verdict
from tests.helpers import mock_cfg
import json


def _v(score):
    return Verdict(round=1, score=score)


def test_evaluate_scores_and_picks_best(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    prov = provider.for_config(cfg)
    cands = [("alice", "one"), ("bob", "two")]
    v1, turn = judge.evaluate(cfg, prov, 1, "task", "prompt", cands,
                              candidate_models=["meta/mock-alice", "anthropic/mock-bob"])
    assert v1.score == 70.0 and (v1.best_label, v1.best_content) in cands
    assert turn.kind == "judge" and turn.tokens_out > 0


def test_should_stop_target():
    cfg = {"run": {"target_score": 85, "max_rounds": 9}}
    stop, reason = judge.should_stop(cfg, [_v(70), _v(88)], 2)
    assert stop and "target" in reason


def test_should_stop_plateau():
    cfg = {"run": {"target_score": 200, "max_rounds": 9, "plateau_delta": 2, "plateau_patience": 2}}
    stop, reason = judge.should_stop(cfg, [_v(50), _v(51), _v(51.5)], 3)
    assert stop and "plateau" in reason


def test_should_stop_cap():
    cfg = {"run": {"target_score": 200, "max_rounds": 3, "plateau_patience": 9, "plateau_delta": 2}}
    stop, reason = judge.should_stop(cfg, [_v(50), _v(55), _v(60)], 3)
    assert stop and "max" in reason


def test_consensus_detection():
    assert judge.consensus_reached(["same answer text", "same answer text"]) is True
    assert judge.consensus_reached(["totally different", "nothing alike here"]) is False


def test_cross_family_guard_prefers_other_vendor(tmp_path):
    # Judge is openai; candidate is also openai -> guard should swap to a non-openai member.
    cfg = mock_cfg(str(tmp_path / "t.db"), council={
        "members": [{"name": "a", "provider": "mock", "model": "anthropic/claude"}],
        "judge": "mock:openai/gpt-4o"}, judge={"cross_family_guard": True})
    chosen = judge._pick_judge(cfg, ["openai/gpt-4o"])
    assert chosen.model == "anthropic/claude"


def _rf_spy(prov, seen):
    orig = prov.complete

    def spy(spec, messages, **kw):
        seen["rf"] = kw.get("response_format")
        return orig(spec, messages, **kw)

    prov.complete = spy  # type: ignore[method-assign]


def test_json_mode_passes_response_format(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"), judge={"json_mode": True})
    prov = provider.for_config(cfg)
    seen = {}
    _rf_spy(prov, seen)
    judge.evaluate(cfg, prov, 1, "task", "prompt", [("a", "ans")], candidate_models=["m"])
    assert seen["rf"] == {"type": "json_object"}


def test_json_mode_off_by_default(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    prov = provider.for_config(cfg)
    seen = {}
    _rf_spy(prov, seen)
    judge.evaluate(cfg, prov, 1, "task", "prompt", [("a", "ans")], candidate_models=["m"])
    assert seen["rf"] is None


def test_judge_shuffle_maps_back_consistently(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))          # shuffle_candidates defaults True
    prov = provider.for_config(cfg)
    cands = [("alice", "one"), ("bob", "two"), ("carol", "three")]
    v, _ = judge.evaluate(cfg, prov, 1, "t", "p", cands, candidate_models=["m1", "m2", "m3"])
    assert (v.best_label, v.best_content) in cands   # mapped back to a real candidate


def test_judge_shuffle_off_picks_first(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"), judge={"shuffle_candidates": False})
    prov = provider.for_config(cfg)
    cands = [("alice", "one"), ("bob", "two")]
    v, _ = judge.evaluate(cfg, prov, 1, "t", "p", cands, candidate_models=["m1", "m2"])
    assert v.best_label == "alice" and v.best_content == "one"


def test_judge_extracts_json_from_prose(tmp_path, monkeypatch):
    cfg = mock_cfg(str(tmp_path / "t.db"), judge={"shuffle_candidates": False})
    prov = provider.for_config(cfg)
    monkeypatch.setattr(provider.MockResponder, "respond",
                        lambda self, spec, msgs: 'verdict: {"score": 77, "best": "A"} ok')
    v, _ = judge.evaluate(cfg, prov, 1, "t", "p", [("a", "one")], candidate_models=["m"])
    assert v.score == 77.0                       # regex-extracted from surrounding prose


def test_judge_overall_from_subscores(tmp_path, monkeypatch):
    cfg = mock_cfg(str(tmp_path / "t.db"), judge={"shuffle_candidates": False})
    prov = provider.for_config(cfg)
    monkeypatch.setattr(provider.MockResponder, "respond",
                        lambda self, spec, msgs: json.dumps(
                            {"sub_scores": {"correctness": 80, "clarity": 60}, "best": "A"}))
    v, _ = judge.evaluate(cfg, prov, 1, "t", "p", [("a", "one")], candidate_models=["m"])
    assert v.score == 73.33                      # weighted by the default rubric (.40/.20)
