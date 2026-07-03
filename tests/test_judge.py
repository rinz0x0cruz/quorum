from quorum import judge, provider
from quorum.model import Verdict
from tests.helpers import mock_cfg


def _v(score):
    return Verdict(round=1, score=score)


def test_evaluate_scores_and_picks_best(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    prov = provider.for_config(cfg)
    cands = [("alice", "one"), ("bob", "two")]
    v1, turn = judge.evaluate(cfg, prov, 1, "task", "prompt", cands,
                              candidate_models=["meta/mock-alice", "anthropic/mock-bob"])
    assert v1.score == 70.0 and v1.best_label == "alice"
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
