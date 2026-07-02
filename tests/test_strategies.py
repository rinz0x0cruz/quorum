import pytest

from quorum import orchestrator
from quorum.store import Store
from tests.helpers import mock_cfg


@pytest.mark.parametrize("strat", ["debate", "council", "moa", "refine", "ensemble"])
def test_strategy_runs_end_to_end(tmp_path, strat):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, f"solve {strat}", store=store, strategy=strat)
        assert store.get_session(sess.id) is not None
    assert sess.final and sess.final_score > 0
    assert len(sess.rounds) >= 1
    assert sess.tokens_in > 0 and sess.tokens_out > 0
    assert sess.status == "ok"


def test_debate_stops_at_target(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "x", store=store, strategy="debate")
    assert "target" in sess.stop_reason  # mock ramp crosses 85 by round 2


def test_promptsmith_adds_round_zero(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "x", store=store, strategy="debate")
    assert any(r.index == 0 for r in sess.rounds)


def test_no_promptsmith_skips_round_zero(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "x", store=store, strategy="refine",
                                        promptsmith_on=False)
    assert all(r.index != 0 for r in sess.rounds)


def test_moa_layers_recorded(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"), run={"moa_layers": 3, "parallel": False})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "x", store=store, strategy="moa",
                                        promptsmith_on=False)
    # 3 layer rounds + 1 aggregation round
    assert len([r for r in sess.rounds if r.index != 0]) == 4
