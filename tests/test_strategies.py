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


def test_top_k_council_still_runs(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"), run={"top_k": 2, "parallel": False, "max_rounds": 1})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "rank then fuse", store=store,
                                        strategy="council", promptsmith_on=False)
    assert sess.final and sess.final_score > 0


def test_top_k_moa_adds_review_turn(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"), run={"top_k": 2, "parallel": False, "moa_layers": 1})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "rank then fuse", store=store,
                                        strategy="moa", promptsmith_on=False)
    kinds = [t.kind for r in sess.rounds for t in r.turns]
    assert "review" in kinds and sess.final


def test_devils_advocate_produces_challenge_turn(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"),
                   run={"devils_advocate": True, "parallel": False,
                        "max_rounds": 2, "target_score": 200})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "argue", store=store,
                                        strategy="debate", promptsmith_on=False)
    kinds = [t.kind for r in sess.rounds for t in r.turns]
    assert "challenge" in kinds


def test_promptsmith_bootstrap_runs_with_seeded_sessions(tmp_path):
    from quorum import promptsmith, provider
    from quorum.model import Session
    cfg = mock_cfg(str(tmp_path / "t.db"), promptsmith={"bootstrap": True, "rounds": 1})
    with Store(cfg["output"]["db_path"]) as store:
        store.save_session(Session(id="s1", task="prior", strategy="debate",
                                   prompt="Solve step by step.", final="a", final_score=95.0))
        prov = provider.for_config(cfg)
        instr = promptsmith.refine(cfg, prov, "new task", store=store)
    assert isinstance(instr, str) and instr
