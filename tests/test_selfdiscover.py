"""Self-Discover strategy (compose a reasoning structure, then solve) -- offline via mock."""
from quorum import orchestrator, prompts
from quorum.store import Store
from tests.helpers import mock_cfg


def test_selfdiscover_plans_then_solves(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "solve it", store=store, strategy="selfdiscover",
                                        promptsmith_on=False)
    assert sess.strategy == "selfdiscover"
    assert sess.final and sess.final_score > 0
    kinds = [t.kind for r in sess.rounds for t in r.turns]
    assert "plan" in kinds and "solve" in kinds and "judge" in kinds
    # discover happens before solve
    assert kinds.index("plan") < kinds.index("solve")
    assert "self-discover" in sess.stop_reason


def test_selfdiscover_no_members_errors(tmp_path):
    from quorum.config import _deep_merge
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")), {"council": {"members": []}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "q", store=store, strategy="selfdiscover",
                                        promptsmith_on=False)
    assert sess.status == "error" and "no members" in sess.stop_reason


def test_discover_prompt_builders():
    plan = prompts.discover("", "what is 2+2")
    assert plan[0]["role"] == "system" and "QUORUM-SELFDISCOVER-PLAN" in plan[0]["content"]
    assert "what is 2+2" in plan[1]["content"]
    solve = prompts.discover_solve("", "what is 2+2", "1. add the numbers")
    assert "QUORUM-SELFDISCOVER-SOLVE" in solve[0]["content"]
    assert "1. add the numbers" in solve[1]["content"]


def test_discover_solve_handles_empty_structure():
    solve = prompts.discover_solve("", "task", "   ")
    # falls back to a generic step-by-step instruction rather than an empty plan
    assert "step by step" in solve[1]["content"].lower()


def test_selfdiscover_registered():
    from quorum.strategies import get
    assert get("selfdiscover") is not None
