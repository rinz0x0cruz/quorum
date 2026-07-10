"""Step-Back strategy (abstract to a principle, then solve) -- offline via mock."""
from quorum import orchestrator, prompts
from quorum.store import Store
from tests.helpers import mock_cfg


def test_stepback_abstracts_then_solves(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "why is the sky blue?", store=store,
                                        strategy="stepback", promptsmith_on=False)
    assert sess.strategy == "stepback"
    assert sess.final and sess.final_score > 0
    kinds = [t.kind for r in sess.rounds for t in r.turns]
    assert "abstract" in kinds and "solve" in kinds and "judge" in kinds
    assert kinds.index("abstract") < kinds.index("solve")  # step back before solving
    assert "step-back" in sess.stop_reason


def test_stepback_no_members_errors(tmp_path):
    from quorum.config import _deep_merge
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")), {"council": {"members": []}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "q", store=store, strategy="stepback",
                                        promptsmith_on=False)
    assert sess.status == "error" and "no members" in sess.stop_reason


def test_stepback_prompt_builders():
    ab = prompts.step_back("", "what is 2+2")
    assert ab[0]["role"] == "system" and "QUORUM-STEPBACK-ABSTRACT" in ab[0]["content"]
    assert "what is 2+2" in ab[1]["content"]
    sv = prompts.step_back_solve("", "what is 2+2", "addition combines quantities")
    assert "QUORUM-STEPBACK-SOLVE" in sv[0]["content"]
    assert "addition combines quantities" in sv[1]["content"]


def test_step_back_solve_handles_empty_principle():
    sv = prompts.step_back_solve("", "task", "   ")
    assert "first principles" in sv[1]["content"].lower()


def test_stepback_registered():
    from quorum.strategies import get
    assert get("stepback") is not None
