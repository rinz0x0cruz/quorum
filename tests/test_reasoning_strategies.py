"""Reflexion + Chain-of-Verification strategies (offline via mock)."""
from quorum import orchestrator, prompts
from quorum.config import _deep_merge
from quorum.store import Store
from tests.helpers import mock_cfg


# --- prompt builders + sentinels -----------------------------------------
def test_reflexion_prompts_have_sentinel():
    assert "QUORUM-REFLECT" in prompts.reflect("p", "t", "ans", "crit")[0]["content"]
    actor = prompts.reflexion_actor("p", "t", ["lesson one", "lesson two"])
    assert "lesson one" in actor[1]["content"] and "lesson two" in actor[1]["content"]


def test_verify_prompts_have_sentinels():
    assert "QUORUM-VERIFY-PLAN" in prompts.plan_checks("p", "t", "draft")[0]["content"]
    assert "QUORUM-VERIFY-ANSWER" in prompts.verify_checks("p", "t", "q?")[0]["content"]
    assert "QUORUM-VERIFY-REVISE" in prompts.verified_final("p", "t", "d", "qa")[0]["content"]


def test_verify_withholds_draft_when_answering():
    draft = "the capital is Sydney"                     # a checkable (wrong) claim
    plan = prompts.plan_checks("", "capital of Australia?", draft)
    ans = prompts.verify_checks("", "capital of Australia?", "1. What is the capital?")
    assert draft in plan[1]["content"]                  # planning sees the draft ...
    assert draft not in ans[1]["content"]               # ... answering does not (independence)


# --- reflexion strategy ---------------------------------------------------
def test_reflexion_runs_and_reflects(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "solve it", store=store, strategy="reflexion",
                                        promptsmith_on=False)
    assert sess.strategy == "reflexion"
    assert sess.final and sess.final_score > 0
    reflects = [t for r in sess.rounds for t in r.turns if t.kind == "reflect"]
    assert len(reflects) >= 1                            # a reflection was stored in memory
    assert "target" in sess.stop_reason                  # mock ramp crosses target by round 2


def test_reflexion_no_members_errors(tmp_path):
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")), {"council": {"members": []}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "q", store=store, strategy="reflexion",
                                        promptsmith_on=False)
    assert sess.status == "error" and "no members" in sess.stop_reason


# --- chain-of-verification strategy --------------------------------------
def test_verify_runs_full_pipeline(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "solve it", store=store, strategy="verify",
                                        promptsmith_on=False)
    assert sess.strategy == "verify"
    assert sess.final and sess.final_score > 0
    kinds = {t.kind for r in sess.rounds for t in r.turns}
    assert {"draft", "plan", "verify", "revise", "judge"} <= kinds
    assert "chain-of-verification" in sess.stop_reason


def test_verify_no_members_errors(tmp_path):
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")), {"council": {"members": []}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "q", store=store, strategy="verify",
                                        promptsmith_on=False)
    assert sess.status == "error" and "no members" in sess.stop_reason
