"""Difficulty-adaptive cascade strategy (offline via mock)."""
from quorum import orchestrator
from quorum.config import _deep_merge
from quorum.store import Store
from tests.helpers import mock_cfg


def test_cascade_stops_at_cheap_stage(tmp_path):
    # Mock judge ramps to 85 by round 2, so the first stage (refine) hits target
    # and the cascade never escalates to debate/council.
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")),
                      {"run": {"cascade": ["refine", "debate", "council"]}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "easy task", store=store, strategy="cascade",
                                        promptsmith_on=False)
    assert sess.strategy == "cascade"
    assert sess.final and sess.final_score >= 85
    assert sess.stop_reason.startswith("cascade: refine reached target")
    assert sess.status == "ok"


def test_cascade_escalates_and_keeps_best(tmp_path):
    # Impossible target + single round -> no stage hits target -> exhaust + best.
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")),
                      {"run": {"cascade": ["refine", "debate"], "target_score": 200,
                               "max_rounds": 1}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "hard task", store=store, strategy="cascade",
                                        promptsmith_on=False)
    assert "exhausted" in sess.stop_reason
    assert sess.final and sess.final_score > 0
    assert sess.status == "ok"


def test_cascade_defaults_when_unset(tmp_path):
    # strategy=cascade with no run.cascade -> default [refine, debate, council].
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "task", store=store, strategy="cascade",
                                        promptsmith_on=False)
    assert sess.strategy == "cascade" and sess.final
    assert sess.stop_reason.startswith("cascade: refine reached target")
