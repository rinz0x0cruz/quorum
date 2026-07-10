"""Self-MoA strategy (single best model sampled + aggregated) -- offline via mock."""
from quorum import orchestrator
from quorum.config import _deep_merge
from quorum.store import Store
from tests.helpers import mock_cfg


def test_selfmoa_samples_one_model_and_aggregates(tmp_path):
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")), {"run": {"samples": 3}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "solve it", store=store, strategy="selfmoa",
                                        promptsmith_on=False)
    assert sess.strategy == "selfmoa"
    assert sess.final and sess.final_score > 0
    kinds = [t.kind for r in sess.rounds for t in r.turns]
    assert kinds.count("propose") == 3        # sampled the single model 3x (cache bypassed)
    assert "aggregate" in kinds and "judge" in kinds
    assert "self-moa" in sess.stop_reason


def test_selfmoa_no_members_errors(tmp_path):
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")), {"council": {"members": []}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "q", store=store, strategy="selfmoa",
                                        promptsmith_on=False)
    assert sess.status == "error" and "no members" in sess.stop_reason
