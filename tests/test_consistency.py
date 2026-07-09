"""Answer clustering + adaptive-consistency stop (offline)."""
from quorum import consistency, orchestrator
from quorum.config import _deep_merge
from quorum.store import Store
from tests.helpers import mock_cfg


def test_numeric_bucketing_votes():
    cl = consistency.cluster(["the answer is 42", "so 42", "clearly 17"])
    top = consistency.leader(cl)
    assert top["count"] == 2                 # two say 42, one says 17
    assert consistency.counts(cl) == [2, 1] or sorted(consistency.counts(cl), reverse=True) == [2, 1]


def test_freeform_similarity_bucketing():
    a = "install the package then run the tests to verify"
    b = "install the package then run the tests to verify it"   # near-identical
    c = "completely unrelated gardening advice about roses"
    cl = consistency.cluster([a, b, c])
    counts = sorted(consistency.counts(cl), reverse=True)
    assert counts == [2, 1]


def test_confident_rule():
    assert consistency.confident(consistency.cluster(["5", "5", "5"])) is True
    # a single sample can't be confident (no margin)
    assert consistency.confident(consistency.cluster(["5"])) is False
    # a tie is not confident
    assert consistency.confident(consistency.cluster(["5", "7"])) is False


def test_ensemble_adaptive_stops_early(tmp_path):
    # mock returns identical text per (model, prompt), so all samples agree and the
    # adaptive loop stops at samples_min instead of drawing all `samples`.
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")),
                      {"run": {"adaptive_samples": True, "samples": 10, "samples_min": 2}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "q", store=store, strategy="ensemble",
                                        promptsmith_on=False)
    proposes = [t for r in sess.rounds for t in r.turns if t.kind == "propose"]
    assert len(proposes) == 2                # early stop at samples_min, not 10
    assert sess.final and sess.final_score > 0
    assert "adaptive vote" in sess.stop_reason


def test_ensemble_fixed_n_unchanged(tmp_path):
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")), {"run": {"samples": 3}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "q", store=store, strategy="ensemble",
                                        promptsmith_on=False)
    proposes = [t for r in sess.rounds for t in r.turns if t.kind == "propose"]
    assert len(proposes) == 3                # fixed-N path untouched
    assert "best of" in sess.stop_reason
