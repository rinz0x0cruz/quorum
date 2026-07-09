"""Judge cadence (run.judge_every) -- defer judging to cut judge calls (offline)."""
from quorum import judge, orchestrator
from quorum.config import _deep_merge
from quorum.store import Store
from tests.helpers import mock_cfg


def test_judge_due_helper():
    assert judge.due(1, 1, 4) is True and judge.due(3, 1, 4) is True   # every=1 -> always
    assert judge.due(1, 2, 4) is True                                  # first round
    assert judge.due(2, 2, 4) is True                                  # multiple of every
    assert judge.due(3, 2, 4) is False                                 # skipped
    assert judge.due(4, 2, 4) is True                                  # last round


def test_refine_defers_judge(tmp_path):
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")),
                      {"run": {"judge_every": 2, "max_rounds": 4, "target_score": 200}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "q", store=store, strategy="refine",
                                        promptsmith_on=False)
    judges = [t for r in sess.rounds for t in r.turns if t.kind == "judge"]
    gens = [t for r in sess.rounds for t in r.turns if t.kind in ("propose", "revise")]
    assert len(gens) == 4                       # generated every round
    assert len(judges) == 3                     # judged 1, 2, 4 (skipped 3)
    assert sess.final and sess.final_score > 0


def test_judge_every_default_unchanged(tmp_path):
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")), {"run": {"max_rounds": 4}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "q", store=store, strategy="refine",
                                        promptsmith_on=False)
    judges = [t for r in sess.rounds for t in r.turns if t.kind == "judge"]
    assert len(judges) == 2 and "target" in sess.stop_reason  # mock hits 85 at round 2
