from quorum import bench
from quorum.store import Store
from tests.helpers import mock_cfg


def _rows():
    return [
        {"strategy": "debate", "task_id": "a", "score": 85, "rounds": 2, "tokens": 100, "cost_usd": 0, "seconds": 0},
        {"strategy": "moa", "task_id": "a", "score": 96, "rounds": 3, "tokens": 90, "cost_usd": 0, "seconds": 0},
        {"strategy": "debate", "task_id": "b", "score": 85, "rounds": 2, "tokens": 100, "cost_usd": 0, "seconds": 0},
        {"strategy": "moa", "task_id": "b", "score": 96, "rounds": 3, "tokens": 90, "cost_usd": 0, "seconds": 0},
    ]


def test_aggregate_ranks_by_score_and_winrate():
    summ = bench.aggregate(_rows(), ["debate", "moa"], 2)
    assert summ[0]["strategy"] == "moa"
    assert summ[0]["win_rate"] == 100.0
    assert summ[1]["strategy"] == "debate" and summ[1]["win_rate"] == 0.0


def test_bench_run_offline(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text("tasks:\n  - id: one\n    task: first task\n  - id: two\n    task: second task\n",
                     encoding="utf-8")
    with Store(cfg["output"]["db_path"]) as store:
        rc = bench.run(cfg, str(tasks), ["debate", "moa", "ensemble"], store, verbose=False)
        assert rc == 0
        assert len(store.bench_rows()) == 6  # 2 tasks x 3 strategies


def test_bench_shares_one_provider_across_tasks(tmp_path, monkeypatch):
    from quorum import orchestrator
    cfg = mock_cfg(str(tmp_path / "t.db"))
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text("tasks:\n  - id: one\n    task: t1\n  - id: two\n    task: t2\n",
                     encoding="utf-8")
    seen = []
    orig = orchestrator.run_session

    def _spy(cfg2, task, **kw):
        seen.append(kw.get("prov"))
        return orig(cfg2, task, **kw)

    monkeypatch.setattr(orchestrator, "run_session", _spy)
    with Store(cfg["output"]["db_path"]) as store:
        bench.run(cfg, str(tasks), ["debate", "refine"], store, verbose=False)
    assert len(seen) == 4                       # 2 tasks x 2 strategies
    assert seen[0] is not None
    assert all(p is seen[0] for p in seen)      # one shared provider -> one rate limiter


def test_bench_persists_and_dashboards_accuracy(tmp_path):
    from quorum import render
    cfg = mock_cfg(str(tmp_path / "t.db"))
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text("tasks:\n  - id: m1\n    task: what is 2+2\n    answer: '#### 4'\n", encoding="utf-8")
    with Store(cfg["output"]["db_path"]) as store:
        bench.run(cfg, str(tasks), ["refine", "ensemble"], store, verbose=False)
        rows = store.bench_rows()
        # match/correct are now persisted on every bench row (graded task -> not None)
        assert all("match" in r and "correct" in r for r in rows)
        assert any(r["match"] is not None for r in rows)
        path = render.build(cfg, store)
    with open(path, encoding="utf-8") as fh:
        page = fh.read()
    # the graded summary (accuracy / mean_match) reaches the dashboard payload
    assert '"accuracy"' in page and '"mean_match"' in page


def test_bench_dashboard_no_accuracy_when_ungraded(tmp_path):
    from quorum import render
    cfg = mock_cfg(str(tmp_path / "t.db"))
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text("tasks:\n  - id: q1\n    task: open ended question\n", encoding="utf-8")
    with Store(cfg["output"]["db_path"]) as store:
        bench.run(cfg, str(tasks), ["refine"], store, verbose=False)
        path = render.build(cfg, store)
    with open(path, encoding="utf-8") as fh:
        page = fh.read()
    assert '"mean_match"' not in page            # no reference -> no accuracy columns
