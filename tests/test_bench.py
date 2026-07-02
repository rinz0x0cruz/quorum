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
