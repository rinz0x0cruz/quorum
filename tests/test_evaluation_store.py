"""Evaluation/profile/tuning persistence is additive and fully offline."""
import sqlite3
import threading

import pytest

from quorum.model import (EvaluationRun, EvaluationSample, ProfilePromotion,
                          TuneRun)
from quorum.store import Store


def test_evaluation_run_and_sample_round_trip(tmp_path):
    with Store(str(tmp_path / "t.db")) as store:
        run = EvaluationRun(
            id="eval-1", target_type="model", target_id="mock:model-a",
            pack_id="reasoning", pack_version="1.0.0", split="validation",
            manifest={"pack_fingerprint": "abc", "seed": 42},
        )
        store.save_eval_run(run)
        sample = EvaluationSample(
            id="sample-1", run_id=run.id, task_id="math-1",
            requested_ref="mock:model-a", actual_ref="mock:model-a",
            score=100.0, match=100.0, correct=True, latency_ms=3,
            tokens_in=10, tokens_out=2, output="42",
            metadata={"catalog_snapshot": "catalog-1"},
        )
        store.save_eval_sample(sample)

        saved_run = store.get_eval_run(run.id)
        saved_sample = store.eval_samples(run.id)[0]
        assert saved_run and saved_run["manifest"]["seed"] == 42
        assert saved_sample["correct"] is True
        assert saved_sample["metadata"]["catalog_snapshot"] == "catalog-1"

        run.status = "ok"
        run.completed = "2026-07-15T12:00:00Z"
        store.save_eval_run(run)
        assert store.get_eval_run(run.id)["status"] == "ok"


def test_evaluation_identity_is_immutable(tmp_path):
    with Store(str(tmp_path / "t.db")) as store:
        run = EvaluationRun(
            id="eval-1", target_type="model", target_id="mock:model-a",
            pack_id="reasoning", manifest={"seed": 42},
        )
        store.save_eval_run(run)
        run.target_id = "mock:model-b"
        with pytest.raises(ValueError, match="manifest is immutable"):
            store.save_eval_run(run)

        sample = EvaluationSample(
            id="sample-1", run_id="eval-1", task_id="task-a",
            requested_ref="mock:model-a",
        )
        store.save_eval_sample(sample)
        sample.task_id = "task-b"
        with pytest.raises(ValueError, match="identity is immutable"):
            store.save_eval_sample(sample)


def test_parallel_evaluation_sample_writes(tmp_path):
    with Store(str(tmp_path / "t.db")) as store:
        store.save_eval_run(EvaluationRun(
            id="eval-1", target_type="model", target_id="mock:model-a", pack_id="reasoning"))

        def writer(worker: int) -> None:
            for item in range(10):
                store.save_eval_sample(EvaluationSample(
                    id=f"sample-{worker}-{item}", run_id="eval-1",
                    task_id=f"task-{worker}-{item}", requested_ref="mock:model-a"))

        threads = [threading.Thread(target=writer, args=(worker,)) for worker in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(store.eval_samples("eval-1")) == 40


def test_profile_promotion_is_append_only(tmp_path):
    promotion = ProfilePromotion(
        id="promotion-1", profile_name="security", profile_version="1.0.0",
        eval_run_id="eval-1", approved_by="reviewer",
        manifest={"profile_hash": "sha256:abc"},
    )
    with Store(str(tmp_path / "t.db")) as store:
        store.add_profile_promotion(promotion)
        rows = store.profile_promotions("security")
        assert rows[0]["eval_run_id"] == "eval-1"
        assert rows[0]["manifest"]["profile_hash"] == "sha256:abc"
        with pytest.raises(sqlite3.IntegrityError):
            store.add_profile_promotion(promotion)


def test_tune_run_lifecycle_and_manifest_immutability(tmp_path):
    run = TuneRun(
        id="tune-1", method="lora", backend="mock", base_model="open/model",
        manifest={"train_fingerprint": "train-1", "seed": 42},
    )
    with Store(str(tmp_path / "t.db")) as store:
        store.save_tune_run(run)
        run.status = "ok"
        run.completed = "2026-07-15T12:00:00Z"
        run.artifact = {"path": "data/tuning/adapter", "hash": "sha256:def"}
        store.save_tune_run(run)
        saved = store.get_tune_run(run.id)
        assert saved and saved["status"] == "ok"
        assert saved["artifact"]["hash"] == "sha256:def"

        run.manifest["seed"] = 7
        with pytest.raises(ValueError, match="manifest is immutable"):
            store.save_tune_run(run)


def test_existing_database_gains_new_tables_without_data_loss(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, created TEXT, task TEXT, strategy TEXT, "
        "prompt TEXT, final TEXT, final_score REAL, status TEXT, rounds INTEGER, "
        "tokens_in INTEGER, tokens_out INTEGER, json TEXT)"
    )
    conn.execute(
        "INSERT INTO sessions (id, created, task, strategy, prompt, final, final_score, status, "
        "rounds, tokens_in, tokens_out, json) VALUES "
        "('legacy', '2026-01-01T00:00:00Z', 'task', 'refine', 'prompt', 'kept', 80, "
        "'ok', 1, 1, 1, '{}')"
    )
    conn.commit()
    conn.close()

    with Store(str(path)) as store:
        tables = {row["name"] for row in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        legacy = store.conn.execute("SELECT final FROM sessions WHERE id = 'legacy'").fetchone()
        assert {"eval_runs", "eval_samples", "profile_promotions", "tune_runs"} <= tables
        assert legacy["final"] == "kept"