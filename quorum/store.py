"""SQLite persistence for quorum.

Follows the claudebudget/jobscope pattern: one ``SCHEMA`` script, an
``sqlite3.Row`` factory, an ``_ensure_columns`` migration hook, and an
``now_iso`` helper. One database holds:

* ``sessions``  - each deliberation, with summary columns + a full JSON blob
* ``ai_cache``  - cached model responses (offline replay + cost savings)
* ``bench``     - per (strategy, task) benchmark rows
* ``eval_*``    - reproducible model/profile evaluation manifests + samples
* ``profile_promotions`` - append-only approvals backed by an evaluation run
* ``tune_runs`` - optional prompt/router/weight tuning job provenance
* ``runs``      - a lightweight audit trail

Everything is local to this machine; nothing is uploaded.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Optional

from .model import (EvaluationRun, EvaluationSample, ProfilePromotion, Session,
                    TuneRun)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created TEXT,
    task TEXT,
    strategy TEXT,
    prompt TEXT,
    final TEXT,
    final_score REAL DEFAULT 0,
    status TEXT,
    stop_reason TEXT,
    rounds INTEGER DEFAULT 0,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    json TEXT
);
CREATE TABLE IF NOT EXISTS ai_cache (
    key TEXT PRIMARY KEY,
    model TEXT,
    prompt TEXT,
    response TEXT,
    ts TEXT
);
CREATE TABLE IF NOT EXISTS bench (
    ts TEXT,
    strategy TEXT,
    task_id TEXT,
    score REAL,
    rounds INTEGER,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd REAL,
    seconds REAL,
    match REAL,
    correct INTEGER
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    created TEXT,
    completed TEXT,
    target_type TEXT,
    target_id TEXT,
    pack_id TEXT,
    pack_version TEXT,
    split TEXT,
    status TEXT,
    json TEXT
);
CREATE TABLE IF NOT EXISTS eval_samples (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    created TEXT,
    task_id TEXT,
    repeat_index INTEGER DEFAULT 0,
    requested_ref TEXT,
    actual_ref TEXT,
    status TEXT,
    score REAL DEFAULT 0,
    match REAL,
    correct INTEGER,
    latency_ms INTEGER DEFAULT 0,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    json TEXT
);
CREATE TABLE IF NOT EXISTS profile_promotions (
    id TEXT PRIMARY KEY,
    created TEXT,
    profile_name TEXT,
    profile_version TEXT,
    eval_run_id TEXT,
    json TEXT
);
CREATE TABLE IF NOT EXISTS tune_runs (
    id TEXT PRIMARY KEY,
    created TEXT,
    completed TEXT,
    method TEXT,
    backend TEXT,
    base_model TEXT,
    status TEXT,
    json TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    ts TEXT,
    action TEXT,
    count INTEGER,
    status TEXT
);
CREATE TABLE IF NOT EXISTS api_calls (
    ts TEXT,
    provider TEXT,
    model TEXT,
    status TEXT,
    http_code INTEGER DEFAULT 0,
    attempt INTEGER DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    retry_after REAL DEFAULT 0,
    rl_limit INTEGER DEFAULT 0,
    rl_remaining INTEGER DEFAULT 0,
    rl_reset TEXT,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created DESC);
CREATE INDEX IF NOT EXISTS idx_bench_strategy ON bench(strategy);
CREATE INDEX IF NOT EXISTS idx_eval_runs_pack ON eval_runs(pack_id, split);
CREATE INDEX IF NOT EXISTS idx_eval_samples_run ON eval_samples(run_id);
CREATE INDEX IF NOT EXISTS idx_profile_promotions_name ON profile_promotions(profile_name, created DESC);
CREATE INDEX IF NOT EXISTS idx_tune_runs_created ON tune_runs(created DESC);
CREATE INDEX IF NOT EXISTS idx_api_calls_ts ON api_calls(ts DESC);
CREATE INDEX IF NOT EXISTS idx_api_calls_model ON api_calls(model);
"""


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Store:
    def __init__(self, path: str):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self.path = path
        # check_same_thread=False + a lock lets the parallel provider fan-out
        # record telemetry (and cache) from worker threads safely.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._ensure_columns()

    def _ensure_columns(self) -> None:
        """Additive migrations for older databases (mirrors the sibling tools)."""
        existing = {r["name"] for r in self.conn.execute("PRAGMA table_info(sessions)")}
        for col, ddl in (("stop_reason", "TEXT"), ("cost_usd", "REAL DEFAULT 0")):
            if col not in existing:
                self.conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {ddl}")
        bench_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(bench)")}
        for col, ddl in (("match", "REAL"), ("correct", "INTEGER")):
            if col not in bench_cols:
                self.conn.execute(f"ALTER TABLE bench ADD COLUMN {col} {ddl}")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- sessions -------------------------------------------------------
    def save_session(self, session: Session) -> None:
        d = session.to_dict()
        self.conn.execute(
            "INSERT INTO sessions (id, created, task, strategy, prompt, final, final_score, "
            "status, stop_reason, rounds, tokens_in, tokens_out, cost_usd, json) "
            "VALUES (:id,:created,:task,:strategy,:prompt,:final,:final_score,:status,"
            ":stop_reason,:rounds,:tokens_in,:tokens_out,:cost_usd,:json) "
            "ON CONFLICT(id) DO UPDATE SET final=excluded.final, final_score=excluded.final_score, "
            "status=excluded.status, stop_reason=excluded.stop_reason, rounds=excluded.rounds, "
            "tokens_in=excluded.tokens_in, tokens_out=excluded.tokens_out, "
            "cost_usd=excluded.cost_usd, json=excluded.json",
            {
                "id": d["id"], "created": d["created"], "task": d["task"],
                "strategy": d["strategy"], "prompt": d["prompt"], "final": d["final"],
                "final_score": d["final_score"], "status": d["status"],
                "stop_reason": d["stop_reason"], "rounds": len(d["rounds"]),
                "tokens_in": d["tokens_in"], "tokens_out": d["tokens_out"],
                "cost_usd": d["cost_usd"], "json": json.dumps(d),
            },
        )
        self.conn.commit()

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        r = self.conn.execute("SELECT json FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return json.loads(r["json"]) if r else None

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, created, task, strategy, final_score, rounds, cost_usd, status "
            "FROM sessions ORDER BY created DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def recent_sessions(self, limit: int = 25) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT json FROM sessions ORDER BY created DESC LIMIT ?", (limit,)
        ).fetchall()
        return [json.loads(r["json"]) for r in rows]

    def session_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"])

    def top_sessions(self, limit: int = 3, min_score: float = 80.0) -> list[dict[str, Any]]:
        """Highest-scoring past deliberations (with a solve-prompt), for few-shot
        bootstrapping the prompt engineer."""
        rows = self.conn.execute(
            "SELECT task, prompt, final_score FROM sessions "
            "WHERE final_score >= ? AND prompt IS NOT NULL AND prompt != '' "
            "ORDER BY final_score DESC LIMIT ?", (min_score, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- AI response cache ---------------------------------------------
    def ai_cache_get(self, key: str) -> Optional[str]:
        with self._lock:
            r = self.conn.execute("SELECT response FROM ai_cache WHERE key = ?", (key,)).fetchone()
        return r["response"] if r else None

    def ai_cache_put(self, key: str, model: str, prompt: str, response: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO ai_cache (key, model, prompt, response, ts) VALUES (?,?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET response=excluded.response, ts=excluded.ts",
                (key, model, prompt, response, now_iso()),
            )
            self.conn.commit()

    # ---- benchmark ------------------------------------------------------
    def add_bench_row(self, strategy: str, task_id: str, score: float, rounds: int,
                      tokens_in: int, tokens_out: int, cost_usd: float, seconds: float,
                      match: Optional[float] = None, correct: Optional[bool] = None) -> None:
        self.conn.execute(
            "INSERT INTO bench (ts, strategy, task_id, score, rounds, tokens_in, tokens_out, "
            "cost_usd, seconds, match, correct) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (now_iso(), strategy, task_id, score, rounds, tokens_in, tokens_out, cost_usd, seconds,
             match, None if correct is None else int(correct)),
        )
        self.conn.commit()

    def bench_rows(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM bench ORDER BY ts DESC").fetchall()
        return [dict(r) for r in rows]

    # ---- model/profile evaluation --------------------------------------
    def save_eval_run(self, run: EvaluationRun) -> None:
        """Create or update an evaluation run's lifecycle and manifest."""
        d = run.to_dict()
        with self._lock:
            existing = self.conn.execute(
                "SELECT json FROM eval_runs WHERE id = ?", (d["id"],)
            ).fetchone()
            if existing:
                prior = json.loads(existing["json"])
                immutable = ("created", "target_type", "target_id", "pack_id",
                             "pack_version", "split", "manifest")
                if any(prior.get(key) != d.get(key) for key in immutable):
                    raise ValueError(f"evaluation run '{d['id']}' manifest is immutable")
            self.conn.execute(
                "INSERT INTO eval_runs (id, created, completed, target_type, target_id, "
                "pack_id, pack_version, split, status, json) VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET completed=excluded.completed, "
                "status=excluded.status, json=excluded.json",
                (d["id"], d["created"], d["completed"], d["target_type"], d["target_id"],
                 d["pack_id"], d["pack_version"], d["split"], d["status"], json.dumps(d)),
            )
            self.conn.commit()

    def get_eval_run(self, run_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("SELECT json FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
        return json.loads(row["json"]) if row else None

    def eval_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT json FROM eval_runs ORDER BY created DESC LIMIT ?", (limit,)
        ).fetchall()
        return [json.loads(r["json"]) for r in rows]

    def save_eval_sample(self, sample: EvaluationSample) -> None:
        """Upsert one target/task/repeat result; safe for parallel evaluators."""
        d = sample.to_dict()
        with self._lock:
            existing = self.conn.execute(
                "SELECT json FROM eval_samples WHERE id = ?", (d["id"],)
            ).fetchone()
            if existing:
                prior = json.loads(existing["json"])
                immutable = ("run_id", "created", "task_id", "repeat_index", "requested_ref")
                if any(prior.get(key) != d.get(key) for key in immutable):
                    raise ValueError(f"evaluation sample '{d['id']}' identity is immutable")
            self.conn.execute(
                "INSERT INTO eval_samples (id, run_id, created, task_id, repeat_index, "
                "requested_ref, actual_ref, status, score, match, correct, latency_ms, "
                "tokens_in, tokens_out, cost_usd, json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET actual_ref=excluded.actual_ref, "
                "status=excluded.status, score=excluded.score, match=excluded.match, "
                "correct=excluded.correct, latency_ms=excluded.latency_ms, "
                "tokens_in=excluded.tokens_in, tokens_out=excluded.tokens_out, "
                "cost_usd=excluded.cost_usd, json=excluded.json",
                (d["id"], d["run_id"], d["created"], d["task_id"], d["repeat_index"],
                 d["requested_ref"], d["actual_ref"], d["status"], d["score"], d["match"],
                 None if d["correct"] is None else int(d["correct"]), d["latency_ms"],
                 d["tokens_in"], d["tokens_out"], d["cost_usd"], json.dumps(d)),
            )
            self.conn.commit()

    def eval_samples(self, run_id: Optional[str] = None) -> list[dict[str, Any]]:
        if run_id is None:
            rows = self.conn.execute(
                "SELECT json FROM eval_samples ORDER BY created, id"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT json FROM eval_samples WHERE run_id = ? ORDER BY created, id", (run_id,)
            ).fetchall()
        return [json.loads(r["json"]) for r in rows]

    # ---- approved profiles --------------------------------------------
    def add_profile_promotion(self, promotion: ProfilePromotion) -> None:
        """Append an approval record; duplicate ids are rejected, never overwritten."""
        d = promotion.to_dict()
        with self._lock:
            self.conn.execute(
                "INSERT INTO profile_promotions (id, created, profile_name, profile_version, "
                "eval_run_id, json) VALUES (?,?,?,?,?,?)",
                (d["id"], d["created"], d["profile_name"], d["profile_version"],
                 d["eval_run_id"], json.dumps(d)),
            )
            self.conn.commit()

    def profile_promotions(self, profile_name: Optional[str] = None) -> list[dict[str, Any]]:
        if profile_name is None:
            rows = self.conn.execute(
                "SELECT json FROM profile_promotions ORDER BY created DESC"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT json FROM profile_promotions WHERE profile_name = ? ORDER BY created DESC",
                (profile_name,),
            ).fetchall()
        return [json.loads(r["json"]) for r in rows]

    # ---- optional tuning jobs -----------------------------------------
    def save_tune_run(self, run: TuneRun) -> None:
        """Create or update an optional tuning job without loading its backend."""
        d = run.to_dict()
        with self._lock:
            existing = self.conn.execute(
                "SELECT json FROM tune_runs WHERE id = ?", (d["id"],)
            ).fetchone()
            if existing:
                prior = json.loads(existing["json"])
                immutable = ("created", "method", "backend", "base_model", "manifest")
                if any(prior.get(key) != d.get(key) for key in immutable):
                    raise ValueError(f"tune run '{d['id']}' manifest is immutable")
            self.conn.execute(
                "INSERT INTO tune_runs (id, created, completed, method, backend, base_model, "
                "status, json) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET completed=excluded.completed, "
                "status=excluded.status, json=excluded.json",
                (d["id"], d["created"], d["completed"], d["method"], d["backend"],
                 d["base_model"], d["status"], json.dumps(d)),
            )
            self.conn.commit()

    def get_tune_run(self, run_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("SELECT json FROM tune_runs WHERE id = ?", (run_id,)).fetchone()
        return json.loads(row["json"]) if row else None

    def tune_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT json FROM tune_runs ORDER BY created DESC LIMIT ?", (limit,)
        ).fetchall()
        return [json.loads(r["json"]) for r in rows]

    # ---- runs -----------------------------------------------------------
    def add_run(self, action: str, count: int, status: str = "ok") -> None:
        self.conn.execute(
            "INSERT INTO runs (ts, action, count, status) VALUES (?,?,?,?)",
            (now_iso(), action, count, status),
        )
        self.conn.commit()

    # ---- API-call telemetry (throttle analysis) ------------------------
    def add_api_call(self, provider: str, model: str, status: str, *, http_code: int = 0,
                     attempt: int = 0, latency_ms: int = 0, retry_after: float = 0.0,
                     rl_limit: int = 0, rl_remaining: int = 0, rl_reset: str = "",
                     tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Record one HTTP attempt. Thread-safe (called from parallel fan-out)."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO api_calls (ts, provider, model, status, http_code, attempt, "
                "latency_ms, retry_after, rl_limit, rl_remaining, rl_reset, tokens_in, tokens_out) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (now_iso(), provider, model, status, http_code, attempt, latency_ms,
                 retry_after, rl_limit, rl_remaining, rl_reset, tokens_in, tokens_out),
            )
            self.conn.commit()

    def api_calls_recent(self, limit: int = 5000) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM api_calls ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
