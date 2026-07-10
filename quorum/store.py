"""SQLite persistence for quorum.

Follows the claudebudget/jobscope pattern: one ``SCHEMA`` script, an
``sqlite3.Row`` factory, an ``_ensure_columns`` migration hook, and an
``now_iso`` helper. One database holds:

* ``sessions``  - each deliberation, with summary columns + a full JSON blob
* ``ai_cache``  - cached model responses (offline replay + cost savings)
* ``bench``     - per (strategy, task) benchmark rows
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

from .model import Session

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
