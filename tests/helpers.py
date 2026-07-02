"""Shared test helpers: a fully offline config that routes every model to mock."""
from __future__ import annotations

from quorum.config import DEFAULT_CONFIG, _deep_merge


def mock_cfg(db_path: str, **overrides) -> dict:
    cfg = _deep_merge(DEFAULT_CONFIG, {
        "council": {
            "members": [
                {"name": "alice", "provider": "mock", "model": "meta/mock-alice"},
                {"name": "bob", "provider": "mock", "model": "anthropic/mock-bob"},
                {"name": "carol", "provider": "mock", "model": "google/mock-carol"},
            ],
            "judge": "mock:openai/mock-judge",
            "chairman": "mock:meta/mock-chair",
            "aggregator": "mock:meta/mock-agg",
        },
        "run": {"parallel": False, "max_rounds": 4, "target_score": 85},
        "promptsmith": {"enabled": True, "rounds": 1},
        "output": {"db_path": db_path, "dashboard_path": db_path.replace(".db", ".html")},
    })
    return _deep_merge(cfg, overrides)
