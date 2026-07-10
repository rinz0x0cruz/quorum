"""Least-to-Most strategy (decompose, then solve sub-questions in order) -- offline via mock."""
from quorum import orchestrator, prompts
from quorum.store import Store
from quorum.strategies import leasttomost
from tests.helpers import mock_cfg


def test_leasttomost_decomposes_then_chains(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "a compositional word problem", store=store,
                                        strategy="leasttomost", promptsmith_on=False)
    assert sess.strategy == "leasttomost"
    assert sess.final and sess.final_score > 0
    kinds = [t.kind for r in sess.rounds for t in r.turns]
    assert "decompose" in kinds and "judge" in kinds
    # the mock decomposes into 3 sub-questions -> 3 solve turns, after the decompose
    assert kinds.count("solve") == 3
    assert kinds.index("decompose") < kinds.index("solve")
    assert "least-to-most" in sess.stop_reason


def test_leasttomost_no_members_errors(tmp_path):
    from quorum.config import _deep_merge
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")), {"council": {"members": []}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "q", store=store, strategy="leasttomost",
                                        promptsmith_on=False)
    assert sess.status == "error" and "no members" in sess.stop_reason


def test_parse_steps_variants():
    assert leasttomost._parse_steps("1. first\n2) second\n- third\n* fourth") == \
        ["first", "second", "third", "fourth"]
    assert leasttomost._parse_steps("no list here\njust prose") == []


def test_solve_calls_are_capped(tmp_path, monkeypatch):
    # If decomposition yields more sub-questions than the cap, solves are limited.
    monkeypatch.setattr(leasttomost, "_parse_steps",
                        lambda _text: [f"step {i}" for i in range(1, 20)])
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "big task", store=store, strategy="leasttomost",
                                        promptsmith_on=False)
    kinds = [t.kind for r in sess.rounds for t in r.turns]
    assert kinds.count("solve") == leasttomost._MAX_SUBPROBLEMS


def test_leasttomost_prompt_builders():
    dec = prompts.decompose("", "solve X")
    assert "QUORUM-LTM-DECOMPOSE" in dec[0]["content"] and "solve X" in dec[1]["content"]
    sv = prompts.solve_subproblem("", "solve X", "sub A", [("prior q", "prior a")])
    assert "QUORUM-LTM-SOLVE" in sv[0]["content"]
    assert "sub A" in sv[1]["content"] and "prior a" in sv[1]["content"]


def test_leasttomost_registered():
    from quorum.strategies import get
    assert get("leasttomost") is not None
