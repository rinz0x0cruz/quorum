"""Structured events (#8): typed Event stream via on_event, string back-compat."""
from quorum import events, orchestrator
from quorum.config import _deep_merge
from quorum.store import Store
from tests.helpers import mock_cfg


def test_event_render_and_coerce():
    e = events.Event("round", "round 1: score 85", round=1, data={"score": 85})
    assert events.render(e) == "round 1: score 85"
    log = events.coerce("hello")
    assert log.kind == "log" and log.message == "hello"
    assert events.coerce(e) is e                     # already an Event -> unchanged


def test_run_session_streams_structured_events(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    seen: list = []
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "solve it", store=store, strategy="refine",
                                        on_event=seen.append)
    kinds = {e.kind for e in seen}
    assert {"phase", "round", "done"} <= kinds
    rounds = [e for e in seen if e.kind == "round"]
    assert rounds and all("score" in e.data for e in rounds)   # structured per-round score
    done = [e for e in seen if e.kind == "done"][-1]
    assert done.data["status"] == sess.status and done.data["score"] == sess.final_score


def test_string_emits_become_log_events(tmp_path):
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")),
                      {"run": {"judge_every": 2, "max_rounds": 4, "target_score": 200}})
    seen: list = []
    with Store(cfg["output"]["db_path"]) as store:
        orchestrator.run_session(cfg, "q", store=store, strategy="refine",
                                 promptsmith_on=False, on_event=seen.append)
    assert any(e.kind == "log" and "deferred judge" in e.message for e in seen)


def test_on_event_failure_never_breaks_run(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))

    def boom(_e):
        raise RuntimeError("observer crashed")

    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "q", store=store, strategy="refine",
                                        promptsmith_on=False, on_event=boom)
    assert sess.final and sess.status == "ok"        # a broken observer can't break the run


def test_verbose_cli_output_unchanged(tmp_path, capsys):
    # No on_event, verbose=True -> the same one-line messages still print.
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        orchestrator.run_session(cfg, "q", store=store, strategy="refine",
                                 promptsmith_on=False, verbose=True)
    out = capsys.readouterr().out
    assert "round 1: score" in out                   # human line preserved
