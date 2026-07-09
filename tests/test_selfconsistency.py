"""Self-consistency strategy (USC selection + adaptive) -- offline via mock."""
import json

from quorum import orchestrator, prompts, provider
from quorum.config import _deep_merge
from quorum.model import ModelSpec
from quorum.store import Store
from tests.helpers import mock_cfg


def test_usc_prompt_and_mock():
    msgs = prompts.usc("what is x", [("a", "answer one"), ("b", "answer two")])
    assert "QUORUM-USC" in msgs[0]["content"]
    assert "CANDIDATE A" in msgs[1]["content"] and "CANDIDATE B" in msgs[1]["content"]
    # the mock provider returns a JSON choice for a USC system prompt
    resp = provider.MockResponder().respond(ModelSpec("x", "mock", "m"), msgs)
    assert json.loads(resp)["choice"] == "A"


def test_selfconsistency_unanimous(tmp_path):
    # identical mock samples -> a single consensus bucket -> one selected answer
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")), {"run": {"samples": 3}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "q", store=store, strategy="selfconsistency",
                                        promptsmith_on=False)
    assert sess.strategy == "selfconsistency"
    assert sess.final and sess.final_score > 0
    assert "self-consistency" in sess.stop_reason


def test_selfconsistency_usc_selects_when_distinct(tmp_path, monkeypatch):
    answers = ["the sky is blue and clouds drift white overhead today",
               "bananas grow on tall tropical plants needing much heat",
               "quantum entanglement links two distant particle states firmly"]
    counter = {"n": 0}

    def _distinct(self, spec, user):
        i = counter["n"]
        counter["n"] += 1
        return answers[i % len(answers)]

    monkeypatch.setattr(provider.MockResponder, "_propose", _distinct)
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")),
                      {"run": {"samples": 3, "adaptive_samples": False}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "q", store=store, strategy="selfconsistency",
                                        promptsmith_on=False)
    selects = [t for r in sess.rounds for t in r.turns if t.kind == "select"]
    assert len(selects) == 1                       # USC was invoked over 3 distinct answers
    assert "USC selection" in sess.stop_reason
    assert sess.final == answers[0]                # mock USC picks CANDIDATE A


def test_selfconsistency_adaptive_early_stops(tmp_path):
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")),
                      {"run": {"adaptive_samples": True, "samples": 10, "samples_min": 2}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "q", store=store, strategy="selfconsistency",
                                        promptsmith_on=False)
    proposes = [t for r in sess.rounds for t in r.turns if t.kind == "propose"]
    assert len(proposes) == 2                       # identical mock samples -> stop at samples_min
    assert sess.final and sess.final_score > 0


def test_selfconsistency_no_members_errors(tmp_path):
    cfg = _deep_merge(mock_cfg(str(tmp_path / "t.db")), {"council": {"members": []}})
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "q", store=store, strategy="selfconsistency",
                                        promptsmith_on=False)
    assert sess.status == "error" and "no members" in sess.stop_reason
