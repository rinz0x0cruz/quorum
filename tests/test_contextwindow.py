from quorum import contextwindow as cw
from quorum.contextwindow import ContextDoc


def test_select_ranks_by_lexical_overlap():
    docs = [
        {"id": "a", "title": "SimpleHelp bypass", "text": "SimpleHelp authentication bypass CVE"},
        {"id": "b", "title": "Unrelated", "text": "kitchen recipes and gardening"},
    ]
    ranked = cw.select("SimpleHelp authentication bypass exploit", docs, k=2)
    assert ranked[0].id == "a" and ranked[0].score > ranked[1].score


def test_select_top_k_limit():
    docs = [{"id": str(i), "text": f"doc {i} shared"} for i in range(5)]
    assert len(cw.select("shared", docs, k=2)) == 2


def test_pack_respects_history_turns():
    history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
    kept, _ = cw.pack(history, [], budget_tokens=100_000, history_turns=3)
    assert len(kept) == 3 and kept[-1]["content"] == "msg 9"  # most-recent, in order


def test_pack_budget_keeps_best_doc_first():
    docs = [ContextDoc(id="big", text="word " * 10_000, score=1.0),
            ContextDoc(id="small", text="tiny", score=0.9)]
    _, kept = cw.pack([], docs, budget_tokens=50, history_turns=8)
    assert kept and kept[0].id == "big" and len(kept) == 1  # small no longer fits


def test_preamble_empty_without_inputs():
    assert cw.preamble({}, history=None, context=None) == ""


def test_preamble_frames_as_data_and_includes_both():
    cfg = {"context": {"budget_tokens": 4000, "history_turns": 8}}
    pre = cw.preamble(cfg, history=[{"role": "user", "content": "hi there"}],
                      context=[{"title": "Story", "text": "prior story text"}])
    assert "DATA ONLY" in pre
    assert "CONVERSATION SO FAR" in pre and "hi there" in pre
    assert "REFERENCE CONTEXT" in pre and "prior story text" in pre
