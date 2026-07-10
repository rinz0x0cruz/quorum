from quorum import grade, provider
from tests.helpers import mock_cfg


def test_extract_gold_variants():
    assert grade.extract_gold("long work ...\n#### 18") == "18"
    assert grade.extract_gold("#### 1,024") == "1024"
    assert grade.extract_gold("42") == "42"
    assert grade.extract_gold("a long prose answer, no gold number") is None


def test_final_number():
    assert grade.final_number("so the total is 27 apples") == "27"
    assert grade.final_number("no digits here") is None


def test_numeric_match():
    assert grade.numeric_match("... therefore 18 total.", "#### 18") is True
    assert grade.numeric_match("I think it is 17.", "#### 18") is False
    assert grade.numeric_match("prose answer", "prose reference") is None


def test_grade_numeric_is_deterministic_and_free(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    prov = provider.for_config(cfg)
    score, correct, turn = grade.grade(cfg, prov, "q", "the answer is 18", "#### 18")
    assert score == 100.0 and correct is True and turn is None  # no model call
    score, correct, turn = grade.grade(cfg, prov, "q", "the answer is 5", "#### 18")
    assert score == 0.0 and correct is False and turn is None


def test_grade_prose_uses_grader_model(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    prov = provider.for_config(cfg)
    score, correct, turn = grade.grade(cfg, prov, "q", "some answer", "a prose reference answer here")
    assert score == 90.0 and correct is True
    assert turn is not None and turn.kind == "grade"


def test_bench_grades_against_reference(tmp_path):
    from quorum import bench
    from quorum.store import Store
    cfg = mock_cfg(str(tmp_path / "t.db"))
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text("tasks:\n  - id: m1\n    task: what is 2+2\n    reference: '#### 4'\n", encoding="utf-8")
    with Store(cfg["output"]["db_path"]) as store:
        rc = bench.run(cfg, str(tasks), ["refine", "ensemble"], store, verbose=False)
        assert rc == 0
        assert len(store.bench_rows()) == 2


def test_final_answer_extraction():
    assert grade.final_answer("reasoning...\n#### 42") == "42"
    assert grade.final_answer("lots of work\nFinal answer: B") == "B"
    assert grade.final_answer("blah\nThe answer is Canberra.") == "Canberra."
    assert grade.final_answer("only one line") == "only one line"
    assert grade.final_answer("") == ""


def test_deterministic_match_choice():
    assert grade.deterministic_match("I pick Answer: B", "B", "choice") is True
    assert grade.deterministic_match("Answer: (C)", "C", "choice") is True
    assert grade.deterministic_match("Answer: A", "B", "choice") is False
    assert grade.deterministic_match("no letter here at all", "B", "choice") is False
    # gold has no letter -> not applicable (None -> AI grader)
    assert grade.deterministic_match("Answer: B", "prose gold", "choice") is None


def test_deterministic_match_boolean():
    assert grade.deterministic_match("...therefore Answer: yes", "yes", "boolean") is True
    assert grade.deterministic_match("Answer: no", "false", "boolean") is True
    assert grade.deterministic_match("Answer: no", "yes", "boolean") is False
    assert grade.deterministic_match("maybe, unsure", "yes", "boolean") is False


def test_deterministic_match_exact_and_contains():
    assert grade.deterministic_match("Answer: Tokyo.", "tokyo", "exact") is True
    assert grade.deterministic_match("The capital is Tokyo", "Tokyo", "exact") is True
    assert grade.deterministic_match("Answer: Kyoto", "Tokyo", "exact") is False
    assert grade.deterministic_match("the value is DECREASE overall", "decrease", "contains") is True
    assert grade.deterministic_match("no match", "decrease", "contains") is False


def test_deterministic_match_regex_and_dict_spec():
    assert grade.deterministic_match("total = 42 units", r"\b42\b", "regex") is True
    assert grade.deterministic_match("total = 41 units", r"\b42\b", "regex") is False
    assert grade.deterministic_match("bad (regex", "(", "regex") is None  # invalid pattern -> AI grader
    # dict form + unknown kind
    assert grade.deterministic_match("Answer: B", "B", {"type": "choice"}) is True
    assert grade.deterministic_match("x", "y", "unknown-kind") is None
    # numeric still auto-detected when no match spec given
    assert grade.deterministic_match("hence 18", "#### 18", None) is True


def test_grade_choice_is_deterministic_and_free(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    prov = provider.for_config(cfg)
    score, correct, turn = grade.grade(cfg, prov, "q", "Answer: B", "B", match="choice")
    assert score == 100.0 and correct is True and turn is None  # no model call
    score, correct, turn = grade.grade(cfg, prov, "q", "Answer: A", "B", match="choice")
    assert score == 0.0 and correct is False and turn is None


def test_bench_grades_choice_task(tmp_path):
    from quorum import bench
    from quorum.store import Store
    cfg = mock_cfg(str(tmp_path / "t.db"))
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text("tasks:\n  - id: c1\n    task: pick one\n    answer: B\n    match: choice\n",
                     encoding="utf-8")
    with Store(cfg["output"]["db_path"]) as store:
        rc = bench.run(cfg, str(tasks), ["refine"], store, verbose=False)
        assert rc == 0
        rows = store.bench_rows()
        assert len(rows) == 1
        # deterministic grading adds no grader cost (turn is None)
        assert rows[0]["cost_usd"] == 0.0


def test_resolve_builtin_eval_name(tmp_path, monkeypatch):
    from quorum import bench
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "mini.yaml").write_text("tasks:\n  - id: a\n    task: q\n    answer: '#### 1'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert bench._resolve_tasks_path("mini").endswith("mini.yaml")
    assert len(bench._load_tasks("mini")) == 1
    assert bench._resolve_tasks_path("does-not-exist") == "does-not-exist"  # real paths untouched


def test_shipped_eval_sets_are_wellformed():
    import glob
    import os
    from quorum import bench
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = glob.glob(os.path.join(root, "evals", "*.yaml"))
    assert len(files) >= 2, files  # reasoning + reasoning-hard
    valid = {None, "numeric", "number", "num", "choice", "mc", "multiple_choice",
             "boolean", "bool", "yesno", "yes_no", "exact", "contains", "regex"}
    for path in files:
        tasks = bench._load_tasks(path)
        assert len(tasks) >= 8, path
        for t in tasks:
            assert t["task"] and t["reference"], (path, t)
            assert t.get("match") in valid, (path, t)
