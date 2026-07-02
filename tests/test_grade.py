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
