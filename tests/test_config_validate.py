"""Config known-keys validator (#7) -- warns on typos, never fatal."""
from quorum.config import DEFAULT_CONFIG, validate_config


def test_validate_accepts_clean_config():
    assert validate_config({"run": {"max_rounds": 5}, "judge": {"json_mode": True}}) == []


def test_validate_flags_typos():
    warns = validate_config({"run": {"max_round": 5}, "collncil": {}})
    assert "run.max_round" in warns and "collncil" in warns


def test_validate_allows_open_subtrees():
    cfg = {"providers": {"myllm": {"base_url": "http://x", "custom": 1}},
           "cost": {"pricing": {"my/model": {"input": 1}}},
           "judge": {"rubric": {"my_criterion": 0.5}}}
    assert validate_config(cfg) == []


def test_validate_default_is_self_clean():
    assert validate_config(DEFAULT_CONFIG) == []
