from quorum.config import (DEFAULT_CONFIG, _deep_merge, load_config, member_specs,
                           parse_ref, role_spec)


def test_deep_merge_overrides_leaf_keeps_siblings():
    merged = _deep_merge(DEFAULT_CONFIG, {"run": {"target_score": 5}})
    assert merged["run"]["target_score"] == 5
    assert merged["run"]["max_rounds"] == DEFAULT_CONFIG["run"]["max_rounds"]


def test_load_config_defaults():
    cfg = load_config(None)
    assert cfg["run"]["strategy"] == "debate"
    assert "mock" in cfg["providers"]


def test_parse_ref_keeps_model_colons():
    # OpenRouter free-model ids carry a ':free' suffix; only the first colon splits.
    assert parse_ref("openrouter:meta-llama/llama-3.1-8b-instruct:free") == (
        "openrouter", "meta-llama/llama-3.1-8b-instruct:free")


def test_member_and_role_specs():
    cfg = _deep_merge(DEFAULT_CONFIG, {
        "council": {"members": [{"name": "x", "provider": "mock", "model": "m/x"}],
                    "judge": "mock:openai/j"}})
    specs = member_specs(cfg)
    assert len(specs) == 1 and specs[0].ref() == "mock:m/x"
    assert role_spec(cfg, "judge").model == "openai/j"


def test_role_falls_back_to_first_member():
    cfg = _deep_merge(DEFAULT_CONFIG, {
        "council": {"members": [{"name": "x", "provider": "mock", "model": "m/x"}], "chairman": ""}})
    assert role_spec(cfg, "chairman").model == "m/x"
