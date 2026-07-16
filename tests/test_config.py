from quorum.config import (DEFAULT_CONFIG, _deep_merge, load_config, member_specs,
                           parse_ref, role_spec)


def test_deep_merge_overrides_leaf_keeps_siblings():
    merged = _deep_merge(DEFAULT_CONFIG, {"run": {"target_score": 5}})
    assert merged["run"]["target_score"] == 5
    assert merged["run"]["max_rounds"] == DEFAULT_CONFIG["run"]["max_rounds"]


def test_load_config_defaults():
    cfg = load_config(None)
    assert cfg["run"]["strategy"] == "refine"
    assert "mock" in cfg["providers"]


def test_new_workflows_are_default_off():
    cfg = load_config(None)
    assert cfg["catalog"]["enabled"] is False
    assert cfg["evaluation"]["enabled"] is False
    assert cfg["profiles"]["enabled"] is False
    assert cfg["routing"]["enabled"] is False
    assert cfg["decision"]["mode"] == "single"
    assert cfg["tune"]["enabled"] is False


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


def test_member_fallbacks_from_run_default():
    cfg = _deep_merge(DEFAULT_CONFIG, {
        "council": {"members": [{"name": "x", "provider": "mock", "model": "m/x"}]},
        "run": {"fallbacks": ["openrouter:alt/model"]}})
    fbs = member_specs(cfg)[0].fallbacks
    assert fbs and fbs[0].ref() == "openrouter:alt/model"


def test_member_own_fallbacks_override_run_default():
    cfg = _deep_merge(DEFAULT_CONFIG, {
        "council": {"members": [
            {"name": "x", "provider": "mock", "model": "m/x", "fallbacks": ["groq:own/model"]}]},
        "run": {"fallbacks": ["openrouter:alt/model"]}})
    fbs = member_specs(cfg)[0].fallbacks
    assert len(fbs) == 1 and fbs[0].ref() == "groq:own/model"


def test_role_fallbacks_resolved():
    cfg = _deep_merge(DEFAULT_CONFIG, {
        "council": {"members": [{"name": "x", "provider": "mock", "model": "m/x"}],
                    "judge": "mock:openai/j", "judge_fallbacks": ["openrouter:jf/model"]}})
    fbs = role_spec(cfg, "judge").fallbacks
    assert fbs and fbs[0].ref() == "openrouter:jf/model"


def test_no_fallbacks_by_default():
    assert member_specs(DEFAULT_CONFIG)[0].fallbacks == []
