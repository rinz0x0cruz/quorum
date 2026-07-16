from quorum import adapters, api, serveapi
from tests.helpers import mock_cfg


def host_cfg():
    """A host tool's config (ai: block) with quorum enabled, routed to mock."""
    return {
        "ai": {"provider": "mock", "model": "mock/m1", "temperature": 0.3,
               "max_tokens": 200, "api_key_env": ""},
        "quorum": {"enabled": True, "strategy": "refine", "max_rounds": 2},
    }


def test_host_config_maps_ai_and_quorum_blocks():
    q = adapters.host_config(host_cfg())
    assert q["run"]["strategy"] == "refine"
    assert q["run"]["max_rounds"] == 2
    assert q["promptsmith"]["enabled"] is False
    assert q["council"]["members"][0]["provider"] == "mock"
    assert q["council"]["judge"] == "mock:mock/m1"


def test_host_config_maps_rate_limit_and_fallbacks():
    cfg = host_cfg()
    cfg["quorum"].update({
        "rate_limit_rpm": 18,
        "fallbacks": ["mock:mock/fallback"],
    })
    q = adapters.host_config(cfg)
    assert q["run"]["rate_limit_rpm"] == 18
    assert q["run"]["fallbacks"] == ["mock:mock/fallback"]


def test_host_config_maps_exact_model_options():
    cfg = host_cfg()
    cfg["ai"]["model_options"] = {
        "mock/m1": {"reasoning_effort": "low", "include_reasoning": False},
    }
    q = adapters.host_config(cfg)
    assert q["providers"]["mock"]["model_options"] == cfg["ai"]["model_options"]


def test_api_build_config_delegates_to_adapter():
    """api.build_config is now a thin wrapper -- identical output to the adapter."""
    cfg = host_cfg()
    assert api.build_config(cfg) == adapters.host_config(cfg)


def test_split_messages_returns_system_history_last_user():
    system, history, user = adapters.split_messages([
        {"role": "system", "content": "a"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "second"},
    ])
    assert system == "a" and user == "second"
    assert history == [{"role": "user", "content": "first"},
                       {"role": "assistant", "content": "..."}]


def test_split_messages_no_user_returns_empty_last():
    system, history, user = adapters.split_messages([{"role": "system", "content": "s"}])
    assert system == "s" and user == "" and history == []


def test_serveapi_split_is_the_adapter():
    """serveapi._split stays importable and is the shared adapter implementation."""
    assert serveapi._split is adapters.split_messages


def test_select_strategy_uses_named_strategy(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"), run={"strategy": "refine"})
    assert adapters.select_strategy("debate", cfg) == "debate"


def test_select_strategy_falls_back_to_default(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"), run={"strategy": "council"})
    assert adapters.select_strategy("not-a-strategy", cfg) == "council"
    assert adapters.select_strategy("", cfg) == "council"
