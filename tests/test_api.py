from quorum import api


def host_cfg():
    """A host tool's config (ai: block) with quorum enabled, routed to mock."""
    return {
        "ai": {"provider": "mock", "model": "mock/m1", "temperature": 0.3,
               "max_tokens": 200, "api_key_env": ""},
        "quorum": {"enabled": True, "strategy": "refine", "max_rounds": 2},
    }


def test_enabled_gating():
    assert api.enabled(host_cfg()) is True
    off = host_cfg()
    off["quorum"]["enabled"] = False
    assert api.enabled(off) is False


def test_enabled_requires_key_for_remote_provider():
    cfg = {"ai": {"provider": "openrouter", "model": "x/y", "api_key_env": "MISSING_ENV_KEY_XYZ"},
           "quorum": {"enabled": True}}
    assert api.enabled(cfg) is False  # no key in env
    cfg["quorum"]["members"] = [{"name": "a", "provider": "mock", "model": "m/x"}]
    assert api.enabled(cfg) is True   # explicit council -> trusted


def test_build_config_from_host():
    q = api.build_config(host_cfg())
    assert q["run"]["strategy"] == "refine"
    assert q["run"]["max_rounds"] == 2
    assert q["promptsmith"]["enabled"] is False
    assert q["council"]["members"][0]["provider"] == "mock"
    assert q["council"]["judge"] == "mock:mock/m1"


def test_chat_returns_text_via_mock():
    out = api.chat(host_cfg(), None, "You are a precise assistant.", "Say hello.")
    assert isinstance(out, str) and out


def test_chat_returns_none_when_disabled():
    off = host_cfg()
    off["quorum"]["enabled"] = False
    assert api.chat(off, None, "sys", "user") is None


def test_chat_accepts_history_and_context_via_mock():
    out = api.chat(host_cfg(), None, "You are precise.", "follow-up question",
                   history=[{"role": "user", "content": "earlier q"},
                            {"role": "assistant", "content": "earlier a"}],
                   context=[{"title": "Doc", "text": "grounding material"}])
    assert isinstance(out, str) and out


def test_deliberate_with_multi_model_council():
    cfg = host_cfg()
    cfg["quorum"]["members"] = [
        {"name": "a", "provider": "mock", "model": "google/mock-a"},
        {"name": "b", "provider": "mock", "model": "openai/mock-b"},
    ]
    cfg["quorum"]["strategy"] = "debate"
    out = api.deliberate("solve this", system="be rigorous", cfg=cfg)
    assert isinstance(out, str) and out


class _CacheOnlyStore:
    """Mimics a host tool store that implements only the AI cache (no sessions)."""
    def __init__(self):
        self._c = {}

    def ai_cache_get(self, key):
        return self._c.get(key)

    def ai_cache_put(self, key, model, prompt, response):
        self._c[key] = response


def test_accepts_foreign_store_without_session_tables():
    store = _CacheOnlyStore()
    out = api.chat(host_cfg(), store, "sys", "user")
    assert isinstance(out, str) and out  # no crash on missing save_session/tables
