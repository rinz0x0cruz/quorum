"""Throttle telemetry + analyzer (offline)."""
import email.message
import json
import time

from quorum import provider, throttle
from quorum.model import ModelSpec
from quorum.store import Store
from tests.helpers import mock_cfg


def test_store_api_call_roundtrip(tmp_path):
    with Store(str(tmp_path / "t.db")) as store:
        store.add_api_call("openrouter", "m:free", "ok", http_code=200,
                           latency_ms=42, rl_remaining=9)
        rows = store.api_calls_recent()
    assert len(rows) == 1
    assert rows[0]["provider"] == "openrouter"
    assert rows[0]["rl_remaining"] == 9
    assert rows[0]["status"] == "ok"


def test_summarize_counts():
    rows = [
        {"ts": "2026-07-10T10:00:01Z", "provider": "openrouter", "model": "a",
         "status": "ok", "http_code": 200, "latency_ms": 100, "rl_remaining": 5},
        {"ts": "2026-07-10T10:00:01Z", "provider": "openrouter", "model": "a",
         "status": "HTTP 429", "http_code": 429, "latency_ms": 0, "rl_remaining": 0},
        {"ts": "2026-07-10T10:00:01Z", "provider": "openrouter", "model": "a",
         "status": "ok", "http_code": 200, "latency_ms": 200, "rl_remaining": 4},
    ]
    s = throttle.summarize(rows)
    assert s["total"] == 3 and s["throttled"] == 1
    m = s["by_model"]["a"]
    assert m["total"] == 3 and m["ok"] == 2 and m["throttled"] == 1
    assert m["rate_429"] == round(1 / 3, 3)
    assert m["avg_latency_ms"] == 150            # only ok calls counted
    assert s["peak_rpm"]["openrouter"] == 3      # all in the same minute bucket


def test_recommendations_flag_ceiling_and_parallel():
    cfg = {"run": {"parallel": True}, "council": {"members": [{}, {}]}}
    summary = {"total": 40, "throttled": 5, "peak_rpm": {"openrouter": 22},
               "by_model": {"x:free": {"rate_429": 0.2}}}
    recs = " ".join(throttle.recommendations(summary, cfg, None))
    assert "rate_limit_rpm" in recs          # peak >= 20 ceiling
    assert "parallel" in recs                # 429s while parallel


def test_recommendations_quiet_when_clean():
    cfg = {"run": {"parallel": False}, "council": {"members": [{}]}}
    summary = {"total": 5, "throttled": 0, "peak_rpm": {"openrouter": 3}, "by_model": {}}
    recs = throttle.recommendations(summary, cfg, None)
    assert any("No throttling" in r for r in recs)


def test_dashboard_throttle_panel(tmp_path):
    from quorum import render
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        store.add_api_call("openrouter", "m:free", "ok", http_code=200, latency_ms=100, rl_remaining=5)
        store.add_api_call("openrouter", "m:free", "HTTP 429", http_code=429)
        path = render.build(cfg, store)
    with open(path, encoding="utf-8") as fh:
        page = fh.read()
    assert '"throttle"' in page and '"by_model"' in page      # payload carries the summary
    assert "m:free" in page and "renderThrottle" in page       # model + panel renderer present


def test_dashboard_no_throttle_without_telemetry(tmp_path):
    from quorum import render
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        path = render.build(cfg, store)
    with open(path, encoding="utf-8") as fh:
        page = fh.read()
    assert '"throttle": null' in page                          # no telemetry -> panel stays hidden


class _Resp:
    headers = {"X-RateLimit-Limit": "20", "X-RateLimit-Remaining": "7", "X-RateLimit-Reset": "123"}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": "hi"}}],
                           "usage": {"prompt_tokens": 3, "completion_tokens": 2}}).encode()


def test_provider_records_ok_with_ratelimit_headers(tmp_path, monkeypatch):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    cfg["providers"]["live"] = {"base_url": "http://example.test/v1", "api_key_env": ""}
    store = Store(cfg["output"]["db_path"])
    prov = provider.Provider(cfg, telemetry=store)
    monkeypatch.setattr(provider.urllib.request, "urlopen", lambda req, timeout=0: _Resp())
    spec = ModelSpec(name="x", provider="live", model="m")
    comp = prov.complete(spec, [{"role": "user", "content": "hi"}], cache=False)
    assert comp.ok
    rows = store.api_calls_recent()
    store.close()
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["rl_remaining"] == 7


def test_provider_records_throttle(tmp_path, monkeypatch):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    cfg["providers"]["live"] = {"base_url": "http://example.test/v1", "api_key_env": ""}
    store = Store(cfg["output"]["db_path"])
    prov = provider.Provider(cfg, telemetry=store, max_retries=0)  # no retries -> no sleep
    hdrs = email.message.Message()
    hdrs["Retry-After"] = "2"
    hdrs["X-RateLimit-Remaining"] = "0"

    def _boom(req, timeout=0):
        raise provider.urllib.error.HTTPError("http://x", 429, "Too Many", hdrs, None)

    monkeypatch.setattr(provider.urllib.request, "urlopen", _boom)
    spec = ModelSpec(name="x", provider="live", model="m")
    comp = prov.complete(spec, [{"role": "user", "content": "hi"}], cache=False)
    assert comp.ok is False
    rows = store.api_calls_recent()
    store.close()
    assert rows and rows[0]["status"] == "HTTP 429" and rows[0]["http_code"] == 429
    assert rows[0]["retry_after"] == 2.0


def test_rate_limiter_disabled_is_instant():
    rl = provider.RateLimiter(0)
    t0 = time.monotonic()
    for _ in range(5):
        assert rl.acquire() == 0.0
    assert time.monotonic() - t0 < 0.05


def test_rate_limiter_paces_calls():
    rl = provider.RateLimiter(1200)          # 0.05s between calls
    first = rl.acquire()                     # fresh bucket -> no wait
    second = rl.acquire()                    # must wait ~interval
    assert first == 0.0
    assert second > 0.0


def test_providers_share_process_wide_limiter():
    provider.reset_rate_limiters()
    cfg = {"providers": {"p": {"base_url": "http://x", "rpm": 30}}, "run": {}}
    a = provider.Provider(cfg)
    b = provider.Provider(cfg)
    assert a._limiter("p") is b._limiter("p")   # one shared limiter per provider, process-wide
    assert a._limiter("p").rpm == 30
    provider.reset_rate_limiters()


def test_throttle_run_prints_report(tmp_path, capsys):
    store = Store(str(tmp_path / "t.db"))
    for _ in range(3):
        store.add_api_call("openrouter", "a:free", "ok", http_code=200, latency_ms=100,
                           rl_remaining=5)
    store.add_api_call("openrouter", "a:free", "HTTP 429", http_code=429, retry_after=2.0)
    cfg = {"run": {"parallel": True}, "council": {"members": [{}, {}]},
           "providers": {"openrouter": {"base_url": "", "api_key_env": ""}}}
    rc = throttle.run(cfg, store, probe=False)
    out = capsys.readouterr().out
    store.close()
    assert rc == 0
    assert "attempts recorded: 4" in out
    assert "a:free" in out and "recommendations" in out


def test_throttle_run_no_telemetry(tmp_path, capsys):
    store = Store(str(tmp_path / "t.db"))
    rc = throttle.run({"run": {}, "council": {"members": []}, "providers": {}}, store, probe=False)
    out = capsys.readouterr().out
    store.close()
    assert rc == 0 and "no API-call telemetry" in out


def test_key_status_missing_config_returns_none():
    # no base_url/key configured -> None, and no network is attempted
    assert throttle.key_status(
        {"providers": {"openrouter": {"base_url": "", "api_key_env": ""}}}) is None


def test_recommendations_no_ceiling_for_non_openrouter():
    # Groq is not OpenRouter -> the 20/min free ceiling must not be applied to it.
    cfg = {"run": {"parallel": False}, "council": {"members": [{}]},
           "providers": {"groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": ""}}}
    summary = {"total": 40, "throttled": 0, "peak_rpm": {"groq": 25}, "by_model": {}}
    recs = " ".join(throttle.recommendations(summary, cfg, None))
    assert "free ceiling" not in recs and "rate_limit_rpm" not in recs


def test_key_status_unsupported_for_non_openrouter(monkeypatch):
    # /key is OpenRouter-specific -> non-OpenRouter providers short-circuit, no network.
    monkeypatch.setenv("GROQ_TEST_KEY", "k")
    cfg = {"providers": {"groq": {"base_url": "https://api.groq.com/openai/v1",
                                  "api_key_env": "GROQ_TEST_KEY"}}}
    assert throttle.key_status(cfg, "groq") == {"unsupported": "groq"}


def test_throttle_run_non_openrouter_skips_probe_and_ceiling(tmp_path, capsys):
    store = Store(str(tmp_path / "t.db"))
    for _ in range(3):
        store.add_api_call("groq", "llama-3.3-70b-versatile", "ok", http_code=200, latency_ms=100)
    cfg = {"run": {}, "council": {"members": [{}]},
           "providers": {"groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": ""}}}
    rc = throttle.run(cfg, store, provider="groq")   # probe defaults True -> must be gated off
    out = capsys.readouterr().out
    store.close()
    assert rc == 0
    assert "OpenRouter free limit" not in out        # no OpenRouter ceiling label for groq
    assert "quota probe failed" not in out           # probe skipped -> no 404


def test_recommendations_with_free_tier_key():
    summary = {"total": 5, "throttled": 0, "peak_rpm": {"openrouter": 3}, "by_model": {}}
    key = {"is_free_tier": True, "usage": 0, "limit_remaining": 12}
    recs = " ".join(throttle.recommendations(summary, {"run": {}, "council": {"members": []}}, key))
    assert "daily" in recs and "remaining: 12" in recs


def test_429_rotates_to_fallback_fast(tmp_path, monkeypatch):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    cfg["providers"]["live"] = {"base_url": "http://example.test/v1", "api_key_env": ""}
    store = Store(cfg["output"]["db_path"])
    prov = provider.Provider(cfg, telemetry=store, backoff=0)   # backoff=0 -> no real sleeps
    calls = {"primary": 0, "alt": 0}
    hdrs = email.message.Message()
    hdrs["Retry-After"] = "0"

    def _urlopen(req, timeout=0):
        model = json.loads(req.data)["model"]
        if model == "primary/m":
            calls["primary"] += 1
            raise provider.urllib.error.HTTPError("http://x", 429, "Too Many", hdrs, None)
        calls["alt"] += 1
        return _Resp()

    monkeypatch.setattr(provider.urllib.request, "urlopen", _urlopen)
    spec = ModelSpec(name="x", provider="live", model="primary/m",
                     fallbacks=[ModelSpec(name="fb", provider="live", model="alt/m")])
    comp = prov.complete(spec, [{"role": "user", "content": "hi"}], cache=False)
    store.close()
    assert comp.ok and comp.model == "alt/m"
    assert calls["primary"] == 2             # 1 try + 1 retry (retry_429=1), then rotate
    assert calls["alt"] == 1
