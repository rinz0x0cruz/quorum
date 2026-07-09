"""Rigorous validation: real concurrency (thread-safe telemetry + limiter pacing),
edge cases, and config-file warnings. Offline/deterministic."""
import email.message
import json
import threading
import time

from quorum import consistency, provider, throttle
from quorum.config import load_config
from quorum.model import ModelSpec
from quorum.store import Store
from tests.helpers import mock_cfg


class _OkResp:
    headers = {"X-RateLimit-Limit": "20", "X-RateLimit-Remaining": "9"}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": "ok"}}],
                           "usage": {"prompt_tokens": 2, "completion_tokens": 1}}).encode()


# --- thread safety: parallel fan-out records every attempt, no loss/corruption ---
def test_parallel_fanout_records_all_telemetry(tmp_path, monkeypatch):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    cfg["run"]["parallel"] = True
    cfg["providers"]["live"] = {"base_url": "http://example.test/v1", "api_key_env": ""}
    store = Store(cfg["output"]["db_path"])
    prov = provider.Provider(cfg, telemetry=store)

    def _urlopen(req, timeout=0):
        time.sleep(0.003)          # force worker threads to overlap
        return _OkResp()

    monkeypatch.setattr(provider.urllib.request, "urlopen", _urlopen)
    jobs = [(ModelSpec(f"m{i}", "live", f"model/{i}"), [{"role": "user", "content": "hi"}])
            for i in range(8)]
    comps = prov.complete_many(jobs, cache=False)
    rows = store.api_calls_recent()
    store.close()
    assert len(comps) == 8 and all(c.ok for c in comps)
    assert len(rows) == 8                       # every parallel attempt persisted safely
    assert {r["model"] for r in rows} == {f"model/{i}" for i in range(8)}


def test_rate_limiter_paces_under_concurrency():
    rl = provider.RateLimiter(1200)             # 0.05s interval
    waits = []
    lock = threading.Lock()

    def worker():
        w = rl.acquire()
        with lock:
            waits.append(w)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.18                       # 5 calls spaced by 0.05s -> ~0.2s
    assert sum(1 for w in waits if w == 0.0) == 1  # exactly one immediate slot


def test_store_concurrent_api_call_writes(tmp_path):
    store = Store(str(tmp_path / "t.db"))

    def writer(n):
        for _ in range(25):
            store.add_api_call("openrouter", f"m{n}", "ok", http_code=200)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = store.api_calls_recent(limit=1000)
    store.close()
    assert len(rows) == 100                      # 4 threads x 25 writes, none lost


# --- edge cases -----------------------------------------------------------
def test_throttle_summarize_empty():
    s = throttle.summarize([])
    assert s["total"] == 0 and s["by_model"] == {} and s["throttled"] == 0
    assert throttle.recommendations(s, {"run": {}, "council": {"members": []}}, None)


def test_consistency_numeric_majority_vote():
    cl = consistency.cluster(["the total is 42", "so 42 apples", "i get 42", "maybe 7 only"])
    top = consistency.leader(cl)
    assert top["key"] == "#42" and top["count"] == 3


def test_429_then_success_records_two_rows(tmp_path, monkeypatch):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    cfg["providers"]["live"] = {"base_url": "http://example.test/v1", "api_key_env": ""}
    store = Store(cfg["output"]["db_path"])
    prov = provider.Provider(cfg, telemetry=store, backoff=0)
    hdrs = email.message.Message()
    hdrs["Retry-After"] = "0"
    state = {"first": True}

    def _urlopen(req, timeout=0):
        if state["first"]:
            state["first"] = False
            raise provider.urllib.error.HTTPError("http://x", 429, "Too Many", hdrs, None)
        return _OkResp()

    monkeypatch.setattr(provider.urllib.request, "urlopen", _urlopen)
    comp = prov.complete(ModelSpec("x", "live", "m"), [{"role": "user", "content": "hi"}],
                         cache=False)
    rows = store.api_calls_recent()
    store.close()
    assert comp.ok                               # retried after the 429 and succeeded
    assert len(rows) == 2
    assert {r["status"] for r in rows} == {"HTTP 429", "ok"}


# --- config-file warnings (end to end via load_config) --------------------
def test_load_config_warns_on_typos(tmp_path, capsys):
    p = tmp_path / "config.yaml"
    p.write_text("run:\n  max_round: 5\n  rate_limit_rpm: 18\nbogus_key: 1\n", encoding="utf-8")
    cfg = load_config(str(p), warn=True)
    err = capsys.readouterr().err
    assert "run.max_round" in err and "bogus_key" in err
    assert "rate_limit_rpm" not in err           # real knob is not flagged
    assert cfg["run"]["max_rounds"] == 4         # typo ignored -> default preserved
    assert cfg["run"]["rate_limit_rpm"] == 18    # valid knob applied


def test_urlerror_records_error_telemetry(tmp_path, monkeypatch):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    cfg["providers"]["live"] = {"base_url": "http://example.test/v1", "api_key_env": ""}
    store = Store(cfg["output"]["db_path"])
    prov = provider.Provider(cfg, telemetry=store, backoff=0, max_retries=0)

    def _boom(req, timeout=0):
        raise provider.urllib.error.URLError("connection refused")

    monkeypatch.setattr(provider.urllib.request, "urlopen", _boom)
    comp = prov.complete(ModelSpec("x", "live", "m"), [{"role": "user", "content": "hi"}],
                         cache=False)
    rows = store.api_calls_recent()
    store.close()
    assert comp.ok is False
    assert rows and rows[0]["status"] == "error" and rows[0]["http_code"] == 0
