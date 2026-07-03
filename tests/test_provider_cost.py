import json

from quorum import cost, provider
from quorum.config import member_specs, role_spec
from tests.helpers import mock_cfg


def test_cost_pricing(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    assert cost.price(cfg, "openai/gpt-4o", 1_000_000, 0) == 2.5
    assert cost.price(cfg, "unknown/model", 1_000_000, 1_000_000) == 0.0
    assert cost.count_tokens("abcdefgh") >= 1
    assert cost.over_budget({"cost": {"budget_usd": 0.1}}, 0.2) is True
    assert cost.over_budget({"cost": {"budget_usd": 0}}, 999) is False


def test_mock_completion_and_cache(tmp_path):
    from quorum.store import Store
    cfg = mock_cfg(str(tmp_path / "t.db"))
    prov = provider.for_config(cfg)
    spec = member_specs(cfg)[0]
    with Store(cfg["output"]["db_path"]) as store:
        c1 = prov.complete(spec, [{"role": "user", "content": "hi"}], store=store)
        assert c1.ok and c1.text and c1.tokens_in > 0 and c1.tokens_out > 0
        c2 = prov.complete(spec, [{"role": "user", "content": "hi"}], store=store)
        assert c2.text == c1.text  # served from cache


def test_mock_judge_returns_ramped_json(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    prov = provider.for_config(cfg)
    judge_spec = role_spec(cfg, "judge")
    msg = [{"role": "system", "content": "QUORUM-JUDGE"},
           {"role": "user", "content": "ROUND=2 CANDIDATE A: x CANDIDATE B: y"}]
    payload = json.loads(prov.complete(judge_spec, msg, cache=False).text)
    assert payload["score"] == 85.0 and payload["best"] == "A"


def test_complete_many_preserves_order(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    prov = provider.for_config(cfg)
    specs = member_specs(cfg)
    jobs = [(s, [{"role": "user", "content": f"q{i}"}]) for i, s in enumerate(specs)]
    out = prov.complete_many(jobs, cache=False)
    assert len(out) == len(specs) and all(c.ok for c in out)


def test_fallback_used_when_primary_fails(tmp_path):
    from quorum.model import ModelSpec
    cfg = mock_cfg(str(tmp_path / "t.db"))
    cfg["providers"]["dead"] = {"base_url": "", "api_key_env": ""}  # non-mock, no endpoint -> fails fast
    prov = provider.for_config(cfg)
    primary = ModelSpec(name="x", provider="dead", model="dead/model",
                        fallbacks=[ModelSpec(name="fb", provider="mock", model="mock/fb")])
    comp = prov.complete(primary, [{"role": "user", "content": "hi"}], cache=False)
    assert comp.ok and comp.provider == "mock" and comp.model == "mock/fb"


def test_no_fallback_returns_error(tmp_path):
    from quorum.model import ModelSpec
    cfg = mock_cfg(str(tmp_path / "t.db"))
    cfg["providers"]["dead"] = {"base_url": "", "api_key_env": ""}
    prov = provider.for_config(cfg)
    spec = ModelSpec(name="x", provider="dead", model="dead/model")
    assert prov.complete(spec, [{"role": "user", "content": "hi"}], cache=False).ok is False


class _FakeResp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {}}).encode()


def test_response_format_forwarded_to_http(tmp_path, monkeypatch):
    from quorum.model import ModelSpec
    cfg = mock_cfg(str(tmp_path / "t.db"))
    cfg["providers"]["live"] = {"base_url": "http://example.test/v1", "api_key_env": ""}
    prov = provider.for_config(cfg)
    captured = {}

    def _fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data)
        return _FakeResp()

    monkeypatch.setattr(provider.urllib.request, "urlopen", _fake_urlopen)
    spec = ModelSpec(name="x", provider="live", model="m")
    prov.complete(spec, [{"role": "user", "content": "hi"}],
                  response_format={"type": "json_object"}, cache=False)
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_response_format_absent_by_default(tmp_path, monkeypatch):
    from quorum.model import ModelSpec
    cfg = mock_cfg(str(tmp_path / "t.db"))
    cfg["providers"]["live"] = {"base_url": "http://example.test/v1", "api_key_env": ""}
    prov = provider.for_config(cfg)
    captured = {}

    def _fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data)
        return _FakeResp()

    monkeypatch.setattr(provider.urllib.request, "urlopen", _fake_urlopen)
    spec = ModelSpec(name="x", provider="live", model="m")
    prov.complete(spec, [{"role": "user", "content": "hi"}], cache=False)
    assert "response_format" not in captured["body"]
