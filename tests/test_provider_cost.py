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
