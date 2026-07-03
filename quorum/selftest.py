"""Offline self-tests for quorum (no network, no keys, no browser).

Mirrors the claudebudget/jobscope/learnscope ``selftest``: a fast confidence
check that the deterministic engine works on a fresh machine using the built-in
``mock`` provider. Returns 0 on success, 1 on failure.
"""
from __future__ import annotations

import os
import tempfile


class _Check:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def ok(self, name: str, cond: bool, detail: str = "") -> None:
        mark = "PASS" if cond else "FAIL"
        line = f"  [{mark}] {name}"
        if detail and not cond:
            line += f"  ({detail})"
        print(line)
        self.passed += int(bool(cond))
        self.failed += int(not cond)


def _mock_cfg(db_path: str) -> dict:
    """A fully offline config: every model routes to the ``mock`` provider."""
    from .config import DEFAULT_CONFIG, _deep_merge
    return _deep_merge(DEFAULT_CONFIG, {
        "council": {
            "members": [
                {"name": "alice", "provider": "mock", "model": "mock/alice"},
                {"name": "bob", "provider": "mock", "model": "anthropic/mock-bob"},
                {"name": "carol", "provider": "mock", "model": "google/mock-carol"},
            ],
            "judge": "mock:openai/mock-judge",
            "chairman": "mock:mock/mock-chair",
            "aggregator": "mock:mock/mock-agg",
        },
        "run": {"max_rounds": 4, "target_score": 85, "parallel": False},
        "promptsmith": {"enabled": True, "rounds": 1},
        "output": {"db_path": db_path, "dashboard_path": os.path.join(os.path.dirname(db_path), "d.html")},
    })


def run() -> int:
    c = _Check()

    # --- model ------------------------------------------------------------
    from .model import ModelSpec, Session, Turn, content_hash, model_vendor, session_id
    c.ok("content_hash stable", content_hash("a", "b") == content_hash("a", "b"))
    c.ok("session_id prefixed", session_id("t", "debate", 1.0).startswith("s-"))
    c.ok("vendor maps claude", model_vendor("anthropic/claude-3.5") == "anthropic")
    c.ok("vendor maps gpt", model_vendor("openai/gpt-4o") == "openai")
    c.ok("modelspec ref", ModelSpec("j", "mock", "x/y").ref() == "mock:x/y")

    # --- config -----------------------------------------------------------
    from .config import DEFAULT_CONFIG, _deep_merge, load_config, member_specs, parse_ref, role_spec
    merged = _deep_merge(DEFAULT_CONFIG, {"run": {"target_score": 5}})
    c.ok("deep_merge overrides leaf", merged["run"]["target_score"] == 5)
    c.ok("deep_merge keeps siblings", merged["run"]["max_rounds"] == 4)
    c.ok("load_config defaults", load_config(None)["run"]["strategy"] == "refine")
    c.ok("parse_ref keeps model colons", parse_ref("openrouter:meta/x:free") == ("openrouter", "meta/x:free"))
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _mock_cfg(os.path.join(tmp, "t.db"))
        c.ok("member_specs count", len(member_specs(cfg)) == 3)
        c.ok("role_spec judge", role_spec(cfg, "judge").model == "openai/mock-judge")

        # --- store --------------------------------------------------------
        from .store import Store
        with Store(cfg["output"]["db_path"]) as store:
            sess = Session(id="s-test", task="t", strategy="debate", final="answer", final_score=88.0)
            sess.rounds = []
            store.save_session(sess)
            c.ok("session round-trips", (store.get_session("s-test") or {}).get("final") == "answer")
            c.ok("session listed", len(store.list_sessions()) == 1)
            store.ai_cache_put("k1", "m", "p", "resp")
            c.ok("ai_cache hit", store.ai_cache_get("k1") == "resp")
            c.ok("ai_cache miss", store.ai_cache_get("nope") is None)

            # --- cost -----------------------------------------------------
            from . import cost
            c.ok("count_tokens heuristic", cost.count_tokens("abcdefgh") >= 1)
            c.ok("price known model", cost.price(cfg, "openai/gpt-4o", 1_000_000, 0) == 2.5)
            c.ok("price unknown -> 0", cost.price(cfg, "totally/unknown", 1_000_000, 1_000_000) == 0.0)
            c.ok("over_budget respects cap", cost.over_budget({"cost": {"budget_usd": 0.1}}, 0.2) is True)

            # --- provider (mock) ------------------------------------------
            from . import provider
            from .config import member_specs, role_spec
            prov = provider.for_config(cfg)
            specs = member_specs(cfg)
            comp = prov.complete(specs[0], [{"role": "user", "content": "hello"}], store=store)
            c.ok("mock completion ok", comp.ok and len(comp.text) > 0)
            c.ok("mock completion cached", prov.complete(specs[0],
                 [{"role": "user", "content": "hello"}], store=store).text == comp.text)
            c.ok("mock counts tokens", comp.tokens_in > 0 and comp.tokens_out > 0)

            judge_spec = role_spec(cfg, "judge")
            jmsg = [{"role": "system", "content": "QUORUM-JUDGE"},
                    {"role": "user", "content": "ROUND=2 CANDIDATE A: foo CANDIDATE B: bar"}]
            jverd = prov.complete(judge_spec, jmsg, cache=False)
            import json as _json
            payload = _json.loads(jverd.text)
            c.ok("mock judge returns json", payload["score"] == 85.0 and payload["best"] == "A")

            jobs = [(s, [{"role": "user", "content": f"q{i}"}]) for i, s in enumerate(specs)]
            many = prov.complete_many(jobs, cache=False)
            c.ok("complete_many order+count", len(many) == len(specs) and all(m.ok for m in many))

            # --- judge + stop logic ---------------------------------------
            from . import judge
            from .model import Verdict
            def mkv(s):
                return Verdict(round=1, score=s)
            cands = [("alice", "answer one"), ("bob", "answer two")]
            v1, jt1 = judge.evaluate(cfg, prov, 1, "task", "prompt", cands,
                                     candidate_models=["mock/alice", "anthropic/mock-bob"], store=store)
            c.ok("judge scores round1", v1.score == 70.0 and v1.best_label == "alice")
            c.ok("judge turn accounted", jt1.tokens_out > 0 and jt1.kind == "judge")
            c.ok("no stop below target", judge.should_stop(cfg, [v1], 1)[0] is False)
            v2, _ = judge.evaluate(cfg, prov, 2, "task", "prompt", cands,
                                   candidate_models=["mock/alice"], store=store)
            stop2, reason2 = judge.should_stop(cfg, [v1, v2], 2)
            c.ok("stop at target", stop2 is True and "target" in reason2)
            capstop, capreason = judge.should_stop(
                {"run": {"max_rounds": 3, "target_score": 200, "plateau_patience": 9, "plateau_delta": 2}},
                [v1, v1, v1], 3)
            c.ok("stop at cap", capstop is True and "max" in capreason)
            plat_stop, plat_reason = judge.should_stop(
                {"run": {"max_rounds": 9, "target_score": 200, "plateau_delta": 2, "plateau_patience": 2}},
                [mkv(50), mkv(51), mkv(51.5)], 3)
            c.ok("stop on plateau", plat_stop is True and "plateau" in plat_reason)
            c.ok("consensus detects agreement",
                 judge.consensus_reached(["the same answer here", "the same answer here"]) is True)

            # --- strategies (end to end on the mock provider) -------------
            from . import format as fmt
            from . import orchestrator, strategies
            c.ok("strategy registry complete",
                 {"debate", "council", "moa", "refine", "ensemble"}.issubset(set(strategies.available())))
            for strat in ("debate", "council", "moa", "refine", "ensemble"):
                sess = orchestrator.run_session(cfg, f"solve {strat}", store=store,
                                                strategy=strat, verbose=False)
                c.ok(f"{strat}: has rounds", len(sess.rounds) >= 1)
                c.ok(f"{strat}: final + score", bool(sess.final) and sess.final_score > 0)
                c.ok(f"{strat}: tokens counted", sess.tokens_in > 0 and sess.tokens_out > 0)
                c.ok(f"{strat}: persisted", store.get_session(sess.id) is not None)

            dbg = orchestrator.run_session(cfg, "make it visible", store=store, strategy="debate")
            c.ok("promptsmith round present", any(r.index == 0 for r in dbg.rounds))
            c.ok("debate stops at target", "target" in dbg.stop_reason)
            c.ok("render shows final", "FINAL ANSWER" in fmt.render_session(dbg))
            c.ok("no-promptsmith skips round 0",
                 all(r.index != 0 for r in orchestrator.run_session(
                     cfg, "skip ps", store=store, strategy="refine", promptsmith_on=False).rounds))

            # --- bench harness --------------------------------------------
            from . import bench
            tasks = [{"id": "a", "task": "task one"}, {"id": "b", "task": "task two"}]
            rows = []
            for t in tasks:
                for strat in ("debate", "moa", "ensemble"):
                    sess = orchestrator.run_session(cfg, t["task"], store=store, strategy=strat)
                    rows.append({"strategy": strat, "task_id": t["id"], "score": sess.final_score,
                                 "rounds": len(sess.rounds), "tokens": sess.tokens_in + sess.tokens_out,
                                 "cost_usd": sess.cost_usd, "seconds": 0.0})
            summ = bench.aggregate(rows, ["debate", "moa", "ensemble"], len(tasks))
            c.ok("bench ranks moa first", summ[0]["strategy"] == "moa")
            c.ok("bench win-rate 100 for moa", summ[0]["win_rate"] == 100.0)
            c.ok("bench covers all strategies", len(summ) == 3)

            # --- render + export ------------------------------------------
            from . import exporter, render
            dpath = render.build(cfg, store)
            with open(dpath, "r", encoding="utf-8") as fh:
                htmltext = fh.read()
            c.ok("dashboard written", os.path.exists(dpath))
            c.ok("dashboard self-contained", "const D =" in htmltext and "__DATA__" not in htmltext)
            c.ok("dashboard shows comparison", "strategy comparison" in htmltext)
            mdpath = os.path.join(os.path.dirname(dpath), "exp.md")
            c.ok("md export ok", exporter.run(cfg, store, fmt="md", session_id=dbg.id, out=mdpath) == 0
                 and os.path.exists(mdpath))
            csvpath = os.path.join(os.path.dirname(dpath), "exp.csv")
            exporter.run(cfg, store, fmt="csv", session_id=dbg.id, out=csvpath)
            c.ok("csv export ok", os.path.exists(csvpath))

            # --- reference grading (vs expected output) -------------------
            from . import grade
            c.ok("gold extract gsm8k", grade.extract_gold("work...\n#### 18") == "18")
            c.ok("numeric match true", grade.numeric_match("hence 18 total", "#### 18") is True)
            c.ok("numeric match false", grade.numeric_match("hence 17 total", "#### 18") is False)
            gs, gc, gt = grade.grade(cfg, prov, "q", "the answer is 18", "#### 18")
            c.ok("grade numeric deterministic", gs == 100.0 and gc is True and gt is None)
            ps, _pc, pt = grade.grade(cfg, prov, "q", "text", "a prose reference with several words")
            c.ok("grade prose via mock grader", ps == 90.0 and pt is not None)

            # --- embed API (backend for other tools) ----------------------
            from . import api
            host = {"ai": {"provider": "mock", "model": "mock/m1", "max_tokens": 200, "api_key_env": ""},
                    "quorum": {"enabled": True, "strategy": "refine", "max_rounds": 2}}
            c.ok("api.enabled on", api.enabled(host) is True)
            host_off = {**host, "quorum": {**host["quorum"], "enabled": False}}
            c.ok("api.enabled off", api.enabled(host_off) is False)
            qc = api.build_config(host)
            c.ok("api build_config", qc["run"]["strategy"] == "refine"
                 and qc["promptsmith"]["enabled"] is False)
            out = api.chat(host, store, "You are precise.", "Say hello.")
            c.ok("api.chat returns text", isinstance(out, str) and len(out) > 0)
            c.ok("api.chat disabled -> None", api.chat(host_off, store, "s", "u") is None)

    print(f"\n  {c.passed} passed, {c.failed} failed")
    return 0 if c.failed == 0 else 1
