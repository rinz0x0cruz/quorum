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
    from .model import ModelSpec, Session, content_hash, model_vendor, session_id
    c.ok("content_hash stable", content_hash("a", "b") == content_hash("a", "b"))
    c.ok("session_id prefixed", session_id("t", "debate", 1.0).startswith("s-"))
    c.ok("vendor maps claude", model_vendor("anthropic/claude-3.5") == "anthropic")
    c.ok("vendor maps gpt", model_vendor("openai/gpt-4o") == "openai")
    c.ok("modelspec ref", ModelSpec("j", "mock", "x/y").ref() == "mock:x/y")

    # --- scoring (leaf lexical primitives + registry) ---------------------
    from . import scoring
    c.ok("scoring tokenizes", scoring.tokens("Hello, WORLD 42!") == {"hello", "world", "42"})
    # overlap_coeff (rank/contextwindow) is asymmetric; jaccard (consensus) is
    # symmetric -- same inputs, different results proves they are NOT the same formula.
    c.ok("overlap_coeff |q&d|/|q|", scoring.overlap_coeff({"a", "b"}, {"a", "b", "c", "d"}) == 1.0)
    c.ok("jaccard |a&b|/|a|b|", scoring.jaccard({"a", "b"}, {"a", "b", "c", "d"}) == 0.5)
    c.ok("overlap empty-query -> 0", scoring.overlap_coeff(set(), {"a"}) == 0.0)
    c.ok("registry resolves lexical", "lexical" in scoring.available()
         and isinstance(scoring.get("lexical"), scoring.Scorer))
    c.ok("LexicalScorer scores via overlap",
         scoring.get("lexical").score("a b", "a b c d") == 1.0)

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

            # --- fallback + rank (optimizations) --------------------------
            fb_cfg = _deep_merge(cfg, {"run": {"fallbacks": ["mock:google/mock-carol"]}})
            c.ok("member fallbacks resolved",
                 bool(member_specs(fb_cfg)[0].fallbacks)
                 and member_specs(fb_cfg)[0].fallbacks[0].provider == "mock")
            dead = _deep_merge(cfg, {"providers": {"dead": {"base_url": "", "api_key_env": ""}}})
            dprov = provider.for_config(dead)
            primary = ModelSpec(name="x", provider="dead", model="dead/m",
                                fallbacks=[ModelSpec(name="fb", provider="mock", model="mock/fb")])
            c.ok("fallback used on failure", dprov.complete(
                primary, [{"role": "user", "content": "hi"}], cache=False).provider == "mock")
            c.ok("no fallback -> error", dprov.complete(
                ModelSpec(name="x", provider="dead", model="dead/m"),
                [{"role": "user", "content": "hi"}], cache=False).ok is False)
            from . import rank
            c.ok("rank consensus best-first", rank.consensus_order(3, ["Ranking: B, A, C"])[0] == 1)
            c.ok("rank top_k limits", rank.top_k_indices(3, ["A, B, C"], 2) == [0, 1])

            # --- judge + stop logic ---------------------------------------
            from . import judge
            from .model import Verdict
            def mkv(s):
                return Verdict(round=1, score=s)
            cands = [("alice", "answer one"), ("bob", "answer two")]
            v1, jt1 = judge.evaluate(cfg, prov, 1, "task", "prompt", cands,
                                     candidate_models=["mock/alice", "anthropic/mock-bob"], store=store)
            c.ok("judge scores round1", v1.score == 70.0 and (v1.best_label, v1.best_content) in cands)
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

            # --- challenger builder + json_mode plumb ---------------------
            from . import prompts as _prompts
            _ch = _prompts.challenge("p", "t", "prev", [("a", "x")], "c", True)
            c.ok("challenge builder", len(_ch) == 2 and "devil" in _ch[0]["content"].lower())
            _pp = _prompts.propose("", "t")
            c.ok("prompts package re-exports",
                 len(_pp) == 2 and _pp[0]["content"] == _prompts.PROPOSER_SYSTEM
                 and "QUORUM-AGGREGATOR" in _prompts.AGGREGATOR_SYSTEM)
            _seen: dict = {}
            _orig_complete = prov.complete
            def _spy(spec, messages, **kw):
                _seen["rf"] = kw.get("response_format")
                return _orig_complete(spec, messages, **kw)
            prov.complete = _spy  # type: ignore[method-assign]
            judge.evaluate(_deep_merge(cfg, {"judge": {"json_mode": True}}), prov, 1, "t", "p",
                           [("a", "ans")], candidate_models=["m"], store=store)
            prov.complete = _orig_complete  # type: ignore[method-assign]
            c.ok("json_mode passes response_format", _seen.get("rf") == {"type": "json_object"})

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

            # --- cascade (difficulty-adaptive escalation; FrugalGPT) ------
            c.ok("cascade registered", "cascade" in strategies.available())
            _casc = orchestrator.run_session(
                _deep_merge(cfg, {"run": {"cascade": ["refine", "debate"]}}),
                "cascade task", store=store, strategy="cascade", promptsmith_on=False)
            c.ok("cascade stops at cheap stage",
                 _casc.stop_reason.startswith("cascade: refine reached target") and bool(_casc.final))

            # --- adaptive-consistency (fewer samples via early stop) ------
            from . import consistency as _cons
            c.ok("consistency clusters numeric",
                 _cons.leader(_cons.cluster(["ans 5", "so 5", "no 7"]))["count"] == 2)
            c.ok("consistency confident on majority",
                 _cons.confident(_cons.cluster(["5", "5", "5"])) is True)
            _ada = orchestrator.run_session(
                _deep_merge(cfg, {"run": {"adaptive_samples": True, "samples": 10, "samples_min": 2}}),
                "adaptive q", store=store, strategy="ensemble", promptsmith_on=False)
            _adaprop = [t for r in _ada.rounds for t in r.turns if t.kind == "propose"]
            c.ok("ensemble adaptive early-stops",
                 len(_adaprop) == 2 and "adaptive vote" in _ada.stop_reason)

            # --- judge cadence (run.judge_every defers judge calls) -------
            c.ok("judge.due skips mid, keeps ends",
                 judge.due(1, 2, 4) and judge.due(4, 2, 4) and not judge.due(3, 2, 4))
            _jc = orchestrator.run_session(
                _deep_merge(cfg, {"run": {"judge_every": 2, "max_rounds": 4, "target_score": 200}}),
                "cadence q", store=store, strategy="refine", promptsmith_on=False)
            _jturns = [t for r in _jc.rounds for t in r.turns if t.kind == "judge"]
            c.ok("refine judges 3 of 4 rounds", len(_jturns) == 3)

            # --- self-consistency (USC selection + majority vote) ---------
            c.ok("usc prompt has sentinel", "QUORUM-USC" in _prompts.usc("t", [("a", "x")])[0]["content"])
            _sc = orchestrator.run_session(
                _deep_merge(cfg, {"run": {"samples": 3}}),
                "sc task", store=store, strategy="selfconsistency", promptsmith_on=False)
            c.ok("selfconsistency reaches consensus",
                 bool(_sc.final) and "self-consistency" in _sc.stop_reason and _sc.final_score > 0)

            # --- reflexion + chain-of-verification ------------------------
            _rx = orchestrator.run_session(cfg, "reflexion q", store=store, strategy="reflexion",
                                           promptsmith_on=False)
            c.ok("reflexion reflects + solves",
                 bool(_rx.final) and any(t.kind == "reflect" for r in _rx.rounds for t in r.turns))
            _vf = orchestrator.run_session(cfg, "verify q", store=store, strategy="verify",
                                           promptsmith_on=False)
            c.ok("verify runs full pipeline",
                 bool(_vf.final) and {"draft", "verify", "revise"}
                 <= {t.kind for r in _vf.rounds for t in r.turns})

            # --- self-MoA (single best model sampled + aggregated) -------
            _sm = orchestrator.run_session(cfg, "selfmoa q", store=store, strategy="selfmoa",
                                           promptsmith_on=False)
            c.ok("selfmoa samples + aggregates",
                 bool(_sm.final) and "self-moa" in _sm.stop_reason
                 and [t.kind for r in _sm.rounds for t in r.turns].count("propose") >= 2)

            # --- structured events (#8): typed on_event stream ------------
            from . import events as _events
            _evs = []
            orchestrator.run_session(cfg, "events q", store=store, strategy="refine",
                                     promptsmith_on=False, on_event=_evs.append)
            c.ok("events stream phase/round/done",
                 {"phase", "round", "done"} <= {e.kind for e in _evs})
            c.ok("event coerce wraps string", _events.coerce("hi").kind == "log")

            # --- config validation (#7): warn on unknown keys -------------
            from .config import validate_config as _vc
            c.ok("config validate clean", _vc({"run": {"max_rounds": 3}}) == [])
            c.ok("config validate flags typo", "run.max_round" in _vc({"run": {"max_round": 3}}))
            c.ok("config validate self-clean", _vc(_deep_merge(DEFAULT_CONFIG, {})) == [])

            # --- top-K fuse + devil's advocate ----------------------------
            tk = orchestrator.run_session(
                _deep_merge(cfg, {"run": {"top_k": 2, "max_rounds": 1}}),
                "rank then fuse", store=store, strategy="council", promptsmith_on=False)
            c.ok("top_k council runs", bool(tk.final) and tk.final_score > 0)
            da = orchestrator.run_session(
                _deep_merge(cfg, {"run": {"devils_advocate": True, "max_rounds": 2, "target_score": 200}}),
                "argue", store=store, strategy="debate", promptsmith_on=False)
            c.ok("devil's advocate challenge turn",
                 any(t.kind == "challenge" for r in da.rounds for t in r.turns))

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

            # --- promptsmith bootstrap (few-shot from store) --------------
            store.save_session(Session(id="top1", task="t", strategy="debate",
                                       prompt="Solve carefully.", final="a", final_score=95.0))
            c.ok("top_sessions filters", len(store.top_sessions(limit=5, min_score=90.0)) >= 1)
            from . import promptsmith
            c.ok("promptsmith bootstrap ok", bool(promptsmith.refine(
                _deep_merge(cfg, {"promptsmith": {"bootstrap": True, "rounds": 1}}),
                prov, "new q", store=store)))

            # --- adapters (external -> quorum, shared by api + serveapi) --
            from . import adapters
            _ahost = {"ai": {"provider": "mock", "model": "mock/m1", "max_tokens": 200, "api_key_env": ""},
                      "quorum": {"enabled": True, "strategy": "refine", "max_rounds": 2}}
            _aqc = adapters.host_config(_ahost)
            c.ok("adapters.host_config maps host",
                 _aqc["run"]["strategy"] == "refine" and _aqc["council"]["judge"] == "mock:mock/m1")
            _asys, _ahist, _alast = adapters.split_messages([
                {"role": "system", "content": "s"}, {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}])
            c.ok("adapters.split_messages triple",
                 _asys == "s" and _alast == "c" and _ahist == [{"role": "user", "content": "a"},
                                                               {"role": "assistant", "content": "b"}])
            c.ok("adapters.select_strategy named", adapters.select_strategy("debate", cfg) == "debate")
            c.ok("adapters.select_strategy default",
                 adapters.select_strategy("nope", cfg) == (cfg.get("run", {}) or {}).get("strategy", "refine"))

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
            c.ok("api.chat per-call strategy",
                 isinstance(api.chat(host, store, "s", "u", strategy="ensemble"), str))
            _sv = api.score(host, store, "Rate this answer.", "a candidate to judge",
                            rubric={"relevance": 0.7, "authority": 0.3})
            c.ok("api.score returns verdict",
                 isinstance(_sv, dict) and _sv["score"] == 55.0 and bool(_sv["sub_scores"]))
            c.ok("api.score disabled -> None", api.score(host_off, store, "t", "c") is None)

            # --- OpenAI-compatible API (serveapi) -------------------------
            from . import serveapi
            code, obj = serveapi.complete_chat(cfg, {"model": "refine", "messages": [
                {"role": "system", "content": "be precise"},
                {"role": "user", "content": "hello"}]})
            c.ok("serveapi 200 + content", code == 200 and bool(obj["choices"][0]["message"]["content"]))
            c.ok("serveapi model tag", obj["model"] == "quorum/refine")
            bad, _ = serveapi.complete_chat(cfg, {"messages": [{"role": "system", "content": "s"}]})
            c.ok("serveapi no-user 400", bad == 400)

            # --- context windows (history + grounding docs) ---------------
            from . import contextwindow as _cw
            _ranked = _cw.select("simplehelp bypass", [
                {"id": "a", "title": "SimpleHelp bypass", "text": "simplehelp authentication bypass"},
                {"id": "b", "title": "x", "text": "unrelated gardening"}], k=2)
            c.ok("context select ranks relevant first", _ranked[0].id == "a")
            _pre = _cw.preamble({"context": {"budget_tokens": 4000, "history_turns": 8}},
                                history=[{"role": "user", "content": "prior turn"}],
                                context=[{"title": "S", "text": "story text"}])
            c.ok("context preamble frames as DATA",
                 "DATA ONLY" in _pre and "story text" in _pre and "prior turn" in _pre)
            c.ok("context empty preamble", _cw.preamble({}) == "")
            _cs = serveapi.complete_chat(cfg, {"model": "refine", "messages": [
                {"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ans"},
                {"role": "user", "content": "now"}], "context": [{"title": "d", "text": "grounding"}]})
            c.ok("serveapi accepts history+context",
                 _cs[0] == 200 and bool(_cs[1]["choices"][0]["message"]["content"]))

            # --- run-options + extension hooks ----------------------------
            from .strategies import RunOptions as _RO
            _ro = _RO.from_cfg({"run": {"max_rounds": 7, "top_k": 2, "devils_advocate": True}})
            c.ok("run-options resolved", _ro.max_rounds == 7 and _ro.top_k == 2 and _ro.devils_advocate)
            from . import hooks as _hooks
            _hooks.clear()
            _hf = {"pre": 0, "post": 0}
            _hooks.register_pre(lambda ctx: _hf.__setitem__("pre", _hf["pre"] + 1))
            _hooks.register_post(lambda ctx: _hf.__setitem__("post", _hf["post"] + 1))
            orchestrator.run_session(cfg, "hooked", store=store, strategy="refine", promptsmith_on=False)
            _hooks.clear()
            c.ok("pre/post hooks fire", _hf == {"pre": 1, "post": 1})

            # --- throttle telemetry + analyzer ----------------------------
            from . import throttle
            store.add_api_call("openrouter", "m:free", "ok", http_code=200,
                               latency_ms=50, rl_remaining=8)
            store.add_api_call("openrouter", "m:free", "HTTP 429", http_code=429,
                               retry_after=2.0, rl_remaining=0)
            c.ok("api_calls recorded", len(store.api_calls_recent()) >= 2)
            _tsum = throttle.summarize([
                {"ts": "2026-07-10T10:00:01Z", "provider": "openrouter", "model": "m:free",
                 "status": "ok", "http_code": 200, "latency_ms": 50, "rl_remaining": 8},
                {"ts": "2026-07-10T10:00:01Z", "provider": "openrouter", "model": "m:free",
                 "status": "HTTP 429", "http_code": 429, "latency_ms": 0, "rl_remaining": 0}])
            c.ok("throttle summarize counts 429",
                 _tsum["throttled"] == 1 and _tsum["by_model"]["m:free"]["total"] == 2)
            c.ok("throttle flags free ceiling",
                 any("rate_limit_rpm" in r for r in throttle.recommendations(
                     {"total": 1, "throttled": 1, "peak_rpm": {"openrouter": 20}, "by_model": {}},
                     cfg, None)))
            with open(render.build(cfg, store), encoding="utf-8") as _tfh:
                _thtml = _tfh.read()
            c.ok("dashboard renders throttle panel", '"by_model"' in _thtml and "m:free" in _thtml)

            # --- rate limiter (paces HTTP bursts under a per-minute cap) ---
            from .provider import RateLimiter as _RL
            c.ok("rate limiter disabled -> no wait", _RL(0).acquire() == 0.0)
            _rl = _RL(1200)  # 0.05s interval
            _w1, _w2 = _rl.acquire(), _rl.acquire()
            c.ok("rate limiter paces 2nd call", _w1 == 0.0 and _w2 > 0.0)

    print(f"\n  {c.passed} passed, {c.failed} failed")
    return 0 if c.failed == 0 else 1
