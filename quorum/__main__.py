"""quorum command-line interface.

Usage:
    python -m quorum init                         Scaffold config + data dir
    python -m quorum run "<task>" [options]       Deliberate to a "good enough" solution
    python -m quorum promptsmith "<task>"         Just design/refine a prompt (phase 1)
    python -m quorum bench --tasks f --strategies debate,council,moa   Compare strategies
    python -m quorum list                         Recent deliberations
    python -m quorum show <session-id>            Print a full transcript
    python -m quorum dashboard [--open]           Render the offline HTML dashboard
    python -m quorum serve [--open]               Serve the dashboard locally
    python -m quorum export [--format json|csv|md] [--session id]
    python -m quorum models [--ping]              List council members (and check reachability)
    python -m quorum throttle                     Analyze API rate-limit/throttle telemetry
    python -m quorum selftest                     Offline self-tests (no network)

Live deliberation talks to the OpenAI-compatible endpoints you configure; the
engine + selftest run fully offline via the built-in ``mock`` provider.
"""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import load_config
from .store import Store


def _force_utf8() -> None:
    """Make Unicode output (glyphs, bars) safe on legacy Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def _store(cfg) -> Store:
    return Store(cfg["output"]["db_path"])


# --------------------------------------------------------------------------
# commands (feature modules imported lazily so the CLI stays light + offline)
# --------------------------------------------------------------------------
def cmd_init(args, cfg):
    from . import scaffold
    return scaffold.run(args)


def cmd_run(args, cfg):
    from . import orchestrator
    with _store(cfg) as store:
        session = orchestrator.run_session(
            cfg, args.task, store=store,
            strategy=args.strategy, max_rounds=args.rounds,
            target=args.target, promptsmith=not args.no_promptsmith,
            verbose=not args.json,
        )
    if args.json:
        import json
        print(json.dumps(session.to_dict(), indent=2))
    else:
        from . import format as fmt
        print(fmt.render_session(session))
    return 0


def cmd_promptsmith(args, cfg):
    from . import promptsmith, provider
    prov = provider.for_config(cfg)
    with _store(cfg) as store:
        refined = promptsmith.refine(cfg, prov, args.task, store=store, verbose=True)
    print("\n--- refined prompt ---\n" + refined)
    return 0


def cmd_bench(args, cfg):
    from . import bench
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    with _store(cfg) as store:
        return bench.run(cfg, args.tasks, strategies, store,
                         as_json=args.json, verbose=not args.json)


def cmd_list(args, cfg):
    with _store(cfg) as store:
        rows = store.list_sessions(limit=args.limit)
    if not rows:
        print("  no deliberations yet -- run `quorum run \"<task>\"`.")
        return 0
    for r in rows:
        print(f"  {r['id']}  {r['strategy']:<8}  score={r['final_score']:>5.1f}  "
              f"rounds={r['rounds']}  {r['created']}  {r['task'][:48]}")
    return 0


def cmd_show(args, cfg):
    from . import format as fmt
    with _store(cfg) as store:
        d = store.get_session(args.id)
    if not d:
        print(f"  no session {args.id}")
        return 1
    print(fmt.render_session_dict(d))
    return 0


def cmd_dashboard(args, cfg):
    from . import render
    with _store(cfg) as store:
        path = render.build(cfg, store)
    print(f"  dashboard -> {path}")
    if getattr(args, "open", False):
        import os
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(path)}")
    return 0


def cmd_serve(args, cfg):
    if getattr(args, "api", False):
        from . import serveapi
        return serveapi.run(cfg, port=args.port, token=args.token,
                            host=args.host, request_timeout=args.timeout)
    from . import serve
    return serve.run(cfg, port=args.port, open_browser=args.open)


def cmd_chat(args, cfg):
    """One-shot deliberation for scripts/CI/other languages (stdout = the answer)."""
    import sys
    from . import orchestrator
    user = args.user if args.user is not None else sys.stdin.read()
    session = orchestrator.run_session(cfg, user, solve_prompt=args.system or "",
                                       promptsmith_on=False, strategy=args.strategy, verbose=False)
    if args.json:
        import json
        print(json.dumps({"content": session.final, "strategy": session.strategy,
                          "status": session.status, "stop_reason": session.stop_reason,
                          "tokens": session.tokens_in + session.tokens_out,
                          "cost_usd": round(session.cost_usd, 6)}))
    else:
        print(session.final)
    return 0 if (session.status == "ok" and session.final) else 1


def cmd_export(args, cfg):
    from . import exporter
    with _store(cfg) as store:
        return exporter.run(cfg, store, fmt=args.format, session_id=args.session, out=args.out)


def cmd_models(args, cfg):
    from . import provider
    return provider.list_models(cfg, ping=args.ping)


def cmd_throttle(args, cfg):
    from . import throttle
    with _store(cfg) as store:
        return throttle.run(cfg, store, provider=args.provider, probe=not args.no_probe)


def cmd_selftest(args, cfg):
    from . import selftest
    return selftest.run()


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quorum",
                                description="Multi-model deliberation: refine a prompt, then "
                                            "debate a solution until it is good enough.")
    p.add_argument("--version", action="version", version=f"quorum {__version__}")
    p.add_argument("--config", default=None, help="Path to config.yaml/json")
    p.add_argument("--db", default=None, help="Override database path")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Scaffold config + data dir").set_defaults(func=cmd_init)

    sp = sub.add_parser("run", help="Deliberate to a good-enough solution")
    sp.add_argument("task", help="The question or task to solve")
    sp.add_argument("--strategy", default=None,
                    choices=["debate", "council", "moa", "refine", "ensemble"],
                    help="Override the configured strategy")
    sp.add_argument("--rounds", type=int, default=None, help="Override max rounds")
    sp.add_argument("--target", type=float, default=None, help="Override target score (0-100)")
    sp.add_argument("--no-promptsmith", action="store_true", help="Skip prompt refinement (phase 1)")
    sp.add_argument("--json", action="store_true", help="Emit the full session as JSON")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("promptsmith", help="Design/refine a prompt (phase 1 only)")
    sp.add_argument("task", help="The task the prompt should solve")
    sp.set_defaults(func=cmd_promptsmith)

    sp = sub.add_parser("bench", help="Compare strategies over a task set")
    sp.add_argument("--tasks", required=True, help="YAML/JSON file of tasks")
    sp.add_argument("--strategies", default="debate,council,moa,refine,ensemble",
                    help="Comma-separated strategy list")
    sp.add_argument("--json", action="store_true", help="Emit the comparison as JSON")
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("list", help="Recent deliberations")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("show", help="Print a full transcript")
    sp.add_argument("id", help="Session id")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("dashboard", help="Render the offline HTML dashboard")
    sp.add_argument("--open", action="store_true", help="Open in browser")
    sp.set_defaults(func=cmd_dashboard)

    sp = sub.add_parser("serve", help="Serve the dashboard locally")
    sp.add_argument("--port", type=int, default=8802)
    sp.add_argument("--open", action="store_true")
    sp.add_argument("--api", action="store_true",
                    help="Serve an OpenAI-compatible /v1/chat/completions endpoint (deliberates per request)")
    sp.add_argument("--token", default="", help="Optional bearer token required by --api")
    sp.add_argument("--host", default="127.0.0.1", help="Bind host for --api (use 0.0.0.0 in Docker)")
    sp.add_argument("--timeout", type=float, default=120.0,
                    help="Per-request deliberation timeout in seconds for --api")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("chat",
                        help="One-shot deliberation of a system+user prompt (for scripts/CI/other languages)")
    sp.add_argument("--user", default=None, help="User message (default: read from stdin)")
    sp.add_argument("--system", default="", help="System / instruction prompt")
    sp.add_argument("--strategy", default=None,
                    choices=["debate", "council", "moa", "refine", "ensemble"],
                    help="Override the configured strategy")
    sp.add_argument("--json", action="store_true",
                    help="Emit JSON {content, strategy, status, tokens, cost_usd}")
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("export", help="Export a session as JSON, CSV, or Markdown")
    sp.add_argument("--format", choices=["json", "csv", "md"], default="json")
    sp.add_argument("--session", default=None, help="Session id (default: latest)")
    sp.add_argument("--out", default=None, help="Output path")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("models", help="List council members")
    sp.add_argument("--ping", action="store_true", help="Check each endpoint is reachable")
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("throttle", help="Analyze recorded API rate-limit/throttle telemetry")
    sp.add_argument("--provider", default="openrouter",
                    help="Provider whose key quota to probe (default: openrouter)")
    sp.add_argument("--no-probe", action="store_true",
                    help="Skip the live key/quota probe (offline: telemetry only)")
    sp.set_defaults(func=cmd_throttle)

    sub.add_parser("selftest", help="Offline self-tests").set_defaults(func=cmd_selftest)
    return p


def main(argv=None) -> int:
    _force_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.db:
        cfg["output"]["db_path"] = args.db
    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
