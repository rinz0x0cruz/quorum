"""Orchestrator: run one deliberation end to end.

Phase 1 refines the prompt (optional); phase 2 runs the chosen strategy until the
judge says "good enough". The session is persisted and returned.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from . import provider as provider_mod, strategies
from .config import member_specs
from .model import Session, session_id


def run_session(cfg: dict, task: str, *, store: Any = None, strategy: Optional[str] = None,
                max_rounds: Optional[int] = None, target: Optional[float] = None,
                promptsmith_on: bool = True, promptsmith: Optional[bool] = None,
                solve_prompt: Optional[str] = None,
                history: Optional[list] = None, context: Optional[list] = None,
                verbose: bool = False, prov: Any = None,
                emit: Optional[Callable[[str], None]] = None) -> Session:
    # CLI passes promptsmith=<bool>; keep that name working too.
    if promptsmith is not None:
        promptsmith_on = promptsmith

    run = cfg.setdefault("run", {})
    if strategy:
        run["strategy"] = strategy
    if max_rounds is not None:
        run["max_rounds"] = max_rounds
    if target is not None:
        run["target_score"] = target
    strat_name = run.get("strategy", "refine")

    prov = prov or provider_mod.for_config(cfg, store=store)
    members = member_specs(cfg)
    log = emit or ((lambda s: print("  " + s)) if verbose else (lambda s: None))
    session = Session(id=session_id(task, strat_name), task=task, strategy=strat_name)

    # phase 1: prompt design/refinement.
    # A caller (e.g. the embed API) may supply the solve-prompt directly, which
    # skips promptsmith and uses their instructions verbatim.
    if solve_prompt is not None:
        prompt = solve_prompt
    else:
        prompt = task
        if promptsmith_on and (cfg.get("promptsmith", {}) or {}).get("enabled", False):
            from . import promptsmith as ps  # local alias (param shadows module name)
            prompt = ps.refine(cfg, prov, task, store=store, session=session, emit=log)
    session.prompt = prompt

    # Optional grounding: prepend caller-supplied conversation history / reference
    # docs as DATA (the whole council sees it; session.prompt stays clean so the
    # stored transcript is not bloated by the caller's context).
    delib_prompt = prompt
    if history or context:
        from . import contextwindow
        pre = contextwindow.preamble(cfg, history=history, context=context)
        if pre:
            delib_prompt = pre + "\n\n" + prompt

    # phase 2: strategy deliberation (with optional pre/post extension hooks)
    from . import hooks
    ctx = strategies.Context(cfg=cfg, prov=prov, store=store, task=task, prompt=delib_prompt,
                             members=members, session=session, emit=log,
                             opts=strategies.RunOptions.from_cfg(cfg))
    hooks.run_pre(ctx)
    strat = strategies.get(strat_name)
    strat(ctx)
    hooks.run_post(ctx)

    # Persist only when the store is a quorum store (embed callers may pass their
    # own tool's store, which only implements the AI cache).
    if store is not None and hasattr(store, "save_session"):
        store.save_session(session)
        store.add_run("run", len(session.rounds), session.status)
    return session
