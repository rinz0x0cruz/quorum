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
    strat_name = run.get("strategy", "debate")

    prov = prov or provider_mod.for_config(cfg)
    members = member_specs(cfg)
    log = emit or ((lambda s: print("  " + s)) if verbose else (lambda s: None))
    session = Session(id=session_id(task, strat_name), task=task, strategy=strat_name)

    # phase 1: prompt design/refinement
    prompt = task
    if promptsmith_on and (cfg.get("promptsmith", {}) or {}).get("enabled", False):
        from . import promptsmith as ps  # local alias (param shadows module name)
        prompt = ps.refine(cfg, prov, task, store=store, session=session, emit=log)
    session.prompt = prompt

    # phase 2: strategy deliberation
    ctx = strategies.Context(cfg=cfg, prov=prov, store=store, task=task, prompt=prompt,
                             members=members, session=session, emit=log)
    strat = strategies.get(strat_name)
    strat(ctx)

    if store is not None:
        store.save_session(session)
        store.add_run("run", len(session.rounds), session.status)
    return session
