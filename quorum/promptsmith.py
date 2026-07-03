"""Phase 1: design + iteratively refine the *prompt* used to solve the task.

An OPRO-style loop (Yang et al., "Large Language Models as Optimizers"): a
prompt-engineer model drafts a solving instruction, then critiques and improves
it across a few rounds. The result feeds phase 2 (the deliberation strategies) as
the shared solve-prompt. Runs offline via the ``mock`` provider.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from . import provider as provider_mod
from .config import role_spec
from .model import Round, Session

_SYSTEM = (
    "QUORUM-PROMPTSMITH. You are an expert prompt engineer. Given a task, write the best possible "
    "INSTRUCTION for another model to solve it well. Do not solve the task and do not restate it; "
    "output only a reusable, self-contained instruction."
)


def _exemplars(cfg: dict, store: Any) -> str:
    """Few-shot block of instructions that scored well on past tasks (DSPy-style
    bootstrapping). Empty unless ``promptsmith.bootstrap`` is on and the store has
    qualifying sessions -- so default behaviour and offline replay are unchanged."""
    ps = cfg.get("promptsmith", {}) or {}
    if not ps.get("bootstrap") or store is None or not hasattr(store, "top_sessions"):
        return ""
    try:
        rows = store.top_sessions(limit=int(ps.get("bootstrap_k", 3)),
                                  min_score=float(ps.get("bootstrap_min", 80.0)))
    except Exception:  # noqa: BLE001 - bootstrapping is best-effort, never fatal
        return ""
    ex = [f"- (scored {r.get('final_score', 0):.0f}) {(r.get('prompt') or '').strip()}"
          for r in rows if (r.get("prompt") or "").strip()]
    if not ex:
        return ""
    return ("\n\nHere are instructions that solved earlier tasks well; learn from their style, "
            "do not copy them:\n" + "\n".join(ex))


def refine(cfg: dict, prov: "provider_mod.Provider", task: str, *,
           store: Any = None, session: Optional[Session] = None,
           emit: Optional[Callable[[str], None]] = None, verbose: bool = False) -> str:
    rounds = int((cfg.get("promptsmith", {}) or {}).get("rounds", 2))
    smith = role_spec(cfg, "chairman")  # reuse the strongest configured model
    log = emit or ((lambda s: print("  " + s)) if verbose else (lambda s: None))
    examples = _exemplars(cfg, store)

    instruction = ""
    turns = []
    for i in range(1, max(1, rounds) + 1):
        if i == 1:
            user = f"TASK:\n{task}{examples}\n\nWrite the initial solving instruction."
        else:
            user = (f"TASK:\n{task}\n\nCURRENT INSTRUCTION:\n{instruction}\n\n"
                    "Critique it briefly, then output only an improved instruction.")
        comp = prov.complete(smith, [{"role": "system", "content": _SYSTEM},
                                     {"role": "user", "content": user}], temperature=0.4, store=store)
        if comp.ok and comp.text:
            instruction = comp.text
            turns.append(provider_mod.to_turn(comp, 0, "promptsmith", "promptsmith"))
        log(f"promptsmith v{i}: {len(instruction)} chars")

    if session is not None and turns:
        rnd = Round(index=0, turns=turns, best_content=instruction)
        session.rounds.append(rnd)
        for t in turns:
            session.account(t)
    return instruction or task
