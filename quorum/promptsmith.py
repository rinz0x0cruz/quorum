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


def refine(cfg: dict, prov: "provider_mod.Provider", task: str, *,
           store: Any = None, session: Optional[Session] = None,
           emit: Optional[Callable[[str], None]] = None, verbose: bool = False) -> str:
    rounds = int((cfg.get("promptsmith", {}) or {}).get("rounds", 2))
    smith = role_spec(cfg, "chairman")  # reuse the strongest configured model
    log = emit or ((lambda s: print("  " + s)) if verbose else (lambda s: None))

    instruction = ""
    turns = []
    for i in range(1, max(1, rounds) + 1):
        if i == 1:
            user = f"TASK:\n{task}\n\nWrite the initial solving instruction."
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
