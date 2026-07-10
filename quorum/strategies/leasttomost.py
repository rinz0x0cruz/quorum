"""Least-to-Most prompting (Zhou et al. 2022, arXiv:2205.10625).

Decompose a hard problem into an ordered list of simpler sub-questions, then
solve them in sequence -- each sub-answer feeding the next -- so the final
sub-question (essentially the original task) is answered with all the
intermediate results in hand. Excels at compositional problems a single shot
fumbles. The number of solve calls is capped so it stays within free-tier
request budgets, and it honors the cost budget.
"""
from __future__ import annotations

import re

from .. import cost, judge, prompts, provider
from ..model import Round
from . import Context

_MAX_SUBPROBLEMS = 6
_STEP_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.*\S)\s*$")


def _parse_steps(text: str) -> list[str]:
    """Pull the numbered/bulleted sub-questions out of the decomposition text."""
    steps = []
    for line in (text or "").splitlines():
        m = _STEP_RE.match(line)
        if m:
            steps.append(m.group(1).strip())
    return steps


def run(ctx: Context):
    cfg, prov = ctx.cfg, ctx.prov
    if not ctx.members:
        ctx.session.status = "error"
        ctx.session.stop_reason = "no members configured"
        return ctx.session
    m = ctx.members[0]
    rnd = Round(index=1)

    def _step(msgs, kind):
        comp = prov.complete(m, msgs, store=ctx.store)
        if comp.ok:
            t = provider.to_turn(comp, 1, m.name, kind)
            rnd.turns.append(t)
            ctx.session.account(t)
        return comp

    # Stage 1: decompose into ordered sub-questions.
    dec = _step(prompts.decompose(ctx.prompt, ctx.task), "decompose")
    subs = _parse_steps(dec.text) if dec.ok else []
    if not subs:
        subs = [ctx.task]                      # fall back to solving the task directly
    subs = subs[:_MAX_SUBPROBLEMS]

    # Stage 2: solve each sub-question in order, feeding prior answers forward.
    solved: list[tuple[str, str]] = []
    answer = ""
    for sub in subs:
        sol = _step(prompts.solve_subproblem(ctx.prompt, ctx.task, sub, solved), "solve")
        if not sol.ok:
            break
        answer = sol.text
        solved.append((sub, answer))
        if cost.over_budget(cfg, ctx.session.cost_usd):
            ctx.session.stop_reason = "least-to-most: cost budget exceeded"
            break

    if not answer:
        ctx.session.status = "error"
        ctx.session.stop_reason = "model failed during decomposition/solve"
        ctx.session.rounds.append(rnd)
        return ctx.session

    verdict, jturn = judge.evaluate(cfg, prov, 1, ctx.task, ctx.prompt,
                                    [("least-to-most", answer)], candidate_models=[m.model],
                                    store=ctx.store)
    rnd.turns.append(jturn)
    ctx.session.account(jturn)
    rnd.verdict = verdict
    rnd.best_content = answer
    ctx.session.rounds.append(rnd)
    ctx.session.final = answer
    ctx.session.final_score = verdict.score
    if not ctx.session.stop_reason:
        ctx.session.stop_reason = f"least-to-most (decomposed into {len(solved)} sub-questions)"
    ctx.event("result", f"least-to-most: {len(solved)} steps, score {verdict.score:.0f}",
              score=verdict.score, steps=len(solved))
    return ctx.session
