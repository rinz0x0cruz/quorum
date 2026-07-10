"""Self-Discover (Zhou et al. 2024, arXiv:2402.03620).

The model first *composes a task-specific reasoning structure* -- selecting and
adapting a handful of atomic reasoning modules into an ordered plan -- then
*follows that structure* to solve. Two cheap single-model calls (discover +
solve) plus one judge: a structured-reasoning boost that fits free-tier request
budgets, unlike search-heavy methods (e.g. tree-of-thoughts) that fan out many
calls per task.
"""
from __future__ import annotations

from .. import judge, prompts, provider
from ..model import Round
from . import Context


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

    # Stage 1: discover -- compose a task-specific reasoning structure.
    plan = _step(prompts.discover(ctx.prompt, ctx.task), "plan")
    structure = plan.text if plan.ok else ""

    # Stage 2: solve -- follow the structure to a final answer.
    solve = _step(prompts.discover_solve(ctx.prompt, ctx.task, structure), "solve")
    if not solve.ok:
        ctx.session.status = "error"
        ctx.session.stop_reason = f"model failed: {solve.error[:60]}"
        ctx.session.rounds.append(rnd)
        return ctx.session
    answer = solve.text

    verdict, jturn = judge.evaluate(cfg, prov, 1, ctx.task, ctx.prompt,
                                    [("self-discover", answer)], candidate_models=[m.model],
                                    store=ctx.store)
    rnd.turns.append(jturn)
    ctx.session.account(jturn)
    rnd.verdict = verdict
    rnd.best_content = answer
    ctx.session.rounds.append(rnd)
    ctx.session.final = answer
    ctx.session.final_score = verdict.score
    ctx.session.stop_reason = "self-discover (compose reasoning structure -> solve)"
    ctx.event("result", f"self-discover: score {verdict.score:.0f}", score=verdict.score)
    return ctx.session
