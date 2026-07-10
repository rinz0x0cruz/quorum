"""Step-Back prompting (Zheng et al. 2023, arXiv:2310.06117).

The model first *steps back* to a higher-level question and the general
principle behind the task (the abstract step), then *reasons from that
principle* to the concrete answer (the solve step). Two cheap single-model calls
plus one judge -- grounding the answer in a first-principle curbs the reasoning
slips of smaller free-tier models, at a request budget that fits their limits.
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

    # Stage 1: abstract -- step back to the general principle behind the task.
    abstract = _step(prompts.step_back(ctx.prompt, ctx.task), "abstract")
    principle = abstract.text if abstract.ok else ""

    # Stage 2: solve -- reason from that principle to the concrete answer.
    solve = _step(prompts.step_back_solve(ctx.prompt, ctx.task, principle), "solve")
    if not solve.ok:
        ctx.session.status = "error"
        ctx.session.stop_reason = f"model failed: {solve.error[:60]}"
        ctx.session.rounds.append(rnd)
        return ctx.session
    answer = solve.text

    verdict, jturn = judge.evaluate(cfg, prov, 1, ctx.task, ctx.prompt,
                                    [("step-back", answer)], candidate_models=[m.model],
                                    store=ctx.store)
    rnd.turns.append(jturn)
    ctx.session.account(jturn)
    rnd.verdict = verdict
    rnd.best_content = answer
    ctx.session.rounds.append(rnd)
    ctx.session.final = answer
    ctx.session.final_score = verdict.score
    ctx.session.stop_reason = "step-back (abstract to a principle -> solve)"
    ctx.event("result", f"step-back: score {verdict.score:.0f}", score=verdict.score)
    return ctx.session
