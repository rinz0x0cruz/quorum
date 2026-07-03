"""Ensemble / self-consistency baseline.

Sample the first member ``run.samples`` times (with a temperature bump for
diversity) and let the judge pick the best in a single round. A cheap baseline
that multi-agent-debate work ("Should we be going MAD?", Smit et al. 2023) found
competitive, so it belongs in the comparison.
"""
from __future__ import annotations

from .. import judge, prompts, provider
from ..model import Round
from . import Context


def run(ctx: Context):
    cfg, prov = ctx.cfg, ctx.prov
    o = ctx.opts
    n = o.samples
    if not ctx.members:
        ctx.session.status = "error"
        ctx.session.stop_reason = "no members configured"
        return ctx.session
    m = ctx.members[0]
    temp = o.temperature + 0.3

    rnd = Round(index=1)
    jobs = [(m, prompts.propose(ctx.prompt, ctx.task)) for _ in range(n)]
    comps = prov.complete_many(jobs, temperature=temp, store=ctx.store)
    candidates, cand_models = [], []
    for i, comp in enumerate(comps):
        if not comp.ok:
            continue
        turn = provider.to_turn(comp, 1, f"sample{i + 1}", "propose")
        rnd.turns.append(turn)
        ctx.session.account(turn)
        candidates.append((f"sample{i + 1}", comp.text))
        cand_models.append(m.model)

    if not candidates:
        ctx.session.status = "error"
        ctx.session.stop_reason = "all samples failed"
        return ctx.session

    verdict, jturn = judge.evaluate(cfg, prov, 1, ctx.task, ctx.prompt, candidates,
                                    candidate_models=cand_models, store=ctx.store)
    rnd.turns.append(jturn)
    ctx.session.account(jturn)
    rnd.verdict = verdict
    rnd.best_content = verdict.best_content
    ctx.session.rounds.append(rnd)

    ctx.session.final = verdict.best_content
    ctx.session.final_score = verdict.score
    ctx.session.stop_reason = f"best of {len(candidates)} samples"
    ctx.emit(f"ensemble: best score {verdict.score:.0f} ({len(candidates)} samples)")
    return ctx.session
