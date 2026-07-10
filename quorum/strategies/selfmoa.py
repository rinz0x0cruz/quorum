"""Self-MoA (Li et al. 2025, "Rethinking Mixture-of-Agents").

Instead of mixing different LLMs, sample the single strongest model several times
and aggregate those samples into one answer. The paper finds this often *beats* a
mixed Mixture-of-Agents, because blending in weaker models drags the average
quality down -- especially relevant on free tiers where model quality varies a
lot. Uses the first council member as the proposer (order your council
best-first) and the configured aggregator to merge; compare it with
``quorum bench --strategies selfmoa,moa,council ...``.
"""
from __future__ import annotations

from .. import judge, prompts, provider
from ..config import role_spec
from ..model import Round
from . import Context


def run(ctx: Context):
    cfg, prov = ctx.cfg, ctx.prov
    o = ctx.opts
    if not ctx.members:
        ctx.session.status = "error"
        ctx.session.stop_reason = "no members configured"
        return ctx.session
    m = ctx.members[0]                 # the single strongest model (council ordered best-first)
    n = max(2, o.samples)
    temp = o.temperature + 0.3

    rnd = Round(index=1)
    # cache=False so repeated identical calls actually resample (diversity).
    jobs = [(m, prompts.propose(ctx.prompt, ctx.task)) for _ in range(n)]
    comps = prov.complete_many(jobs, temperature=temp, store=ctx.store, cache=False)
    proposals = []
    for i, comp in enumerate(comps):
        if not comp.ok:
            continue
        turn = provider.to_turn(comp, 1, f"sample{i + 1}", "propose")
        rnd.turns.append(turn)
        ctx.session.account(turn)
        proposals.append((f"sample{i + 1}", comp.text))

    if not proposals:
        ctx.session.status = "error"
        ctx.session.stop_reason = "all samples failed"
        return ctx.session

    agg = role_spec(cfg, "aggregator")
    acomp = prov.complete(agg, prompts.aggregate(ctx.task, ctx.prompt, proposals,
                                                 anonymize=o.anonymize), store=ctx.store)
    answer = acomp.text if acomp.ok else proposals[0][1]
    aturn = provider.to_turn(acomp, 1, "aggregator", "aggregate")
    rnd.turns.append(aturn)
    ctx.session.account(aturn)

    verdict, jturn = judge.evaluate(cfg, prov, 1, ctx.task, ctx.prompt,
                                    [("aggregated", answer)], candidate_models=[agg.model],
                                    store=ctx.store)
    rnd.turns.append(jturn)
    ctx.session.account(jturn)
    rnd.verdict = verdict
    rnd.best_content = answer
    ctx.session.rounds.append(rnd)

    ctx.session.final = answer
    ctx.session.final_score = verdict.score
    ctx.session.stop_reason = f"self-moa ({len(proposals)} samples of {m.name} aggregated)"
    ctx.event("result", f"self-moa: score {verdict.score:.0f} ({len(proposals)} samples)",
              score=verdict.score, samples=len(proposals))
    return ctx.session
