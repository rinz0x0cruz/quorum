"""Mixture-of-Agents (Wang et al. 2024).

A layered architecture: in each layer every member answers, seeing all of the
previous layer's responses as auxiliary information; a final aggregator merges
the last layer into one answer, which the judge scores. Compute is fixed by
``run.moa_layers`` rather than driven by the score.
"""
from __future__ import annotations

from .. import judge, prompts, provider, rank
from ..config import role_spec
from ..model import Round
from . import Context


def run(ctx: Context):
    cfg, prov, members = ctx.cfg, ctx.prov, ctx.members
    run = cfg.get("run", {}) or {}
    layers = max(1, int(run.get("moa_layers", 2)))
    anon = bool(run.get("anonymize", True))

    prev: list[tuple[str, str]] = []   # (member, text) from the previous layer
    for layer in range(1, layers + 1):
        rnd = Round(index=layer)
        jobs = []
        for m in members:
            if layer == 1:
                msgs = prompts.propose(ctx.prompt, ctx.task)
            else:
                msgs = prompts.moa_layer(ctx.prompt, ctx.task, prev, anonymize=anon)
            jobs.append((m, msgs))
        comps = prov.complete_many(jobs, store=ctx.store)
        current = []
        for m, comp in zip(members, comps):
            if not comp.ok:
                continue
            turn = provider.to_turn(comp, layer, m.name, "propose")
            rnd.turns.append(turn)
            ctx.session.account(turn)
            current.append((m.name, comp.text))
        if not current:
            ctx.session.status = "error"
            ctx.session.stop_reason = "all members failed"
            return ctx.session
        prev = current
        ctx.session.rounds.append(rnd)
        ctx.emit(f"layer {layer}: {len(current)} responses")

    # final aggregation + judge
    agg = role_spec(cfg, "aggregator")

    # optional (LLM-Blender): rank the last layer and aggregate only the top-K.
    # MoA has no peer reviews, so add one lightweight review call when enabled.
    top_k = int(run.get("top_k", 0) or 0)
    fuse = prev
    if 0 < top_k < len(prev):
        reviewer = role_spec(cfg, "aggregator")
        rcomp = prov.complete(reviewer, prompts.review(ctx.task, prev, anonymize=anon),
                              store=ctx.store)
        if rcomp.ok:
            rturn = provider.to_turn(rcomp, layers, "reviewer", "review")
            ctx.session.rounds[-1].turns.append(rturn)
            ctx.session.account(rturn)
            idxs = rank.top_k_indices(len(prev), [rcomp.text], top_k)
            fuse = [prev[i] for i in idxs]
            ctx.emit(f"rank: fusing top {len(fuse)} of {len(prev)}")

    final_round = Round(index=layers + 1)
    acomp = prov.complete(agg, prompts.aggregate(ctx.task, ctx.prompt, fuse, anonymize=anon),
                          store=ctx.store)
    aturn = provider.to_turn(acomp, layers + 1, "aggregator", "aggregate")
    final_round.turns.append(aturn)
    ctx.session.account(aturn)

    verdict, jturn = judge.evaluate(cfg, prov, layers + 1, ctx.task, ctx.prompt,
                                    [("aggregator", acomp.text)], candidate_models=[agg.model],
                                    store=ctx.store)
    final_round.turns.append(jturn)
    ctx.session.account(jturn)
    final_round.verdict = verdict
    final_round.best_content = acomp.text
    ctx.session.rounds.append(final_round)

    ctx.session.final = acomp.text
    ctx.session.final_score = verdict.score
    ctx.session.stop_reason = f"completed {layers} layers"
    ctx.emit(f"aggregate: score {verdict.score:.0f}")
    return ctx.session
