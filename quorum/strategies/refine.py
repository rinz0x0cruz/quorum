"""Self-Refine baseline (Madaan et al. 2023).

A single model generates, critiques its own answer, and revises -- iterating
until the judge is satisfied or the round cap is hit. Included as a cheap
reference point for the benchmark (single agent, no cross-model debate).
"""
from __future__ import annotations

from .. import cost, judge, prompts, provider
from ..model import Round
from . import Context


def run(ctx: Context):
    cfg, prov = ctx.cfg, ctx.prov
    max_rounds = ctx.opts.max_rounds
    if not ctx.members:
        ctx.session.status = "error"
        ctx.session.stop_reason = "no members configured"
        return ctx.session
    m = ctx.members[0]

    verdicts = []
    content = ""
    for r in range(1, max_rounds + 1):
        rnd = Round(index=r)
        if r == 1:
            msgs = prompts.propose(ctx.prompt, ctx.task)
        else:
            critique = verdicts[-1].rationale if verdicts else ""
            msgs = prompts.self_refine(ctx.prompt, ctx.task, content, critique)
        comp = prov.complete(m, msgs, store=ctx.store)
        if not comp.ok:
            ctx.session.status = "error"
            ctx.session.stop_reason = f"model failed: {comp.error[:60]}"
            break
        turn = provider.to_turn(comp, r, m.name, "propose" if r == 1 else "revise")
        rnd.turns.append(turn)
        ctx.session.account(turn)
        content = comp.text

        verdict, jturn = judge.evaluate(cfg, prov, r, ctx.task, ctx.prompt,
                                        [(m.name, content)], candidate_models=[m.model],
                                        store=ctx.store)
        rnd.turns.append(jturn)
        ctx.session.account(jturn)
        rnd.verdict = verdict
        rnd.best_content = content
        verdicts.append(verdict)
        ctx.session.rounds.append(rnd)
        ctx.emit(f"round {r}: score {verdict.score:.0f}")

        if cost.over_budget(cfg, ctx.session.cost_usd):
            ctx.session.stop_reason = "cost budget exceeded"
            ctx.session.status = "aborted"
            break
        stop, reason = judge.should_stop(cfg, verdicts, r)
        if stop:
            verdict.stop = True
            verdict.reason = reason
            ctx.session.stop_reason = reason
            break

    if verdicts:
        best = max(verdicts, key=lambda v: v.score)
        ctx.session.final = best.best_content
        ctx.session.final_score = best.score
    return ctx.session
