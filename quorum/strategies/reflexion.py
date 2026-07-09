"""Reflexion (Shinn et al. 2023).

A single actor generates an answer, the judge scores it, and the actor writes a
short verbal self-reflection appended to a growing memory. Each later attempt
conditions on *all* accumulated reflections -- a richer learning signal than the
last critique alone (cf. self-refine). Stops on the judge's target/plateau/cap;
honors run.judge_every and the cost budget.
"""
from __future__ import annotations

from .. import cost, judge, prompts, provider
from ..model import Round
from . import Context


def run(ctx: Context):
    cfg, prov = ctx.cfg, ctx.prov
    o = ctx.opts
    max_rounds = o.max_rounds
    if not ctx.members:
        ctx.session.status = "error"
        ctx.session.stop_reason = "no members configured"
        return ctx.session
    m = ctx.members[0]

    reflections: list[str] = []
    verdicts = []
    content = ""
    for r in range(1, max_rounds + 1):
        rnd = Round(index=r)
        msgs = (prompts.propose(ctx.prompt, ctx.task) if r == 1
                else prompts.reflexion_actor(ctx.prompt, ctx.task, reflections))
        comp = prov.complete(m, msgs, store=ctx.store)
        if not comp.ok:
            ctx.session.status = "error"
            ctx.session.stop_reason = f"model failed: {comp.error[:60]}"
            break
        turn = provider.to_turn(comp, r, m.name, "propose" if r == 1 else "act")
        rnd.turns.append(turn)
        ctx.session.account(turn)
        content = comp.text
        rnd.best_content = content

        judged = judge.due(r, o.judge_every, max_rounds)
        if judged:
            verdict, jturn = judge.evaluate(cfg, prov, r, ctx.task, ctx.prompt,
                                            [(m.name, content)], candidate_models=[m.model],
                                            store=ctx.store)
            rnd.turns.append(jturn)
            ctx.session.account(jturn)
            rnd.verdict = verdict
            verdicts.append(verdict)
            ctx.emit(f"round {r}: score {verdict.score:.0f}")
        else:
            ctx.emit(f"round {r}: (deferred judge)")

        if cost.over_budget(cfg, ctx.session.cost_usd):
            ctx.session.stop_reason = "cost budget exceeded"
            ctx.session.status = "aborted"
            ctx.session.rounds.append(rnd)
            break

        stop = False
        if verdicts and judged:
            stop, reason = judge.should_stop(cfg, verdicts, r)
            if stop:
                verdicts[-1].stop = True
                verdicts[-1].reason = reason
                ctx.session.stop_reason = reason

        # Reflect for the next attempt (skip on the last round / when stopping).
        if not stop and r < max_rounds:
            critique = verdicts[-1].rationale if verdicts else ""
            rcomp = prov.complete(m, prompts.reflect(ctx.prompt, ctx.task, content, critique),
                                  store=ctx.store)
            if rcomp.ok:
                reflections.append(rcomp.text)
                rturn = provider.to_turn(rcomp, r, m.name, "reflect")
                rnd.turns.append(rturn)
                ctx.session.account(rturn)

        ctx.session.rounds.append(rnd)
        if stop:
            break

    if verdicts:
        best = max(verdicts, key=lambda v: v.score)
        ctx.session.final = best.best_content
        ctx.session.final_score = best.score
    return ctx.session
