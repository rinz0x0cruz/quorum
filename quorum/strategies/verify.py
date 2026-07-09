"""Chain-of-Verification (Dhuliawala et al. 2023).

Draft an answer, plan verification questions about it, answer those questions
*independently* (the draft is withheld so its errors are not just echoed), then
revise the draft from the findings. A single verified pass -- the judge scores the
final answer. Cheaper than debate but markedly less hallucination-prone.
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

    draft = _step(prompts.propose(ctx.prompt, ctx.task), "draft")
    if not draft.ok:
        ctx.session.status = "error"
        ctx.session.stop_reason = f"model failed: {draft.error[:60]}"
        return ctx.session

    plan = _step(prompts.plan_checks(ctx.prompt, ctx.task, draft.text), "plan")
    questions = plan.text if plan.ok else ""
    verify = _step(prompts.verify_checks(ctx.prompt, ctx.task, questions), "verify")
    qa = verify.text if verify.ok else ""
    final = _step(prompts.verified_final(ctx.prompt, ctx.task, draft.text, qa), "revise")
    answer = final.text if final.ok else draft.text

    verdict, jturn = judge.evaluate(cfg, prov, 1, ctx.task, ctx.prompt,
                                    [("verified", answer)], candidate_models=[m.model],
                                    store=ctx.store)
    rnd.turns.append(jturn)
    ctx.session.account(jturn)
    rnd.verdict = verdict
    rnd.best_content = answer
    ctx.session.rounds.append(rnd)
    ctx.session.final = answer
    ctx.session.final_score = verdict.score
    ctx.session.stop_reason = "chain-of-verification (draft -> checks -> verify -> revise)"
    ctx.event("result", f"verify: score {verdict.score:.0f}", score=verdict.score)
    return ctx.session
