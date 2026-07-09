"""Council + chairman (iterative extension of Karpathy's llm-council).

Each round: members answer, then peer-review the anonymised answers, then a
chairman synthesises a single final answer. The judge scores the chairman's
answer; if it is not yet good enough the draft + critique are fed back for
another round. Unlike the single-pass original, this loops until good enough.
"""
from __future__ import annotations

from .. import cost, judge, prompts, provider, rank
from ..config import role_spec
from ..model import Round
from . import Context


def run(ctx: Context):
    cfg, prov, members = ctx.cfg, ctx.prov, ctx.members
    o = ctx.opts
    max_rounds = o.max_rounds
    anon = o.anonymize
    chair = role_spec(cfg, "chairman")

    verdicts = []
    draft = ""

    for r in range(1, max_rounds + 1):
        rnd = Round(index=r)

        # 1) members answer (or improve the current draft)
        jobs = []
        for m in members:
            if r == 1:
                msgs = prompts.propose(ctx.prompt, ctx.task)
            else:
                critique = verdicts[-1].rationale if verdicts else ""
                msgs = prompts.revise_from_draft(ctx.prompt, ctx.task, draft, critique)
            jobs.append((m, msgs))
        comps = prov.complete_many(jobs, store=ctx.store)
        answers = []
        for m, comp in zip(members, comps):
            if not comp.ok:
                continue
            turn = provider.to_turn(comp, r, m.name, "propose" if r == 1 else "revise")
            rnd.turns.append(turn)
            ctx.session.account(turn)
            answers.append((m.name, comp.text))
        if not answers:
            ctx.session.status = "error"
            ctx.session.stop_reason = "all members failed"
            break

        # 2) peer review (anonymised)
        rjobs = [(m, prompts.review(ctx.task, answers, anonymize=True)) for m in members]
        rcomps = prov.complete_many(rjobs, store=ctx.store)
        reviews = []
        for m, comp in zip(members, rcomps):
            if not comp.ok:
                continue
            turn = provider.to_turn(comp, r, m.name, "review")
            rnd.turns.append(turn)
            ctx.session.account(turn)
            reviews.append(comp.text)

        # optional (LLM-Blender): rank by the peer reviews and fuse only the top-K
        top_k = o.top_k
        fuse_answers = answers
        if 0 < top_k < len(answers):
            idxs = rank.top_k_indices(len(answers), reviews, top_k)
            fuse_answers = [answers[i] for i in idxs]
            ctx.emit(f"round {r}: fusing top {len(fuse_answers)} of {len(answers)}")

        # 3) chairman synthesis
        cmsgs = prompts.synthesize(ctx.task, ctx.prompt, fuse_answers, reviews, anonymize=anon)
        ccomp = prov.complete(chair, cmsgs, store=ctx.store)
        cturn = provider.to_turn(ccomp, r, "chairman", "synthesize")
        rnd.turns.append(cturn)
        ctx.session.account(cturn)
        draft = ccomp.text if ccomp.ok else draft

        # 4) judge the chairman's answer (deferrable via run.judge_every)
        rnd.best_content = draft
        if judge.due(r, o.judge_every, max_rounds):
            verdict, jturn = judge.evaluate(cfg, prov, r, ctx.task, ctx.prompt,
                                            [("chairman", draft)], candidate_models=[chair.model],
                                            store=ctx.store)
            rnd.turns.append(jturn)
            ctx.session.account(jturn)
            rnd.verdict = verdict
            verdicts.append(verdict)
            ctx.event("round", f"round {r}: score {verdict.score:.0f} (chairman)", round=r, score=verdict.score)
        else:
            ctx.emit(f"round {r}: (deferred judge)")
        ctx.session.rounds.append(rnd)

        if cost.over_budget(cfg, ctx.session.cost_usd):
            ctx.session.stop_reason = "cost budget exceeded"
            ctx.session.status = "aborted"
            break
        if verdicts and judge.due(r, o.judge_every, max_rounds):
            stop, reason = judge.should_stop(cfg, verdicts, r)
            if stop:
                verdicts[-1].stop = True
                verdicts[-1].reason = reason
                ctx.session.stop_reason = reason
                break

    if verdicts:
        best = max(verdicts, key=lambda v: v.score)
        ctx.session.final = best.best_content
        ctx.session.final_score = best.score
    return ctx.session
