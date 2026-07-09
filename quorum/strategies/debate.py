"""Debate + judge (Du et al. 2023; Liang et al. MAD, EMNLP 2024).

Every member proposes an answer; each subsequent round they revise given the
other (optionally anonymised) answers plus the judge's critique. The judge scores
the round's best candidate; the loop stops on target/plateau/cap (or consensus).
"""
from __future__ import annotations

from .. import cost, judge, prompts, provider
from ..model import Round
from . import Context


def run(ctx: Context):
    cfg, prov, members = ctx.cfg, ctx.prov, ctx.members
    o = ctx.opts
    max_rounds = o.max_rounds
    anon = o.anonymize
    # devil's advocate (MAD): one member argues the counter-case from round 2 on,
    # keeping the debate divergent and avoiding premature consensus.
    devil = members[-1] if (o.devils_advocate and len(members) >= 2) else None

    verdicts = []
    latest: dict[str, str] = {}   # member name -> latest answer

    for r in range(1, max_rounds + 1):
        rnd = Round(index=r)
        jobs = []
        for m in members:
            if r == 1:
                msgs = prompts.propose(ctx.prompt, ctx.task)
            else:
                peers = [(nm, latest[nm]) for nm in latest if nm != m.name]
                critique = verdicts[-1].rationale if verdicts else ""
                if m is devil:
                    msgs = prompts.challenge(ctx.prompt, ctx.task, latest.get(m.name, ""),
                                             peers, critique, anon)
                else:
                    msgs = prompts.revise(ctx.prompt, ctx.task, latest.get(m.name, ""),
                                          peers, critique, anon)
            jobs.append((m, msgs))

        comps = prov.complete_many(jobs, store=ctx.store)
        candidates, cand_models = [], []
        for m, comp in zip(members, comps):
            if not comp.ok:
                ctx.emit(f"  round {r}: {m.name} failed ({comp.error[:60]})")
                continue
            kind = "propose" if r == 1 else ("challenge" if m is devil else "revise")
            turn = provider.to_turn(comp, r, m.name, kind)
            rnd.turns.append(turn)
            ctx.session.account(turn)
            latest[m.name] = comp.text
            candidates.append((m.name, comp.text))
            cand_models.append(m.model)

        if not candidates:
            ctx.session.status = "error"
            ctx.session.stop_reason = "all members failed"
            break

        rnd.best_content = candidates[0][1]
        if judge.due(r, o.judge_every, max_rounds):
            verdict, jturn = judge.evaluate(cfg, prov, r, ctx.task, ctx.prompt, candidates,
                                            candidate_models=cand_models, store=ctx.store)
            rnd.turns.append(jturn)
            ctx.session.account(jturn)
            rnd.verdict = verdict
            rnd.best_content = verdict.best_content
            verdicts.append(verdict)
            ctx.emit(f"round {r}: score {verdict.score:.0f} (best={verdict.best_label})")
        else:
            ctx.emit(f"round {r}: (deferred judge)")
        ctx.session.rounds.append(rnd)

        if cost.over_budget(cfg, ctx.session.cost_usd):
            ctx.session.stop_reason = "cost budget exceeded"
            ctx.session.status = "aborted"
            break

        if verdicts and judge.due(r, o.judge_every, max_rounds):
            stop, reason = judge.should_stop(cfg, verdicts, r)
            if not stop and o.consensus and judge.consensus_reached(list(latest.values())):
                stop, reason = True, "members reached consensus"
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
