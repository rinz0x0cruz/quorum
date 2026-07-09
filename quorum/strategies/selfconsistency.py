"""Self-Consistency with USC selection + adaptive stopping.

Sample one model several times and return the *consensus* answer rather than a
judge-picked one (Wang et al. 2022). Selection is free when answers carry a
number (majority vote); for free-form answers it uses Universal Self-Consistency
-- ask a model to choose the most consistent candidate (Chen et al. 2023). With
``run.adaptive_samples`` the loop stops once a confident majority emerges
(Aggarwal et al. 2023), so it also spends fewer calls. Sequential by design,
which additionally avoids the burst that trips a per-minute rate cap.
"""
from __future__ import annotations

from .. import consistency, judge, prompts, provider
from ..model import Round
from . import Context


def run(ctx: Context):
    o = ctx.opts
    if not ctx.members:
        ctx.session.status = "error"
        ctx.session.stop_reason = "no members configured"
        return ctx.session
    m = ctx.members[0]
    temp = o.temperature + 0.3
    max_n = max(o.samples, o.samples_min)

    rnd = Round(index=1)
    clusters: list[dict] = []
    n_ok = 0
    for i in range(1, max_n + 1):
        comp = ctx.prov.complete(m, prompts.propose(ctx.prompt, ctx.task),
                                 temperature=temp, store=ctx.store, cache=False)
        if not comp.ok:
            continue
        turn = provider.to_turn(comp, 1, f"sample{i}", "propose")
        rnd.turns.append(turn)
        ctx.session.account(turn)
        consistency.assign(clusters, comp.text)
        n_ok += 1
        if o.adaptive_samples and n_ok >= o.samples_min and consistency.confident(clusters):
            break

    if not clusters:
        ctx.session.status = "error"
        ctx.session.stop_reason = "all samples failed"
        return ctx.session

    top = consistency.leader(clusters)
    if top.get("key"):                                   # numeric -> free majority vote
        winner, how = top["rep"], f"numeric vote {top['count']}/{n_ok}"
    else:                                                # free-form -> USC selection
        winner, how = _usc_pick(ctx, m, clusters, rnd, fallback=top["rep"])

    verdict, jturn = judge.evaluate(ctx.cfg, ctx.prov, 1, ctx.task, ctx.prompt,
                                    [("consensus", winner)], candidate_models=[m.model],
                                    store=ctx.store)
    rnd.turns.append(jturn)
    ctx.session.account(jturn)
    rnd.verdict = verdict
    rnd.best_content = winner
    ctx.session.rounds.append(rnd)
    ctx.session.final = winner
    ctx.session.final_score = verdict.score
    ctx.session.stop_reason = f"self-consistency ({how})"
    ctx.event("result", f"self-consistency: {n_ok} samples, {len(clusters)} distinct, {how}, "
              f"score {verdict.score:.0f}", score=verdict.score, samples=n_ok)
    return ctx.session


def _usc_pick(ctx: Context, m, clusters: list[dict], rnd: Round, *, fallback: str):
    """Universal Self-Consistency: let the model pick the most consistent answer."""
    reps = [(f"c{i}", c["rep"]) for i, c in enumerate(clusters)]
    if len(reps) == 1:
        return reps[0][1], "unanimous"
    comp = ctx.prov.complete(m, prompts.usc(ctx.task, reps), temperature=0.0, store=ctx.store)
    if not comp.ok:
        return fallback, "USC fallback (majority)"
    turn = provider.to_turn(comp, 1, "usc", "select")
    rnd.turns.append(turn)
    ctx.session.account(turn)
    payload = judge._parse_json(comp.text)
    letter = str(payload.get("choice", "A")).strip().upper()[:1]
    idx = ord(letter) - 65 if letter.isalpha() else 0
    if 0 <= idx < len(reps):
        return reps[idx][1], "USC selection"
    return fallback, "USC fallback (majority)"
