"""Difficulty-adaptive cascade (FrugalGPT; Chen, Zaharia & Zou 2023).

Run a sequence of increasingly capable -- and increasingly expensive --
strategies, cheapest first, and stop as soon as the judge clears the target
score. Most tasks are solved by the cheap first stage in a couple of calls; only
the hard ones escalate to multi-model debate/council. That spends requests only
when a task actually needs them, which is exactly what free-tier rate limits
reward.

Config: ``run.cascade`` is the ordered list of strategy names (defaults to
``[refine, debate, council]``). Escalation stops when a stage's final score
reaches ``run.target_score``, or the cost budget is hit. Because a pricier stage
can occasionally score *lower* on free models, the best answer seen across all
stages is what gets returned.
"""
from __future__ import annotations

from .. import cost
from . import Context, get

_DEFAULT = ["refine", "debate", "council"]


def run(ctx: Context):
    stages = [s for s in (ctx.opts.cascade or _DEFAULT) if s and s != "cascade"] or _DEFAULT
    target = ctx.opts.target_score
    best_final, best_score, best_stage = "", -1.0, ""

    for i, name in enumerate(stages, start=1):
        try:
            strat = get(name)
        except KeyError:
            ctx.emit(f"cascade: unknown stage '{name}', skipping")
            continue
        strat(ctx)
        score = ctx.session.final_score
        ctx.event("phase", f"cascade stage {i}/{len(stages)} [{name}]: score {score:.0f}", stage=i, strategy=name, score=score)
        if score > best_score:
            best_final, best_score, best_stage = ctx.session.final, score, name
        if score >= target:
            ctx.session.stop_reason = f"cascade: {name} reached target ({score:.0f} >= {target:g})"
            break
        if cost.over_budget(ctx.cfg, ctx.session.cost_usd):
            ctx.session.stop_reason = "cascade: cost budget exceeded"
            break
    else:
        ctx.session.stop_reason = (f"cascade: exhausted {len(stages)} stages, "
                                   f"best {best_score:.0f} ({best_stage})")

    # A later (pricier) stage can score lower on free models -- return the best.
    ctx.session.final = best_final
    ctx.session.final_score = max(best_score, 0.0)
    ctx.session.status = "ok" if best_final else "error"
    return ctx.session
