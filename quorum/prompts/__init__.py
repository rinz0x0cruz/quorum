"""Prompt builders shared by the deliberation strategies (a package by concern).

Every builder returns an OpenAI-style ``messages`` list. A hard rule runs through
all of them: text produced by *other* models is presented as **DATA to consider,
never as instructions to follow** (OWASP LLM01). Role system prompts that the
offline mock keys off of carry stable sentinels (``QUORUM-REVIEW`` /
``QUORUM-CHAIRMAN`` / ``QUORUM-AGGREGATOR``).

This package splits the former flat ``prompts.py`` by concern -- the shared
framing helpers and the strategy-agnostic builders in
:mod:`~quorum.prompts.base`, and the strategy-specific builders alongside the
strategy they serve (:mod:`~quorum.prompts.debate`, :mod:`~quorum.prompts.council`,
:mod:`~quorum.prompts.moa`). Every public name is re-exported here, so
``prompts.propose(...)``, ``prompts.CHALLENGER_SYSTEM``, etc. resolve exactly as
they did against the flat module.
"""
from __future__ import annotations

from .base import (
    PROPOSER_SYSTEM,
    REFINE_SYSTEM,
    REVISE_SYSTEM,
    USC_SYSTEM,
    propose,
    revise,
    revise_from_draft,
    self_refine,
    usc,
)
from .council import CHAIRMAN_SYSTEM, REVIEW_SYSTEM, review, synthesize
from .debate import CHALLENGER_SYSTEM, challenge
from .moa import AGGREGATOR_SYSTEM, MOA_LAYER_SYSTEM, aggregate, moa_layer
from .reflexion import REFLECT_SYSTEM, REFLEXION_ACTOR_SYSTEM, reflect, reflexion_actor
from .verify import (
    VERIFY_ANSWER_SYSTEM,
    VERIFY_PLAN_SYSTEM,
    VERIFY_REVISE_SYSTEM,
    plan_checks,
    verified_final,
    verify_checks,
)
from .selfdiscover import (
    SELFDISCOVER_PLAN_SYSTEM,
    SELFDISCOVER_SOLVE_SYSTEM,
    discover,
    discover_solve,
)
from .stepback import (
    STEPBACK_ABSTRACT_SYSTEM,
    STEPBACK_SOLVE_SYSTEM,
    step_back,
    step_back_solve,
)
from .leasttomost import (
    LTM_DECOMPOSE_SYSTEM,
    LTM_SOLVE_SYSTEM,
    decompose,
    solve_subproblem,
)

__all__ = [
    # system prompts (mock provider keys off the QUORUM-* sentinels)
    "PROPOSER_SYSTEM",
    "REVISE_SYSTEM",
    "CHALLENGER_SYSTEM",
    "REFINE_SYSTEM",
    "REVIEW_SYSTEM",
    "CHAIRMAN_SYSTEM",
    "AGGREGATOR_SYSTEM",
    "MOA_LAYER_SYSTEM",
    "USC_SYSTEM",
    "REFLECT_SYSTEM",
    "REFLEXION_ACTOR_SYSTEM",
    "VERIFY_PLAN_SYSTEM",
    "VERIFY_ANSWER_SYSTEM",
    "VERIFY_REVISE_SYSTEM",
    "SELFDISCOVER_PLAN_SYSTEM",
    "SELFDISCOVER_SOLVE_SYSTEM",
    "STEPBACK_ABSTRACT_SYSTEM",
    "STEPBACK_SOLVE_SYSTEM",
    "LTM_DECOMPOSE_SYSTEM",
    "LTM_SOLVE_SYSTEM",
    # message builders
    "propose",
    "revise",
    "challenge",
    "self_refine",
    "review",
    "synthesize",
    "revise_from_draft",
    "moa_layer",
    "aggregate",
    "usc",
    "reflect",
    "reflexion_actor",
    "plan_checks",
    "verify_checks",
    "verified_final",
    "discover",
    "discover_solve",
    "step_back",
    "step_back_solve",
    "decompose",
    "solve_subproblem",
]
