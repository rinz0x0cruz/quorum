"""Unit tests for the ``quorum.prompts`` package.

These lock roadmap item #5 (splitting the flat ``prompts.py`` into a package by
concern) as a behavior-preserving move: every builder and every SYSTEM constant
the flat module exposed must remain importable as ``prompts.<name>``, and the
``mock``-provider sentinels baked into the role system prompts must stay
byte-identical. All offline, deterministic, stdlib-only.
"""
from quorum import prompts


EXPECTED_SYSTEMS = [
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
]

EXPECTED_BUILDERS = [
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
]


# --- the package re-exports every prior public name ----------------------
def test_package_reexports_every_public_name():
    for name in EXPECTED_SYSTEMS + EXPECTED_BUILDERS:
        assert hasattr(prompts, name), f"prompts.{name} missing after the package split"


def test_all_lists_exactly_the_public_names():
    assert set(prompts.__all__) == set(EXPECTED_SYSTEMS + EXPECTED_BUILDERS)


def test_every_system_is_a_nonempty_str():
    for name in EXPECTED_SYSTEMS:
        assert isinstance(getattr(prompts, name), str) and getattr(prompts, name)


def test_every_builder_is_callable():
    for name in EXPECTED_BUILDERS:
        assert callable(getattr(prompts, name))


# --- mock-provider sentinels are intact ----------------------------------
def test_mock_sentinels_stay_byte_identical():
    # The offline MockResponder routes on these substrings; they must not drift.
    assert "QUORUM-AGGREGATOR" in prompts.AGGREGATOR_SYSTEM
    assert "QUORUM-CHAIRMAN" in prompts.CHAIRMAN_SYSTEM
    assert "QUORUM-REVIEW" in prompts.REVIEW_SYSTEM


# --- builders return the same message shapes -----------------------------
def test_propose_returns_two_message_list():
    msgs = prompts.propose("instruction", "the task")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == prompts.PROPOSER_SYSTEM
    assert "the task" in msgs[1]["content"]


def test_challenge_uses_the_devils_advocate_system():
    msgs = prompts.challenge("p", "t", "prev", [("a", "x")], "c", True)
    assert len(msgs) == 2
    assert msgs[0]["content"] == prompts.CHALLENGER_SYSTEM
    assert "devil" in msgs[0]["content"].lower()


def test_synthesize_and_aggregate_carry_their_sentinels():
    syn = prompts.synthesize("t", "p", [("a", "x")], ["r"], anonymize=True)
    agg = prompts.aggregate("t", "p", [("a", "x")], anonymize=True)
    assert "QUORUM-CHAIRMAN" in syn[0]["content"]
    assert "QUORUM-AGGREGATOR" in agg[0]["content"]
