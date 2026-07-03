from quorum import rank


def test_consensus_order_best_first():
    # B is ranked first by both reviewers -> index 1 should lead.
    reviews = ["Ranking (best first): B, A, C", "Ranking: B, C, A"]
    order = rank.consensus_order(3, reviews)
    assert order[0] == 1


def test_top_k_indices_limits_and_orders():
    assert rank.top_k_indices(3, ["A, B, C"], 2) == [0, 1]
    # k <= 0 returns the full order
    assert rank.top_k_indices(3, ["A, B, C"], 0) == [0, 1, 2]


def test_candidate_word_form():
    reviews = ["Candidate C is best, then Candidate A, then Candidate B"]
    assert rank.consensus_order(3, reviews)[0] == 2


def test_no_reviews_returns_identity_order():
    assert rank.consensus_order(3, []) == [0, 1, 2]
    assert rank.consensus_order(3, ["", None]) == [0, 1, 2]  # unusable reviews


def test_unmentioned_candidates_are_appended():
    # Only A and C mentioned; B (index 1) still ranked (last).
    order = rank.consensus_order(3, ["Best: A then C"])
    assert set(order) == {0, 1, 2} and order[0] == 0
