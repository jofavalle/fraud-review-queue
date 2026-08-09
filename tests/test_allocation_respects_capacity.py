"""Executable specification of allocate.py, THE THESIS included.

Like test_cost_functions: committed before the implementation, numbers worked
out by hand, and never edited to make it pass. The case
`test_thesis_in_one_case` is the whole project in four lines: an allocation
that passes it is no longer the naive queue.

Reference parameters: F=20, m=0.25, phi=10, r=2.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fraudq.policy.allocate import allocate_day
from fraudq.policy.costs import cost_approve, cost_block, value_of_review

CFG = SimpleNamespace(F=20.0, m=0.25, phi=10.0, r=2.0)


def _rng_case(n: int = 500, seed: int = 11):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.5, 6.0, size=n)  # mostly low p, with a high tail
    amt = np.exp(rng.normal(3.5, 1.2, size=n))  # log-normal amounts
    return p, amt


# ----------------------------------------------------------------- the thesis


def test_thesis_in_one_case():
    """Capacity 1: the high score is NOT reviewed; the ambiguous large one IS.

    A: p=0.95, amt=50  -> approve=66.5, block=1.125, V=-0.875 < 0. Already
       decided: blocking risks almost nothing, and a human adds nothing.
    B: p=0.25, amt=200 -> approve=55, block=45, V=43. Genuinely ambiguous and
       expensive: exactly what is worth putting in front of an analyst.

    The top-1 BY SCORE would review A. The top-1 BY VALUE reviews B.
    """
    p = np.array([0.95, 0.25])
    amt = np.array([50.0, 200.0])
    actions = allocate_day(p, amt, capacity=1, cfg=CFG)
    assert actions[1] == "review"
    assert actions[0] == "block"  # its cheaper automatic action


# --------------------------------------------------------------- the contract


def test_output_contract():
    p, amt = _rng_case()
    actions = np.asarray(allocate_day(p, amt, 25, CFG))
    assert actions.shape == p.shape
    assert set(np.unique(actions)) <= {"approve", "review", "block"}


def test_capacity_is_never_exceeded():
    p, amt = _rng_case()
    for capacity in (0, 1, 7, 50, 10_000):
        actions = np.asarray(allocate_day(p, amt, capacity, CFG))
        assert (actions == "review").sum() <= capacity


def test_reviews_are_exactly_the_top_value_candidates():
    """Genuinely greedy: the reviewed ones are the highest V among the V>0.

    With a stable tie-break by position, the expected set is computable without
    ambiguity: the first `capacity` indices of np.argsort(-v, kind="stable")
    restricted to v > 0.
    """
    p, amt = _rng_case()
    capacity = 25
    v = value_of_review(p, amt, CFG)
    eligible_sorted = [i for i in np.argsort(-v, kind="stable") if v[i] > 0]
    expected = set(eligible_sorted[:capacity])

    actions = np.asarray(allocate_day(p, amt, capacity, CFG))
    got = set(np.flatnonzero(actions == "review"))
    assert got == expected


def test_spare_capacity_is_not_wasted_on_nonpositive_value():
    """V <= 0 is not reviewed EVEN WITH CAPACITY TO SPARE: reviewing what is
    already decided is paying r for nothing (design.md §2.3)."""
    # All clear cases: very high or very low p, on small amounts.
    p = np.array([0.99, 0.995, 0.001, 0.002])
    amt = np.array([10.0, 15.0, 20.0, 25.0])
    assert np.all(value_of_review(p, amt, CFG) <= 0)  # the premise of the case
    actions = np.asarray(allocate_day(p, amt, capacity=10, cfg=CFG))
    assert (actions == "review").sum() == 0


def test_non_reviewed_get_the_cheaper_auto_action():
    p, amt = _rng_case()
    actions = np.asarray(allocate_day(p, amt, 25, CFG))
    approve_cost = cost_approve(p, amt, CFG)
    block_cost = cost_block(p, amt, CFG)
    for i in np.flatnonzero(actions != "review"):
        if approve_cost[i] <= block_cost[i]:  # a tie goes to approve, per the contract
            assert actions[i] == "approve"
        else:
            assert actions[i] == "block"


def test_capacity_zero_means_fully_automatic():
    p, amt = _rng_case(n=50)
    actions = np.asarray(allocate_day(p, amt, 0, CFG))
    assert (actions == "review").sum() == 0


def test_negative_capacity_raises():
    p, amt = _rng_case(n=10)
    with pytest.raises(ValueError):
        allocate_day(p, amt, -1, CFG)


def test_deterministic_under_ties():
    """Identical transactions and room for one: the FIRST wins, by the stable
    order, and two runs agree. The same principle as the TransactionID
    tie-break in the feature windows."""
    p = np.array([0.25, 0.25, 0.25])
    amt = np.array([200.0, 200.0, 200.0])
    first = np.asarray(allocate_day(p, amt, 1, CFG))
    second = np.asarray(allocate_day(p, amt, 1, CFG))
    assert (first == second).all()
    assert first[0] == "review"
    assert (first == "review").sum() == 1
