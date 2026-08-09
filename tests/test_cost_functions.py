"""Executable specification of costs.py, with cases worked out BY HAND.

This test was committed before costs.py was implemented. Every number here
came off a pencil, not off running the code: if the implementation disagrees,
the implementation is what is wrong. Do not edit the test to make it pass. Edit
it only on finding an error IN THE PENCIL WORK.

Reference parameters, from design.md §2.1: F=20, m=0.25, phi=10, r=2.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fraudq.policy.costs import (
    cost_approve,
    cost_block,
    realized_cost,
    value_of_review,
)

CFG = SimpleNamespace(F=20.0, m=0.25, phi=10.0, r=2.0)


# --------------------------------------------------------- extremes (p=0, p=1)


def test_certain_fraud_makes_blocking_free():
    """p=1: blocking loses nothing, there was no legitimate sale to lose."""
    assert cost_block(1.0, 100.0, CFG) == pytest.approx(0.0)
    # ...and approving costs the full amount plus the chargeback fee.
    assert cost_approve(1.0, 100.0, CFG) == pytest.approx(120.0)


def test_certain_legit_makes_approving_free():
    """p=0: approving is free; blocking costs margin plus friction."""
    assert cost_approve(0.0, 100.0, CFG) == pytest.approx(0.0)
    assert cost_block(0.0, 100.0, CFG) == pytest.approx(0.25 * 100.0 + 10.0)  # 35


# ------------------------------------------------------------ the pencil cases


def test_hand_computed_moderate_case():
    """p=0.2, amt=100: approve = 0.2*120 = 24; block = 0.8*35 = 28; V = 24-2 = 22."""
    assert cost_approve(0.2, 100.0, CFG) == pytest.approx(24.0)
    assert cost_block(0.2, 100.0, CFG) == pytest.approx(28.0)
    assert value_of_review(0.2, 100.0, CFG) == pytest.approx(22.0)


def test_hand_computed_small_amount():
    """p=0.5, amt=10: approve = 0.5*30 = 15; block = 0.5*12.5 = 6.25; V = 4.25."""
    assert cost_approve(0.5, 10.0, CFG) == pytest.approx(15.0)
    assert cost_block(0.5, 10.0, CFG) == pytest.approx(6.25)
    assert value_of_review(0.5, 10.0, CFG) == pytest.approx(4.25)


def test_value_is_negative_when_the_case_is_already_decided():
    """p=0.99, amt=10: block = 0.01*12.5 = 0.125 < r  =>  V = -1.875 < 0.

    The thesis in miniature: on an already clear case the human adds nothing,
    so reviewing destroys value.
    """
    assert value_of_review(0.99, 10.0, CFG) == pytest.approx(0.125 - 2.0)


def test_value_peaks_at_moderate_probability():
    """V(., amt) peaks at p* = (m*a+phi)/((a+F)+(m*a+phi)), design.md §2.4.

    For amt=100: p* = 35/155, about 0.2258, and there both costs come to
    0.2258*120 = 0.7742*35, about 27.10. This is a NUMERICAL check, so it does
    not require p_star to be implemented: V at p* beats V at p* plus or minus
    0.1, and the two costs cross at p*.
    """
    amt = 100.0
    p_star = (0.25 * amt + 10.0) / ((amt + 20.0) + (0.25 * amt + 10.0))
    assert p_star == pytest.approx(35.0 / 155.0)
    assert cost_approve(p_star, amt, CFG) == pytest.approx(cost_block(p_star, amt, CFG))
    v_peak = value_of_review(p_star, amt, CFG)
    assert v_peak > value_of_review(p_star - 0.1, amt, CFG)
    assert v_peak > value_of_review(p_star + 0.1, amt, CFG)
    assert v_peak == pytest.approx(p_star * (amt + 20.0) - 2.0)


# ------------------------------------------------------------- vectorisation


def test_vectorized_matches_scalar():
    """The contract demands numpy: arrays in, arrays out, no loops."""
    p = np.array([0.0, 0.2, 0.5, 0.99, 1.0])
    amt = np.array([100.0, 100.0, 10.0, 10.0, 100.0])
    va = cost_approve(p, amt, CFG)
    vb = cost_block(p, amt, CFG)
    vv = value_of_review(p, amt, CFG)
    for i in range(len(p)):
        assert va[i] == pytest.approx(cost_approve(float(p[i]), float(amt[i]), CFG))
        assert vb[i] == pytest.approx(cost_block(float(p[i]), float(amt[i]), CFG))
        assert vv[i] == pytest.approx(value_of_review(float(p[i]), float(amt[i]), CFG))


# -------------------------------------------------------------- realised cost


def test_realized_cost_hand_table():
    """The full accounting table, case by case (see realized_cost's docstring)."""
    actions = np.array(["approve", "approve", "block", "block", "review", "review"], dtype=object)
    is_fraud = np.array([1, 0, 0, 1, 1, 0])
    amt = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    out = np.asarray(realized_cost(actions, is_fraud, amt, CFG), dtype=float)
    #        fraud approved, legitimate approved, legitimate blocked,
    #        fraud blocked, review (always r)
    assert out == pytest.approx([120.0, 0.0, 35.0, 0.0, 2.0, 2.0])


def test_expected_cost_is_probability_weighted_realized_cost():
    """Internal coherence: E[realised] == expected, for each action.

    approve: p*(amt+F) + (1-p)*0 == cost_approve
    block:   p*0 + (1-p)*(m*amt+phi) == cost_block

    If this fails, the two sets of books contradict each other and the
    simulation is not measuring what the policy optimises.
    """
    p, amt = 0.3, 80.0
    exp_approve = p * float(
        realized_cost(np.array(["approve"], dtype=object), np.array([1]), np.array([amt]), CFG)[0]
    ) + (1 - p) * float(
        realized_cost(np.array(["approve"], dtype=object), np.array([0]), np.array([amt]), CFG)[0]
    )
    exp_block = p * float(
        realized_cost(np.array(["block"], dtype=object), np.array([1]), np.array([amt]), CFG)[0]
    ) + (1 - p) * float(
        realized_cost(np.array(["block"], dtype=object), np.array([0]), np.array([amt]), CFG)[0]
    )
    assert exp_approve == pytest.approx(cost_approve(p, amt, CFG))
    assert exp_block == pytest.approx(cost_block(p, amt, CFG))
