"""The sweep is an analysis, not a justification (design.md §7.2).

Two of these tests exist for a failure mode that does not raise. If the cost
parameters did not flow into the decision layer as an argument, every point of
the sweep would return the base answer, every swing would be zero, and the
tornado would report that the conclusion is perfectly robust. It would look like
a strong result and it would be an artefact of the plumbing. `test_the_sweep_moves_the_answer`
is the assertion that turns that into a failure.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fraudq.analysis import conclusion_survives, make_evaluate_fn
from fraudq.config import SENSITIVITY_RANGES, CostConfig
from fraudq.evaluate.sensitivity import savings_per_1k, tornado_data

TORNADO_COLUMNS = ["param", "low", "high", "savings_at_low", "savings_at_high", "swing"]


@pytest.fixture
def scored() -> pd.DataFrame:
    """A scored partition with enough spread for the two rankings to disagree.

    Amount and probability are deliberately anti-correlated: the cheapest
    transactions get the highest scores. That is the situation the project is
    about, and it is what makes ranking by value differ from ranking by score.
    """
    rng = np.random.default_rng(0)
    n_days, per_day = 6, 200
    p = rng.beta(0.6, 6.0, size=n_days * per_day)
    amt = 1000.0 / (1.0 + 40.0 * p) * rng.lognormal(0.0, 0.4, size=len(p))
    return pd.DataFrame(
        {
            "day": np.repeat(np.arange(n_days), per_day),
            "TransactionAmt": amt,
            "p": p,
            "isFraud": rng.binomial(1, p),
        }
    )


@pytest.fixture
def evaluate(scored):
    return make_evaluate_fn(scored, scored, capacity_pct=0.05)


def test_tornado_has_one_row_per_parameter(evaluate):
    tornado = tornado_data(CostConfig(), SENSITIVITY_RANGES, evaluate)
    assert list(tornado.columns) == TORNADO_COLUMNS
    assert set(tornado["param"]) == set(SENSITIVITY_RANGES)
    assert len(tornado) == len(SENSITIVITY_RANGES)


def test_tornado_is_sorted_by_swing_descending(evaluate):
    """The plot is read top to bottom, so the order is part of the contract."""
    swing = tornado_data(CostConfig(), SENSITIVITY_RANGES, evaluate)["swing"].to_numpy()
    assert np.all(np.diff(swing) <= 0)


def test_swing_is_the_absolute_gap_between_the_ends(evaluate):
    tornado = tornado_data(CostConfig(), SENSITIVITY_RANGES, evaluate)
    expected = (tornado["savings_at_high"] - tornado["savings_at_low"]).abs()
    assert np.allclose(tornado["swing"], expected)


def test_the_sweep_moves_the_answer(evaluate):
    """At least one parameter has to change the result, or nothing was swept."""
    tornado = tornado_data(CostConfig(), SENSITIVITY_RANGES, evaluate)
    assert tornado["swing"].max() > 0.0


def test_the_base_config_is_not_mutated(evaluate):
    """`dataclasses.replace` on a frozen config copies; it must not write through."""
    base = CostConfig()
    before = (base.F, base.m, base.phi, base.r)
    tornado_data(base, SENSITIVITY_RANGES, evaluate)
    assert (base.F, base.m, base.phi, base.r) == before


def test_the_ends_of_the_sweep_are_the_declared_ranges(evaluate):
    """Ranges come from config.py, fixed before the results were seen."""
    tornado = tornado_data(CostConfig(), SENSITIVITY_RANGES, evaluate).set_index("param")
    for param, (low, high) in SENSITIVITY_RANGES.items():
        assert (tornado.loc[param, "low"], tornado.loc[param, "high"]) == (low, high)


def test_savings_per_1k_is_score_ranked_minus_value_ranked(evaluate):
    comparison = evaluate(CostConfig())
    expected = (
        comparison.loc["topk_by_score", "cost_per_1k"]
        - comparison.loc["topk_by_value", "cost_per_1k"]
    )
    assert savings_per_1k(comparison) == pytest.approx(expected)


def test_conclusion_survives_reads_both_ends(evaluate):
    """False as soon as a single end of a single range takes the saving to zero."""
    tornado = tornado_data(CostConfig(), SENSITIVITY_RANGES, evaluate)
    assert conclusion_survives(tornado) == bool(
        (tornado[["savings_at_low", "savings_at_high"]] > 0).all().all()
    )

    broken = tornado.copy()
    broken.loc[0, "savings_at_low"] = -0.01
    assert not conclusion_survives(broken)


def test_the_threshold_is_refitted_per_sweep_point(scored):
    """Policy 2 has to be consistent with the config it is compared under.

    Chargeback fee and gross margin move the cost of approving against the cost
    of blocking, so the cost minimising threshold moves with them. A sweep that
    held one threshold fixed would compare policy 2 against assumptions it was
    not fitted for.
    """
    from fraudq.evaluate.policies import fit_single_threshold

    cheap_to_block = replace(CostConfig(), chargeback_fee=40.0)
    expensive_to_block = replace(CostConfig(), gross_margin=0.40)
    assert fit_single_threshold(scored, cheap_to_block) != fit_single_threshold(
        scored, expensive_to_block
    )
