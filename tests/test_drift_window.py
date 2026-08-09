"""The drift report has to bin the window it actually has (design.md §7.3).

`performance_by_month` defaults to 30 day bins, and the test partition is 27
days long. Left at the default it returns a single row, and a one point series
plotted as a trend line is worse than no figure: it invites a reading of drift
where none was measured. `fraudq.analysis` bins by seven days instead, and calls
them weeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudq.analysis import DRIFT_BIN_DAYS
from fraudq.evaluate.drift import performance_by_month

FIRST_TEST_DAY, LAST_TEST_DAY = 156, 182


@pytest.fixture
def scored_test_window() -> pd.DataFrame:
    """Both classes on every day, over the real span of the test partition."""
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange(FIRST_TEST_DAY, LAST_TEST_DAY + 1), 40)
    p = rng.beta(0.7, 5.0, size=len(days))
    return pd.DataFrame({
        "day": days,
        "p": p,
        "isFraud": rng.binomial(1, np.clip(p * 3.0, 0.0, 1.0)),
    })


def test_the_default_bin_collapses_the_window_to_one_point(scored_test_window):
    """The reason the driver does not use the default. If this ever stops being
    true the bin size deserves revisiting rather than silently keeping seven."""
    assert len(performance_by_month(scored_test_window)) == 1


def test_seven_day_bins_give_four_points(scored_test_window):
    drift = performance_by_month(scored_test_window, days_per_month=DRIFT_BIN_DAYS)
    span = LAST_TEST_DAY - FIRST_TEST_DAY + 1
    assert len(drift) == -(-span // DRIFT_BIN_DAYS) == 4


def test_columns_and_totals(scored_test_window):
    drift = performance_by_month(scored_test_window, days_per_month=DRIFT_BIN_DAYS)
    assert list(drift.columns) == ["month", "n", "fraud_rate", "pr_auc"]
    assert drift["n"].sum() == len(scored_test_window)
    assert drift["pr_auc"].between(0.0, 1.0).all()


def test_pr_auc_is_nan_for_a_single_class_bin(scored_test_window):
    """Average precision is undefined without both classes, and a bin that is all
    legitimate must report that rather than a number."""
    df = scored_test_window.copy()
    first_bin = df["day"] < FIRST_TEST_DAY + DRIFT_BIN_DAYS
    df.loc[first_bin, "isFraud"] = 0

    drift = performance_by_month(df, days_per_month=DRIFT_BIN_DAYS)
    assert np.isnan(drift.loc[0, "pr_auc"])
    assert drift.loc[1:, "pr_auc"].notna().all()


def test_bins_are_relative_to_the_first_day_present(scored_test_window):
    """Day numbers are absolute in the data, so the binning has to be offset by
    the first day of the window and not by day zero of the dataset."""
    drift = performance_by_month(scored_test_window, days_per_month=DRIFT_BIN_DAYS)
    assert drift["month"].tolist() == [0, 1, 2, 3]
