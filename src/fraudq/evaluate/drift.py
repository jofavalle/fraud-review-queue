"""Drift: PSI by month, and performance degradation.

The fraud distribution moves (design.md §7.3). Two cheap and very realistic
diagnostics:

1. **PSI** (Population Stability Index) per feature and per month: how much did
   the DISTRIBUTION of the inputs move away from train?
2. **Degradation by month**: PR-AUC month by month over test. How much does the
   MODEL lose as time passes?

Read together they give the sentence in the README: performance falls X %
between the first and the last window of test, which suggests a retraining
cadence of N weeks.

The convention for a "month": blocks of 30 days RELATIVE to the start of each
partition, since `day` is an offset and not a calendar (§3.3). The rule of
thumb for PSI, from industry rather than any theorem: below 0.10 is stable,
0.10 to 0.25 is moderate movement, above 0.25 is serious drift.

This module imports nothing from policy/: it measures the model and the data,
not the policy. sklearn is imported inside the function that uses it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Smoothing for empty proportions: it avoids log(0) and division by zero. The
# exact value does not matter as long as it is small and CONSTANT across
# comparisons.
_EPS = 1e-6

PSI_THRESHOLDS = {"stable": 0.10, "moderate": 0.25}


def psi(expected, actual, n_bins: int = 10) -> float:
    """PSI of `actual` against `expected`, the reference, typically train.

    Bins come from QUANTILES of the reference, so every bin of `expected`
    weighs about 1/n and no arbitrary bin of a skewed distribution dominates
    the PSI. NaN is excluded on both sides; if the null rate drifts too, that
    shows up separately in the counts, and LightGBM consumes them natively
    anyway.

    PSI = sum( (a_i - e_i) * ln(a_i / e_i) ) over the per-bin proportions.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    # Quantiles of the reference, with unique edges, since discrete features
    # collapse bins.
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:  # near-constant feature: no distribution to compare
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_prop = np.clip(e_counts / e_counts.sum(), _EPS, None)
    a_prop = np.clip(a_counts / a_counts.sum(), _EPS, None)
    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))


def add_relative_month(
    df: pd.DataFrame, day_col: str = "day", days_per_month: int = 30
) -> pd.DataFrame:
    """A copy of `df` with a `month` column: 30-day blocks from ITS own start."""
    out = df.copy()
    out["month"] = (out[day_col] - out[day_col].min()) // days_per_month
    return out


def psi_by_month(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    features: list[str],
    day_col: str = "day",
    days_per_month: int = 30,
    n_bins: int = 10,
) -> pd.DataFrame:
    """PSI per feature, month by month of `df_current` against ALL `df_reference`.

    The reference is the whole of train, which is what the model learned, and
    every month of test is compared against it. Returns a wide DataFrame: rows
    are months, columns are features, values are PSI. Ready for a heatmap in
    the notebook.
    """
    missing = [c for c in features if c not in df_reference.columns or c not in df_current.columns]
    if missing:
        raise KeyError(f"Features absent from the reference or the current frame: {missing}.")

    current = add_relative_month(df_current, day_col, days_per_month)
    rows = {}
    for month, g in current.groupby("month", sort=True):
        rows[int(month)] = {
            f: psi(df_reference[f].to_numpy(), g[f].to_numpy(), n_bins=n_bins) for f in features
        }
    out = pd.DataFrame(rows).T
    out.index.name = "month"
    return out


def performance_by_month(
    df: pd.DataFrame,
    p_col: str = "p",
    target: str = "isFraud",
    day_col: str = "day",
    days_per_month: int = 30,
) -> pd.DataFrame:
    """PR-AUC and fraud rate by relative month of test (design.md §7.3).

    It consumes the PERSISTED scoring (reports/scored_test.parquet). It
    rescores nothing and does not look at test again to take any decision: it
    describes, window by window, the evaluation that already happened. The
    percentage change between the first and the last is the retraining-cadence
    sentence of the README.
    """
    from sklearn.metrics import average_precision_score

    monthly = add_relative_month(df, day_col, days_per_month)
    rows = []
    for month, g in monthly.groupby("month", sort=True):
        y = g[target].to_numpy()
        rows.append(
            {
                "month": int(month),
                "n": len(g),
                "fraud_rate": float(y.mean()),
                "pr_auc": float(average_precision_score(y, g[p_col].to_numpy()))
                if 0 < y.sum() < len(g)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)
