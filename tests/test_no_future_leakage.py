"""The leakage test: the executable form of invariant 3 (design.md §9).

## The idea (design.md §5.3)

If a feature is honest, meaning it only looks backwards, computing it over the
FULL history or over the history TRUNCATED at a time `t` must give EXACTLY the
same result for every row before `t`. If the two disagree, some feature is
looking at the present or the future.

This test was written and committed BEFORE `features/build.py`. Run against
naive features it **fails**, and that failure is the one that saves the
project. It passes only once `build.py` computes everything strictly backwards.

The contract of `build_features` (see features/build.py): it returns a
DataFrame ordered by (TransactionDT, TransactionID) with a reset index, holding
the identity columns plus `FEATURE_COLUMNS`.
"""

from __future__ import annotations

import pandas as pd

from fraudq.features.build import FEATURE_COLUMNS, build_features


def test_features_do_not_look_forward(sample_df: pd.DataFrame) -> None:
    """An honest feature is invariant under temporal truncation."""
    cutoff = sample_df["TransactionDT"].quantile(0.6)

    # Features over the FULL history and over the history truncated at `cutoff`.
    full = build_features(sample_df)
    truncated = build_features(sample_df[sample_df["TransactionDT"] <= cutoff])

    # `full` is ordered by (TransactionDT, TransactionID). Keep the rows of
    # `full` at or before the cut, to compare them against `truncated`.
    full_past = full[full["TransactionDT"] <= cutoff].reset_index(drop=True)

    # `TransactionID` is compared alongside the features: if the ordering ever
    # broke, the identity column would give it away instead of letting through
    # a comparison of different rows that happened to agree. Both tables come
    # ordered by (TransactionDT, TransactionID) with a reset index, so they are
    # already aligned row by row.
    cols = ["TransactionID", *FEATURE_COLUMNS]

    # NULL == NULL counts as equality, which is exactly what
    # `assert_frame_equal` does with NaN. Here a null is not missing data: it is
    # the CORRECT value of a feature with no prior history, either a uid's first
    # transaction or a row with no uid. Treating it as unequal would fail the
    # test for the very behaviour the contract demands.
    #
    # `check_dtype=False` because truncating can flip a column's inferred type
    # from integer to float as soon as a null appears, without any value moving.
    pd.testing.assert_frame_equal(full_past[cols], truncated[cols], check_dtype=False)


def test_first_txn_per_uid_has_no_history(sample_df: pd.DataFrame) -> None:
    """The FIRST transaction of each UID has no past to look at.

    A second, finer guard: if a backward-looking feature had history on a UID's
    first transaction, it would be inventing a past, or looking at the current
    row. Expected: `uid_txn_count_prior == 0`, and the means, ratios and
    z-score are NULL on that first row.
    """
    feats = build_features(sample_df)
    first_per_uid = feats.sort_values(["TransactionDT", "TransactionID"]).groupby("uid").head(1)

    assert not first_per_uid.empty, "the fixture must carry at least one uid"

    # Counting backwards from the first transaction gives zero, not null: there
    # is an answer and it is "none". A NaN here would hide the difference
    # between "has no past" and "could not be computed".
    assert (first_per_uid["uid_txn_count_prior"] == 0).all()

    # Aggregations over an empty set are NULL: there is no mean of zero
    # amounts, nor any ratio against it.
    assert first_per_uid["uid_amt_prior_mean"].isna().all()
    assert first_per_uid["uid_amt_ratio"].isna().all()

    # `uid_seconds_since_last` comes from LAG over the uid partition: on the
    # first row there is no previous one, so NULL. A 0 would be worse than
    # useless, asserting that the previous transaction happened at that instant.
    assert first_per_uid["uid_seconds_since_last"].isna().all()

    # `uid_amt_zscore` divides by STDDEV_SAMP of the prior history, which needs
    # two observations: NULL on the first transaction and on the second too.
    # That is documented in build.py, and is correct rather than a defect.
    assert first_per_uid["uid_amt_zscore"].isna().all()
