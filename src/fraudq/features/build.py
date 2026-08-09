"""Backward-looking aggregates over the UID: the anti-leakage invariant in SQL.

## The invariant (design.md §5.3)

> **Every aggregate over the UID is strictly backward-looking.** For row `i`,
> the value is computed from **earlier transactions of the same UID only**,
> never the current row and never a future one.

In SQL that is the frame:

    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING

That `1 PRECEDING` **is** pandas' `.shift(1)`: it excludes the current row. It
is, literally, the anti-leakage invariant expressed in window functions.
`LAG(...)` plays the same role for the time gap.

## The contract (what tests/test_no_future_leakage.py assumes)

`build_features(df)`:

- Requires in `df`: `TransactionID`, `TransactionDT`, `TransactionAmt`, and the
  UID columns (`day`, `card1`, `addr1`, `D1`). It never uses `isFraud`.
- Returns a `DataFrame` **ordered by (`TransactionDT`, `TransactionID`)** with a
  reset index, whose columns are the identity ones plus the features.
- **Rows with no uid**, where some component is null, get **NULL** in all five
  features. Without an identity there is no history, and grouping the nulls
  together would manufacture history across unrelated customers.
- **The key property, the one the test checks:** for any time cut `t`, the
  features of the rows with `TransactionDT <= t` are **identical** whether they
  are computed over the full history or over the history truncated at `t`. It
  holds because every feature depends only on strictly earlier rows.

The `TransactionID` tie-break in the window's `ORDER BY` makes the computation
**deterministic** when `TransactionDT` ties. Without it, the split between rows
sharing an instant would be up to the engine and the test could flicker.

## Why DuckDB and not pandas

SQL states the invariant declaratively, runs over the `DataFrame` without
copying it into another structure, and keeps the window-function work in the
language built for it. DuckDB queries a pandas `DataFrame` directly by its
registered variable name.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from fraudq.features.uid import add_uid

# Identity columns, carried through to the output untouched.
_ID_COLS = ("TransactionID", "uid", "TransactionDT", "TransactionAmt")

# Strictly backward-looking aggregates. Every window orders by
# (TransactionDT, TransactionID) to stay deterministic; the frame excludes the
# current row with `1 PRECEDING`.
_FEATURE_SQL = """
SELECT
    TransactionID,
    uid,
    TransactionDT,
    TransactionAmt,

    -- Every feature is NULL when uid IS NULL: with no identity there is no
    -- history to look at, and the nulls are NOT grouped together, which would
    -- be manufactured history.

    -- Number of EARLIER transactions of the same uid (0 on the first).
    CASE WHEN uid IS NULL THEN NULL
         ELSE COUNT(*) OVER w_prior END AS uid_txn_count_prior,

    -- Seconds since the uid's previous transaction (NULL on the first).
    CASE WHEN uid IS NULL THEN NULL
         ELSE TransactionDT - LAG(TransactionDT) OVER w_order END AS uid_seconds_since_last,

    -- Mean of the PREVIOUS amounts (NULL on the first).
    CASE WHEN uid IS NULL THEN NULL
         ELSE AVG(TransactionAmt) OVER w_prior END AS uid_amt_prior_mean,

    -- Current amount over the previous mean. NULLIF avoids dividing by 0.
    CASE WHEN uid IS NULL THEN NULL
         ELSE TransactionAmt / NULLIF(AVG(TransactionAmt) OVER w_prior, 0)
         END AS uid_amt_ratio,

    -- z-score against the uid's PREVIOUS distribution.
    -- STDDEV_SAMP needs >= 2 earlier points, so NULL on the 1st and 2nd txn.
    CASE WHEN uid IS NULL THEN NULL
         ELSE (TransactionAmt - AVG(TransactionAmt) OVER w_prior)
              / NULLIF(STDDEV_SAMP(TransactionAmt) OVER w_prior, 0)
         END AS uid_amt_zscore

FROM t
WINDOW
    w_order AS (PARTITION BY uid ORDER BY TransactionDT, TransactionID),
    w_prior AS (PARTITION BY uid ORDER BY TransactionDT, TransactionID
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
ORDER BY TransactionDT, TransactionID
"""

FEATURE_COLUMNS = (
    "uid_txn_count_prior",
    "uid_seconds_since_last",
    "uid_amt_prior_mean",
    "uid_amt_ratio",
    "uid_amt_zscore",
)


def build_features(
    df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Compute the backward-looking UID aggregates. The contract is in the module.

    Parameters
    ----------
    df:
        Raw transactions, carrying the UID columns. Not modified in place.
    con:
        An optional DuckDB connection, to reuse one. If None, an ephemeral
        connection is opened and closed on the way out.

    Returns
    -------
    DataFrame ordered by (TransactionDT, TransactionID), index reset.
    """
    with_uid = add_uid(df)

    owns_con = con is None
    con = con or duckdb.connect()
    try:
        con.register("t", with_uid)
        out = con.execute(_FEATURE_SQL).df()
        con.unregister("t")
    finally:
        if owns_con:
            con.close()

    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Base features (design.md §5.1)
# --------------------------------------------------------------------------
#
# The UID aggregates above are the expensive, delicate part. These are the
# cheap part, and without them the model would train on the five UID columns
# alone.
#
# They are split in two for a reason that is not cosmetic: the ones in
# `add_base_features` depend on their own row only, so they are immune to
# leakage and can be computed over the whole dataset. The ones in
# `FrequencyEncoder` summarise the distribution of a column, so they are
# **fitted on train only** and applied to the rest; computing them over the
# whole dataset would let calibration and test shape the representation of the
# training rows.

BASE_FEATURE_COLUMNS = ("amt_log", "amt_decimal", "hour")

#: High-cardinality categoricals, encoded by frequency.
FREQ_ENCODED_COLS = ("card1", "addr1", "P_emaildomain", "R_emaildomain")

SECONDS_PER_HOUR = 3_600
HOURS_PER_DAY = 24


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """The §5.1 features that depend on their own row only. Nothing to fit.

    - ``amt_log``: ``log1p`` of the amount, which compresses the long tail.
    - ``amt_decimal``: the fractional part of the amount. Amounts converted
      from another currency or generated by a program leave odd signatures
      here, while a human purchase tends to end in .00 or .99.
    - ``hour``: hour of the day. The ingestion derives it; it is recomputed if
      absent, so the function also works on a raw DataFrame.

    Email domains are reduced to their base provider (``gmail.com`` ->
    ``gmail``) and left categorical: what turns them into a number is
    `FrequencyEncoder`.
    """
    out = df.copy()

    amt = out["TransactionAmt"].astype(float)
    out["amt_log"] = np.log1p(amt)
    out["amt_decimal"] = amt % 1.0

    if "hour" not in out.columns:
        out["hour"] = (out["TransactionDT"] // SECONDS_PER_HOUR) % HOURS_PER_DAY

    for col in ("P_emaildomain", "R_emaildomain"):
        if col in out.columns:
            out[col] = out[col].astype("string").str.split(".").str[0]

    return out


class FrequencyEncoder:
    """Frequency encoding of high-cardinality categoricals (§5.1).

    Fitted on train ONLY, which is what makes it legitimate: the frequency of a
    `card1` is a summary of the distribution, and computing it over the whole
    dataset would leak information from the future partitions into training.

    A category unseen in train gets frequency 0, which is the correct answer:
    at training time, that value did not exist.
    """

    def __init__(self, cols: tuple[str, ...] = FREQ_ENCODED_COLS) -> None:
        self.cols = cols
        self.freqs_: dict[str, pd.Series] = {}

    @property
    def feature_names(self) -> list[str]:
        return [f"{c}_freq" for c in self.freqs_]

    def fit(self, df_train: pd.DataFrame) -> FrequencyEncoder:
        self.freqs_ = {
            col: df_train[col].value_counts(normalize=True)
            for col in self.cols
            if col in df_train.columns
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.freqs_:
            raise RuntimeError("FrequencyEncoder is not fitted: call fit() first.")
        out = df.copy()
        for col, freq in self.freqs_.items():
            out[f"{col}_freq"] = out[col].map(freq).astype(float).fillna(0.0)
        return out
