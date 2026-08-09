"""Building the UID: entity resolution, NOT leakage.

## The decision (design.md §5.2)

IEEE-CIS carries no customer identifier, but an approximate one can be
**reconstructed**:

    D1n = day - D1

`D1` is "days since the card was first seen". Subtracting it from the current
day gives an approximation of the **card's registration date**, which is
(approximately) **constant for that card**. Combined with `card1` and `addr1`,
it identifies a customer:

    uid = card1 + '_' + addr1 + '_' + D1n

### Is this leakage? No.

- **Building the UID is *entity resolution*.** A real fraud system **does** have
  a persistent customer identifier at scoring time. Reconstructing it from
  anonymised columns recovers information that would be available in
  production. That is legitimate.
- **What WOULD be leakage** is aggregating the *target* (`isFraud`) over the UID
  across the whole dataset, since that looks at labels from the future. This
  module **never touches `isFraud`**; the backward-looking aggregates live in
  `build.py` and `tests/test_no_future_leakage.py` watches them.

It requires `df` to already carry the `day` column, which the ingestion derives
from `TransactionDT` (see data/ingest.py).
"""

from __future__ import annotations

import pandas as pd

UID_COLS = ("card1", "addr1", "D1")


def add_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with the `D1n` and `uid` columns added.

    - `D1n = day - D1`, a proxy for the card's registration date, as a nullable
      integer (`Int64`). The cast matters: if `D1` is float, because it carries
      NaN, then without it `D1n` would be '5.0' in one subset and '5' in
      another, and the uid would change with which rows happen to be present.
    - `uid` = 'card1_addr1_D1n' as a string.

    It does not modify `df` in place.

    **Nulls:** if ANY component is null, the whole `uid` comes out **NULL**,
    because pd.NA propagates through concatenation of dtype 'string'. That is
    deliberate and `build.py` honours it: the features of a row with no uid are
    NULL. The alternative, grouping the nulls under one literal uid, would be a
    serious mistake: `PARTITION BY` would gather transactions from unrelated
    customers into a single giant "identity" and the aggregates would
    manufacture false history.
    """
    missing = [c for c in ("day", *UID_COLS) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Columns missing to build the UID: {missing}. "
            "Was the ingestion, which derives `day`, run before the features?"
        )

    out = df.copy()

    # round() before Int64: these columns arrive as float, since nulls force
    # float64 in pandas, and a direct cast either fails or is unsafe against
    # floating-point artefacts. Without the normalisation, addr1=50.0 would
    # give the uid '1000_50.0_5' instead of '1000_50_5'.
    def _int_str(series: pd.Series) -> pd.Series:
        return series.round().astype("Int64").astype("string")

    out["D1n"] = (out["day"] - out["D1"]).round().astype("Int64")
    out["uid"] = (
        _int_str(out["card1"]) + "_" + _int_str(out["addr1"]) + "_" + out["D1n"].astype("string")
    )
    return out
