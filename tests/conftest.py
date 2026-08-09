"""Shared test fixtures.

`sample_df` is a small synthetic dataset, built deliberately for the leakage
test: several UIDs with several transactions each, spread over time, **one
`TransactionDT` tie** inside a UID (which checks that the `TransactionID`
tie-break makes the computation deterministic) and **one row with a null
`addr1`** (which checks that a transaction with no identity neither receives
nor manufactures history).
"""

from __future__ import annotations

import pandas as pd
import pytest

SECONDS_PER_DAY = 86_400


def _row(txn_id: int, dt: int, card1: int, addr1: int, d1n: int, amt: float, fraud: int) -> dict:
    """Build a row with `day` and `D1` derived from a constant target `d1n`.

    `D1 = day - d1n` is chosen so that `D1n = day - D1 == d1n` stays constant
    per customer, which is exactly the UID assumption (uid.py, design.md §5.2).
    """
    day = dt // SECONDS_PER_DAY
    return {
        "TransactionID": txn_id,
        "TransactionDT": dt,
        "day": day,
        "card1": card1,
        "addr1": addr1,
        "D1": day - d1n,
        "TransactionAmt": amt,
        "isFraud": fraud,  # present, but the features must NEVER use it
    }


@pytest.fixture
def sample_df() -> pd.DataFrame:
    # Customer A: card1=1000, addr1=50, D1n=5. Four transactions, two of which
    # share a TransactionDT, a deliberate tie, to test determinism.
    # Customer B: card1=2000, addr1=60, D1n=8. Three transactions.
    rows = [
        _row(1, 1 * SECONDS_PER_DAY + 100, 1000, 50, 5, 50.0, 0),
        _row(2, 3 * SECONDS_PER_DAY + 200, 1000, 50, 5, 70.0, 0),
        _row(3, 3 * SECONDS_PER_DAY + 200, 1000, 50, 5, 30.0, 1),  # DT tie with id=2
        _row(4, 9 * SECONDS_PER_DAY + 500, 1000, 50, 5, 120.0, 0),
        _row(5, 2 * SECONDS_PER_DAY + 300, 2000, 60, 8, 200.0, 0),
        _row(6, 6 * SECONDS_PER_DAY + 100, 2000, 60, 8, 210.0, 0),
        _row(7, 8 * SECONDS_PER_DAY + 900, 2000, 60, 8, 400.0, 1),
    ]
    # Row with no identity: null addr1 -> NULL uid -> all five features NULL.
    # (pandas' groupby("uid") drops NaN by default, so the second test never
    # sees it, and that is right: it is not "a uid's first txn", it has NO uid.)
    rows.append(_row(8, 4 * SECONDS_PER_DAY + 400, 4000, None, 3, 99.0, 0))
    # Input order deliberately shuffled, so that nothing can depend on it.
    df = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
    return df
