"""A generator of synthetic transactions shaped like the real schema.

It exists for one thing: to run the whole pipeline without having downloaded
the 590k IEEE-CIS records. It does not stand in for the data, and no number
that comes out of it is a result of this project. What it verifies is that the
chain (ingestion, features, split, training, calibration, allocation and the
policy comparison) does not break, which is not what you want to discover on
the machine where training costs hours.

It reproduces the schema properties the code depends on:

- ``TransactionDT`` in seconds from an unspecified origin, not a date.
- ``D1`` built as ``day - D1n`` with ``D1n`` constant per card, which is the
  assumption `features/uid.py` uses to reconstruct the customer.
- Fraud that is rare and amount-dependent, so there is something to learn and
  so the cost layer gets a non-degenerate case.
- Rows with no ``addr1``, because the real data has them and the UID features
  have to come out null there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400

#: A reduced split for the synthetic dataset. The production one (train through
#: day 119, calibration through 155) does not fit a dataset of a few days, and
#: forcing it would give empty partitions.
SYNTHETIC_SPLIT = {
    "train_end_day": 39,
    "embargo_days": 3,
    "calib_end_day": 59,
    "n_cv_folds": 2,
}


def make_synthetic_transactions(
    n_days: int = 75,
    # The daily volume is not cosmetic: review capacity is a percentage of the
    # day's volume, and with few transactions `int(n * 0.01)` falls to zero,
    # the queue stays empty and policies 3 and 4 degenerate into the same one.
    # The smoke test would pass without exercising the allocation, which is
    # precisely the piece that needs testing.
    txns_per_day: int = 400,
    n_cards: int = 300,
    fraud_base_rate: float = 0.035,
    missing_addr_frac: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a DataFrame with the schema the pipeline expects.

    Returns
    -------
    DataFrame with ``TransactionID``, ``TransactionDT``, ``day``, ``hour``,
    ``card1``, ``addr1``, ``D1``, ``TransactionAmt``, ``P_emaildomain``,
    ``C1``, ``C2``, ``V1`` and ``isFraud``, sorted by ``TransactionDT``.
    """
    rng = np.random.default_rng(seed)
    n = n_days * txns_per_day

    # A customer is (card1, addr1, D1n). D1n being constant per card is what
    # makes the UID reconstructible.
    card1 = rng.integers(1000, 1000 + n_cards, size=n)
    card_addr = rng.integers(50, 90, size=n_cards)
    card_d1n = rng.integers(-400, 0, size=n_cards)
    addr1 = card_addr[card1 - 1000].astype(float)
    d1n = card_d1n[card1 - 1000]

    day = np.repeat(np.arange(n_days), txns_per_day)
    second_in_day = rng.integers(0, SECONDS_PER_DAY, size=n)
    dt = day * SECONDS_PER_DAY + second_in_day

    amt = np.round(np.exp(rng.normal(3.6, 1.1, size=n)), 2)

    # Fraud that depends on the amount and on a covariate, so the model has
    # signal and the calibration has something to correct.
    v1 = rng.normal(0.0, 1.0, size=n)
    logit = (
        np.log(fraud_base_rate / (1 - fraud_base_rate))
        + 0.55 * (np.log1p(amt) - np.log1p(amt).mean())
        + 0.9 * v1
    )
    is_fraud = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))

    # Rows with no identity: their UID features must come out null.
    addr1[rng.random(n) < missing_addr_frac] = np.nan

    df = pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1),
            "TransactionDT": dt,
            "day": day,
            "hour": (dt // 3_600) % 24,
            "card1": card1,
            "addr1": addr1,
            "D1": day - d1n,
            "TransactionAmt": amt,
            "P_emaildomain": rng.choice(
                ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com"],
                size=n,
                p=[0.5, 0.25, 0.15, 0.10],
            ),
            "C1": rng.poisson(3.0, size=n).astype(float),
            "C2": rng.poisson(1.5, size=n).astype(float),
            "V1": v1,
            "isFraud": is_fraud,
        }
    )
    return df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
