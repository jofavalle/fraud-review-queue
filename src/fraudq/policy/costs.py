"""The decision layer: expected cost of each action, and the value of a review.

This is where the derivations of `docs/design.md` §2 become numpy. It is the
core of the project: everything else, the allocation, the simulation, the API
and the queue simulator, consumes these five functions.

The contract, assumed by `allocate.py`, `simulate.py`, `policies.py`, the API
and the Streamlit app:

- `cfg` is duck-typed with the attributes `F`, `m`, `phi` and `r`. `CostConfig`
  in `config.py` exposes them as aliases of its fields; in the tests a
  `SimpleNamespace` is enough.
- Every function takes scalars or numpy arrays and returns the same. They are
  vectorised with `np.minimum` and `np.where`, with no loops.
- `cfg` ALWAYS arrives as an argument, never from a global. That is invariant 8
  of `docs/design.md`: a function that reaches for `CONFIG` on its own silently
  ignores the sensitivity sweep, and fails without raising anything.

The executable specification is `tests/test_cost_functions.py`, whose cases are
worked out by hand.
"""

from __future__ import annotations

import numpy as np


def cost_approve(p, amt, cfg):
    """Expected cost of APPROVING the transaction (design.md §2.2, first row).

    If it was fraud, with probability ``p``, the issuer reverses the charge: the
    full amount is lost plus the fixed chargeback fee. If it was legitimate, it
    costs nothing.

        E[cost | approve] = p * (amt + F)
    """
    return p * (amt + cfg.F)


def cost_block(p, amt, cfg):
    """Expected cost of BLOCKING the transaction (design.md §2.2, second row).

    If it was legitimate, with probability ``1 - p``, the gross margin on that
    sale is lost plus the friction cost: support, and the share of churn it
    causes. If it was fraud, blocking costs nothing: the loss was avoided.

        E[cost | block] = (1 - p) * (m * amt + phi)
    """
    return (1.0 - p) * (cfg.m * amt + cfg.phi)


def value_of_review(p, amt, cfg):
    """Value of sending the transaction to a human (design.md §2.3).

    The expected-cost reduction against the BEST automatic action, assuming the
    review resolves the case correctly at a cost of ``r``:

        V = min( cost_approve, cost_block ) - r

    It can be NEGATIVE, and that sign is the whole thesis of the project in one
    inequality: reviewing a case the system has already decided costs ``r`` and
    buys nothing. ``V`` is maximised at MODERATE probabilities, at ``p_star``,
    and grows with the amount.
    """
    return np.minimum(cost_approve(p, amt, cfg), cost_block(p, amt, cfg)) - cfg.r


def realized_cost(actions, is_fraud, amt, cfg):
    """REALISED cost of each action, given the true outcome.

    This is the accounting counterpart of the functions above: the expected
    costs decide, this one settles. `simulate.py` uses it to evaluate policies
    against real labels, on calibration to fit thresholds and on test once.

    | action    | fraud (y=1)     | legitimate (y=0)   |
    |-----------|-----------------|--------------------|
    | approve   | amt + F         | 0                  |
    | block     | 0               | m*amt + phi        |
    | review    | r               | r                  |

    A review costs ``r`` whatever happens: the assumption in §2.2 is that the
    analyst resolves the case, so neither column's loss is incurred.

    Parameters
    ----------
    actions:
        Array of strings, exactly "approve", "review" or "block".
    is_fraud:
        True label, 0 or 1, aligned with ``actions``.
    amt:
        Amounts, aligned with ``actions``.

    Returns
    -------
    Array of floats of the same length.

    Raises
    ------
    ValueError
        If an action turns up that is none of the three. A silent typo here
        would contaminate the accounting of an entire policy.
    """
    actions = np.asarray(actions, dtype=object)
    y = np.asarray(is_fraud, dtype=float)
    amt = np.asarray(amt, dtype=float)

    is_approve = actions == "approve"
    is_block = actions == "block"
    is_review = actions == "review"

    unknown = ~(is_approve | is_block | is_review)
    if np.any(unknown):
        raise ValueError(f"Unknown actions: {sorted(set(np.asarray(actions)[unknown]))}")

    return np.where(
        is_approve,
        y * (amt + cfg.F),
        np.where(is_block, (1.0 - y) * (cfg.m * amt + cfg.phi), float(cfg.r)),
    )


def p_star(amt, cfg):
    """The probability at which ``value_of_review`` is highest (design.md §2.4).

    ``V`` is the minimum of a line increasing in ``p`` (approve) and a
    decreasing one (block), minus a constant. That minimum is maximised exactly
    where the two cross:

        p * (a + F) = (1 - p) * (m*a + phi)
        p_star = (m*a + phi) / ( (a + F) + (m*a + phi) )

    It sits around 0.2 to 0.3 and depends little on the amount, which is why
    ranking the queue by score concentrates analysts where they add least. No
    production consumer needs it: the results notebook uses it to annotate the
    decision-region figure.
    """
    blocked = cfg.m * amt + cfg.phi
    return blocked / ((amt + cfg.F) + blocked)
