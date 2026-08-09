"""Allocation of the review queue under finite capacity.

One day of transactions comes in; one of three actions comes out per
transaction. Capacity renews each day, and grouping by day is the job of
`simulate.py`: what arrives here is already a single day.

The contract, assumed by `simulate.py` and by the policies compared:

1. At most ``capacity`` transactions are reviewed, and ALL of them have
   ``V > 0``. Leftover capacity goes unspent: reviewing a case with no value is
   paying ``r`` for nothing.
2. The ones reviewed are those with the highest ``V``. Greedy is optimal here
   because every review costs the same: it is a knapsack with unit weights
   (`docs/design.md` §2.3).
3. Everything else gets its cheaper automatic action. On a tie, "approve".
4. Deterministic when ``V`` ties: a stable order by position, the same
   principle as the `TransactionID` tie-break in the backward-looking windows.

The executable specification is `tests/test_allocation_respects_capacity.py`,
including the case that encodes the thesis of the project: with capacity for
one, a transaction scoring 0.95 on a small amount is blocked and one scoring
0.25 on a large amount is reviewed. Ranking by score would do the opposite.
"""

from __future__ import annotations

import numpy as np

from fraudq.policy.costs import cost_approve, cost_block, value_of_review


def allocate_day(p, amt, capacity, cfg):
    """Assign approve / review / block to one day of transactions.

    Parameters
    ----------
    p:
        CALIBRATED probabilities. Without calibration ``V`` means nothing and
        this function optimises a fiction (`docs/design.md` §4.3).
    amt:
        Amounts (``TransactionAmt``), aligned with ``p``.
    capacity:
        The day's review quota, an integer >= 0. At 0, everything is automatic.
    cfg:
        Duck-typed with ``F``, ``m``, ``phi`` and ``r``.

    Returns
    -------
    np.ndarray
        dtype object, the same length as ``p``, with values exactly "approve",
        "review" or "block".

    Raises
    ------
    ValueError
        If ``capacity`` is negative.
    """
    capacity = int(capacity)
    if capacity < 0:
        raise ValueError(f"capacity must be >= 0, got {capacity}")

    p = np.asarray(p, dtype=float)
    amt = np.asarray(amt, dtype=float)

    # The starting automatic action: the cheaper of the two, ties to "approve".
    actions = np.where(
        cost_approve(p, amt, cfg) <= cost_block(p, amt, cfg), "approve", "block"
    ).astype(object)

    if capacity == 0:
        return actions

    v = value_of_review(p, amt, cfg)

    # Stable order: on identical V the lower position wins, so two runs over
    # the same input give the same result.
    order = np.argsort(-v, kind="stable")
    to_review = order[v[order] > 0][:capacity]
    actions[to_review] = "review"

    return actions
