"""Day-by-day simulation of the queue.

A mechanical harness: it groups by day, works out that day's capacity, hands
the decision to a policy (`choose_actions`) and settles the realised cost with
the functions in costs.py. Allocation is PER DAY because analyst capacity
renews each day (design.md §2.3); a global queue would be spending today the
analysts of last week.

The intellectual kernel is not here, it is in costs.py and allocate.py. This
module does not know what fraud is. It only keeps the books.

## The policy interface

    choose_actions(p, amt, capacity, cfg) -> np.ndarray of
    "approve" | "review" | "block"

`allocate_day` satisfies that signature, and so do the comparison policies in
evaluate/policies.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudq.policy.costs import realized_cost

_VALID_ACTIONS = frozenset({"approve", "review", "block"})


def daily_capacity(n_transactions: int, capacity_pct: float) -> int:
    """One day's review quota: floor(pct * that day's volume).

    `capacity_pct` comes from PolicyConfig (0.005 to 0.02 in design.md §2.3).
    floor and not round: capacity is real people, and half an analyst does not
    exist. It can come out 0 on small days with a low pct, which is correct
    rather than a bug.
    """
    if not 0.0 <= capacity_pct <= 1.0:
        raise ValueError(f"capacity_pct outside [0,1]: {capacity_pct}.")
    return int(np.floor(n_transactions * capacity_pct))


@dataclass
class QueueResult:
    """The result of simulating one policy over a whole partition."""

    actions: pd.Series  # action per transaction, aligned to the input df
    per_day: pd.DataFrame  # one row per day: costs, counts, capacity

    # ------------------------------------------------------------ aggregates
    @property
    def total_cost(self) -> float:
        return float(self.per_day["cost"].sum())

    @property
    def total_volume(self) -> float:
        return float(self.per_day["volume"].sum())

    @property
    def cost_per_1k(self) -> float:
        """THE headline metric: loss per $1,000 of transacted volume."""
        return self.total_cost / self.total_volume * 1_000.0

    @property
    def frauds_caught(self) -> int:
        return int(self.per_day["frauds_caught"].sum())

    @property
    def frauds_missed(self) -> int:
        return int(self.per_day["frauds_missed"].sum())

    @property
    def legit_blocked(self) -> int:
        return int(self.per_day["legit_blocked"].sum())

    @property
    def reviews(self) -> int:
        return int(self.per_day["reviews"].sum())

    @property
    def capacity(self) -> int:
        return int(self.per_day["capacity"].sum())

    @property
    def utilization(self) -> float:
        """Reviews used over total capacity. NaN if capacity was 0."""
        return self.reviews / self.capacity if self.capacity else float("nan")

    def summary(self) -> dict:
        return {
            "total_cost": self.total_cost,
            "cost_per_1k": self.cost_per_1k,
            "frauds_caught": self.frauds_caught,
            "frauds_missed": self.frauds_missed,
            "legit_blocked": self.legit_blocked,
            "reviews": self.reviews,
            "capacity": self.capacity,
            "utilization": self.utilization,
        }


def simulate_queue(
    df: pd.DataFrame,
    choose_actions,
    cfg,
    capacity_pct: float,
    p_col: str = "p",
    amt_col: str = "TransactionAmt",
    day_col: str = "day",
    target: str = "isFraud",
) -> QueueResult:
    """Run a policy day by day over `df` and settle the realised cost.

    `df` needs `p_col` (a CALIBRATED probability), `amt_col`, `day_col` and
    `target`. The days are walked in order, and each day's capacity is
    `daily_capacity(n_that_day, capacity_pct)`.
    """
    missing = [c for c in (p_col, amt_col, day_col, target) if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing for the simulation: {missing}.")

    p_all = df[p_col].to_numpy(dtype=float)
    if np.any(np.isnan(p_all)) or p_all.min() < 0.0 or p_all.max() > 1.0:
        # Invalid p means invalid V means the whole policy is fiction. Better
        # to die here.
        raise ValueError(
            f"'{p_col}' is not a valid probability (range "
            f"[{np.nanmin(p_all):.4f}, {np.nanmax(p_all):.4f}], "
            f"NaN={int(np.isnan(p_all).sum())}). Was it calibrated?"
        )

    actions_all = pd.Series(index=df.index, dtype=object)
    day_rows = []

    for day_value, g in df.groupby(day_col, sort=True):
        p = g[p_col].to_numpy(dtype=float)
        amt = g[amt_col].to_numpy(dtype=float)
        y = g[target].to_numpy()
        capacity = daily_capacity(len(g), capacity_pct)

        actions = np.asarray(choose_actions(p, amt, capacity, cfg))
        if actions.shape != p.shape:
            raise ValueError(
                f"The policy returned {actions.shape} actions for "
                f"{p.shape} transactions (day {day_value})."
            )
        bad = set(np.unique(actions)) - _VALID_ACTIONS
        if bad:
            raise ValueError(f"Unknown actions from the policy: {sorted(bad)}.")

        costs = np.asarray(realized_cost(actions, y, amt, cfg), dtype=float)
        is_fraud = y == 1
        reviewed = actions == "review"
        blocked = actions == "block"
        approved = actions == "approve"

        day_rows.append(
            {
                "day": day_value,
                "n": len(g),
                "capacity": capacity,
                "reviews": int(reviewed.sum()),
                "cost": float(costs.sum()),
                "volume": float(amt.sum()),
                # the review resolves the case correctly (§2.2): a reviewed
                # fraud counts as caught.
                "frauds_caught": int(((blocked | reviewed) & is_fraud).sum()),
                "frauds_missed": int((approved & is_fraud).sum()),
                "legit_blocked": int((blocked & ~is_fraud).sum()),
            }
        )
        actions_all.loc[g.index] = actions

    return QueueResult(actions=actions_all, per_day=pd.DataFrame(day_rows))
