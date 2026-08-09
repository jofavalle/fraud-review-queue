"""The four comparison policies (design.md §7.1).

| # | Policy             | What it stands for                              |
|---|--------------------|-------------------------------------------------|
| 1 | approve_all        | The baseline loss. Doing nothing.               |
| 2 | single_threshold   | The naive system: block if p >= t.              |
| 3 | topk_by_score      | What most people do: review the top-K by p.     |
| 4 | topk_by_value      | The policy proposed here: top-K by V.           |

## The design decision worth being able to defend

Policies 3 and 4 share the SAME automatic rule for whatever is not reviewed:
the action of lowest expected cost. That way the only variable between them is
THE RANKING of the queue, which is the thesis. Had policy 3 also been handed a
dumb automatic rule, the difference would mix two effects and the number in the
README would be inflated. Choosing the conservative comparison is what makes it
credible.

The threshold of policy 2 is fitted on CALIB, never on test; §4 gives that
partition the job of fixing thresholds. `compare_policies` receives the
threshold already fixed.

## The single-look protocol

`compare_policies(parts["test"], ...)` runs ONCE, with the base parameters from
config.py, and the result is written to disk (reports/policy_comparison.csv and
reports/scored_test.parquet). The tornado and the Streamlit app REUSE that
persisted scoring: they vary the COST parameters over predictions already made,
and never look at test again to decide anything about the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudq.policy.allocate import allocate_day
from fraudq.policy.costs import cost_approve, cost_block, realized_cost
from fraudq.policy.simulate import QueueResult, simulate_queue

POLICY_ORDER = ("approve_all", "single_threshold", "topk_by_score", "topk_by_value")


# ---------------------------------------------------------------- the policies


def actions_approve_all(p, amt, capacity, cfg) -> np.ndarray:
    """Policy 1: approve everything. The baseline loss everything measures against."""
    return np.full(len(np.asarray(p)), "approve", dtype=object)


def make_actions_single_threshold(threshold: float):
    """Policy 2: block if p >= t, approve otherwise. No review queue.

    This is the "one threshold on the score" system of §2.2, the structurally
    wrong policy, because the optimal threshold depends on the amount.
    """

    def actions_single_threshold(p, amt, capacity, cfg) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return np.where(p >= threshold, "block", "approve").astype(object)

    return actions_single_threshold


def actions_topk_by_score(p, amt, capacity, cfg) -> np.ndarray:
    """Policy 3: review the top-K by SCORE; the rest get the best auto action.

    The naive implementation of the queue: it sends analysts to p > 0.9, where
    the answer is already certain and a human adds no information (§2.4). It
    does not filter by V > 0, because the naive policy does not know what V is.
    Spending capacity on already-decided cases IS its defect, and it is
    simulated exactly as such.
    """
    p = np.asarray(p, dtype=float)
    amt = np.asarray(amt, dtype=float)
    order = np.argsort(-p, kind="stable")  # deterministic on ties
    to_review = order[: max(int(capacity), 0)]

    approve_cheaper = cost_approve(p, amt, cfg) <= cost_block(p, amt, cfg)
    actions = np.where(approve_cheaper, "approve", "block").astype(object)
    actions[to_review] = "review"
    return actions


def actions_topk_by_value(p, amt, capacity, cfg) -> np.ndarray:
    """Policy 4: the proposed one. It delegates to allocate_day, top-K by V > 0."""
    return allocate_day(p, amt, capacity, cfg)


# ------------------------------------------- fitting the threshold (on CALIB)


def fit_single_threshold(
    df_calib: pd.DataFrame,
    cfg,
    p_col: str = "p",
    amt_col: str = "TransactionAmt",
    target: str = "isFraud",
    grid: np.ndarray | None = None,
) -> float:
    """Pick policy 2's threshold by minimising the REALISED cost on calib.

    On calib the labels are legitimately usable: the partition exists to fix
    thresholds and explore K. There is no per-day loop, because policy 2 has no
    capacity, so the day does not matter. On ties, the lowest threshold in the
    grid wins, being first in order.
    """
    p = df_calib[p_col].to_numpy(dtype=float)
    amt = df_calib[amt_col].to_numpy(dtype=float)
    y = df_calib[target].to_numpy()
    if grid is None:
        grid = np.linspace(0.005, 0.995, 199)

    costs = np.empty(len(grid))
    for i, t in enumerate(grid):
        actions = np.where(p >= t, "block", "approve").astype(object)
        costs[i] = float(np.sum(realized_cost(actions, y, amt, cfg)))
    return float(grid[int(np.argmin(costs))])


# ------------------------------------------------------------ the comparison


def compare_policies(
    df: pd.DataFrame,
    cfg,
    capacity_pct: float,
    threshold: float,
    p_col: str = "p",
    amt_col: str = "TransactionAmt",
    day_col: str = "day",
    target: str = "isFraud",
) -> pd.DataFrame:
    """Run the four policies over `df` and return the README table (§7.1).

    One row per policy: total cost, cost per $1,000 (THE metric), frauds caught
    and missed, legitimate transactions blocked, reviews and utilisation.
    """
    policies = {
        "approve_all": actions_approve_all,
        "single_threshold": make_actions_single_threshold(threshold),
        "topk_by_score": actions_topk_by_score,
        "topk_by_value": actions_topk_by_value,
    }
    rows = {}
    for name in POLICY_ORDER:
        result: QueueResult = simulate_queue(
            df,
            policies[name],
            cfg,
            capacity_pct,
            p_col=p_col,
            amt_col=amt_col,
            day_col=day_col,
            target=target,
        )
        rows[name] = result.summary()

    table = pd.DataFrame(rows).T.loc[list(POLICY_ORDER)]
    table.index.name = "policy"
    return table


def headline_savings(comparison: pd.DataFrame) -> dict:
    """THE number in the README: policy 4 against policy 3 (design.md §7.1).

    What an organisation that ranks its queue by score leaves on the table. A
    positive value means the proposed policy wins. If it comes out small, the
    parameters are NOT touched: the thesis changes, not the assumptions.
    """
    cost3 = comparison.loc["topk_by_score", "total_cost"]
    cost4 = comparison.loc["topk_by_value", "total_cost"]
    per1k_3 = comparison.loc["topk_by_score", "cost_per_1k"]
    per1k_4 = comparison.loc["topk_by_value", "cost_per_1k"]
    return {
        "savings_total": float(cost3 - cost4),
        "savings_per_1k": float(per1k_3 - per1k_4),
    }
