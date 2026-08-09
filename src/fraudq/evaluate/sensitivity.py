"""Sensitivity of the conclusion to the cost assumptions (design.md §7.2).

The four cost parameters are assumed, not measured, so the honest question is
not what the saving is but how much of it survives the assumptions moving.

## The two rules that make this an analysis rather than a justification

1. **The ranges are read from `SENSITIVITY_RANGES` in config.py**, fixed BEFORE
   the results were seen. Choosing ranges once the number is known turns the
   tornado into marketing.
2. **The config is frozen and the variants are built with
   `dataclasses.replace`.** If the parameters did not flow through as
   arguments, the sweep would return identical results for every value without
   raising anything, and the conclusion would look more robust than it is.

## The contract of `tornado_data` (what plot_tornado and the notebook assume)

A DataFrame with one row per swept parameter:

    param     | low   | high  | savings_at_low | savings_at_high | swing
    "F"       | 10.0  | 40.0  | ...            | ...             | |high-low|

sorted by `swing` DESCENDING, since the tornado is read from the top down.
`savings_*` is the per-$1,000 saving of policy 4 over policy 3 with ALL the
other parameters at their base value.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd


def savings_per_1k(comparison: pd.DataFrame) -> float:
    """Pull the per-$1,000 saving (policy 4 over 3) out of a compare_policies table."""
    return float(
        comparison.loc["topk_by_score", "cost_per_1k"]
        - comparison.loc["topk_by_value", "cost_per_1k"]
    )


def tornado_data(base_cfg, ranges: dict, evaluate_fn) -> pd.DataFrame:
    """One-at-a-time sweep over the cost assumptions (design.md §7.2).

    Parameters
    ----------
    base_cfg:
        The frozen CostConfig, at its base values.
    ranges:
        `SENSITIVITY_RANGES` from config.py: {field_name: (low, high)}. The
        names are the dataclass FIELDS (`chargeback_fee` and so on), not the
        aliases, because `dataclasses.replace` knows nothing about properties.
    evaluate_fn:
        callable(cfg) -> the compare_policies DataFrame over the PERSISTED
        scoring (reports/scored_test.parquet). Note that varying COST
        parameters does not look at test again: the predictions are already
        made, and only the decision layer changes.

    Returns
    -------
    The DataFrame of the contract above, sorted by `swing` descending.
    """
    rows = []
    for param, (low, high) in ranges.items():
        # `replace` on a frozen dataclass returns a copy: `base_cfg` is never
        # touched, and each variant moves ONE parameter leaving the rest at
        # base. That is what makes the swing attributable to that parameter.
        cfg_low = replace(base_cfg, **{param: low})
        cfg_high = replace(base_cfg, **{param: high})

        savings_at_low = savings_per_1k(evaluate_fn(cfg_low))
        savings_at_high = savings_per_1k(evaluate_fn(cfg_high))

        rows.append(
            {
                "param": param,
                "low": low,
                "high": high,
                "savings_at_low": savings_at_low,
                "savings_at_high": savings_at_high,
                # Absolute value: what the tornado measures is how much the
                # parameter MOVES the conclusion, not in which direction. The
                # direction stays readable in the two columns above.
                "swing": abs(savings_at_high - savings_at_low),
            }
        )

    # Descending: the tornado is read from the top down, and the longest bar is
    # the parameter the business should go and measure properly.
    return (
        pd.DataFrame(
            rows, columns=["param", "low", "high", "savings_at_low", "savings_at_high", "swing"]
        )
        .sort_values("swing", ascending=False)
        .reset_index(drop=True)
    )


def sensitivity_grid_2d(
    base_cfg, param_x: str, values_x, param_y: str, values_y, evaluate_fn
) -> pd.DataFrame:
    """An optional 2D grid: the saving as a function of two parameters.

    Returns a long DataFrame (param_x, param_y, savings_per_1k) ready for a
    heatmap that colours the region where the value-ranked policy wins.
    """
    rows = [
        {
            param_x: vx,
            param_y: vy,
            "savings_per_1k": savings_per_1k(
                evaluate_fn(replace(base_cfg, **{param_x: vx, param_y: vy}))
            ),
        }
        for vx in values_x
        for vy in values_y
    ]
    return pd.DataFrame(rows, columns=[param_x, param_y, "savings_per_1k"])


def plot_tornado(
    tornado_df: pd.DataFrame, base_savings: float, ax=None, param_labels: dict | None = None
):
    """The tornado plot: horizontal bars.

    Each bar runs from `savings_at_low` to `savings_at_high`, sorted by `swing`
    with the largest on top; the vertical line is the saving under the base
    parameters. If ALL the bars live to the right of zero, the conclusion
    survives the full range of assumptions, and that sentence, as it stands, is
    the one that goes into the README.

    matplotlib is imported inside, so the module imports without it.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.6 * len(tornado_df) + 1.5))

    df = tornado_df.iloc[::-1]  # largest swing on top
    labels = [(param_labels or {}).get(row["param"], row["param"]) for _, row in df.iterrows()]
    lo = df[["savings_at_low", "savings_at_high"]].min(axis=1)
    hi = df[["savings_at_low", "savings_at_high"]].max(axis=1)

    ax.barh(labels, hi - lo, left=lo, height=0.6, alpha=0.85)
    ax.axvline(base_savings, linestyle="--", linewidth=1.2, label="base")
    ax.axvline(0.0, color="black", linewidth=0.8)
    # A single `$` in the string, on purpose: two of them delimit mathtext, so
    # matplotlib eats the symbols and italicises whatever sits between them.
    ax.set_xlabel("Savings of the value-ranked queue over the score-ranked, per $1,000")
    # Lower left: the bars live to the right of the base line, and at lower
    # right the legend overlapped the row of the smallest-swing parameter.
    ax.legend(loc="lower left")
    ax.figure.tight_layout()
    return ax
