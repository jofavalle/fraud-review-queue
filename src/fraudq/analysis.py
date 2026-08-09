"""Analysis driver: from the persisted scoring to the sensitivity and drift reports.

    python -m fraudq.analysis

Reads what `fraudq.pipeline` left on disk and writes:

    reports/sensitivity_tornado.csv   the one at a time sweep over the cost assumptions
    reports/drift_by_week.csv         PR-AUC and fraud rate across the test window
    reports/analysis_summary.json     the figures the README quotes

Nothing here re-scores anything and nothing here looks at the test partition to
decide anything. Predictions were made once and persisted; varying a cost
parameter moves only the decision layer on top of them, which is what keeps the
single-look rule true in practice (`docs/design.md` §9, invariant 5).

The threshold is the one subtlety. `run_pipeline` fits it on calib and prints it
but does not persist it, so it is refitted here from `reports/scored_calib.parquet`
with the same deterministic grid, which reproduces it exactly. It is refitted per
sweep point too: the optimal single threshold depends on the cost parameters, so
holding it fixed while sweeping them would leave policy 2 inconsistent with the
config it is being compared under. It does not move the tornado, which compares
policies 3 and 4 only, but it keeps every row of every sweep point coherent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraudq.config import CONFIG, FIGURES_DIR, SENSITIVITY_RANGES
from fraudq.evaluate.drift import performance_by_month
from fraudq.evaluate.policies import compare_policies, fit_single_threshold
from fraudq.evaluate.sensitivity import savings_per_1k, tornado_data

REPORTS_DIR = FIGURES_DIR.parent

#: The test window is 27 days and `performance_by_month` bins by 30, which
#: collapses it into a single point and makes the figure meaningless. Seven day
#: bins give four points. They are weeks, and every output here says so.
DRIFT_BIN_DAYS = 7

#: Field names to the symbols the design document uses. Lives here so the README
#: and the notebook label the tornado the same way instead of each inventing it.
PARAM_LABELS = {
    "chargeback_fee": "F, chargeback fee",
    "gross_margin": "m, gross margin",
    "friction_cost": "phi, friction cost",
    "review_cost": "r, review cost",
}


def make_evaluate_fn(scored_calib: pd.DataFrame, scored_test: pd.DataFrame, capacity_pct: float):
    """Build the `callable(cost_cfg) -> DataFrame` that `tornado_data` sweeps with."""

    def evaluate(cost_cfg):
        threshold = fit_single_threshold(scored_calib, cost_cfg)
        return compare_policies(scored_test, cost_cfg, capacity_pct, threshold)

    return evaluate


def conclusion_survives(tornado: pd.DataFrame) -> bool:
    """Whether the value-ranked queue still wins at every end of every range.

    This is the question §7.2 asks first. If it is False the finding is
    conditional on the assumptions, and the README has to say so rather than
    quiet it down.
    """
    return bool((tornado[["savings_at_low", "savings_at_high"]] > 0).all().all())


def _load(reports_dir: Path, name: str) -> pd.DataFrame:
    path = reports_dir / name
    if not path.exists():
        raise SystemExit(
            f"Cannot find {path}.\n"
            "Run the pipeline first:  python -m fraudq.pipeline\n"
            "Or without real data:    python -m fraudq.pipeline --synthetic"
        )
    return pd.read_parquet(path)


def run_analysis(reports_dir: Path) -> dict:
    """Run the sweep and the drift report, write both, and return the headline figures."""
    scored_calib = _load(reports_dir, "scored_calib.parquet")
    scored_test = _load(reports_dir, "scored_test.parquet")
    capacity_pct = CONFIG.policy.daily_capacity_pct

    print("[1/2] Cost sensitivity, one parameter at a time")
    evaluate = make_evaluate_fn(scored_calib, scored_test, capacity_pct)
    threshold = fit_single_threshold(scored_calib, CONFIG.cost)
    base_savings = savings_per_1k(evaluate(CONFIG.cost))
    tornado = tornado_data(CONFIG.cost, SENSITIVITY_RANGES, evaluate)
    tornado.to_csv(reports_dir / "sensitivity_tornado.csv", index=False)
    print(f"  Threshold refitted on calib: {threshold:.4f}")
    print(f"  Savings at base assumptions: {base_savings:.4f} per $1,000")
    print(tornado.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"[2/2] Drift across the test window, {DRIFT_BIN_DAYS} day bins")
    drift = performance_by_month(scored_test, days_per_month=DRIFT_BIN_DAYS)
    drift = drift.rename(columns={"month": "week"})
    drift.to_csv(reports_dir / "drift_by_week.csv", index=False)
    print(drift.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    summary = {
        "threshold": float(threshold),
        "capacity_pct": float(capacity_pct),
        "base_savings_per_1k": float(base_savings),
        "conclusion_survives_range": conclusion_survives(tornado),
        "dominant_param": str(tornado.iloc[0]["param"]),
        "drift_bin_days": DRIFT_BIN_DAYS,
    }
    (reports_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  {summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    run_analysis(args.reports_dir)


if __name__ == "__main__":
    main()
