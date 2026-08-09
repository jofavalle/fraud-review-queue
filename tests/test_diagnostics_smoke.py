"""The diagnostics driver runs end to end, and its two abort conditions work.

Same purpose as `test_pipeline_smoke.py` and the same caveat: numbers out of an
invented dataset say nothing about fraud. What is verified is that the chain
does not break and that the two guards which protect the real run actually fire.

Those two guards are the reason this file exists rather than just the unit tests:

1. **The rebuilt test scores must equal the persisted ones.** The real run
   recomputes 412 features to get PSI and correlation, and if that reconstruction
   drifted from what produced `reports/scored_test.parquet`, every diagnostic
   would be describing a different model while looking perfectly healthy.
2. **The retrained folds must choose the trees the booster already has.** That is
   what entitles the learning curve to be presented as the overtraining of the
   published model rather than of a lookalike.

Both are cheap to check here and expensive to discover after a 14 minute run.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fraudq.config import CONFIG
from fraudq.data.synthetic import SYNTHETIC_SPLIT, make_synthetic_transactions
from fraudq.diagnostics import run_diagnostics
from fraudq.pipeline import run_pipeline

#: Every report the notebook and `docs/report.md` read. `scored_train.parquet` is
#: among them because the score-distribution figure needs all three partitions and
#: the pipeline leaves only two behind.
_EXPECTED_REPORTS = (
    "feature_importance.csv",
    "scored_train.parquet",
    "roc_curve.csv",
    "pr_curve.csv",
    "operating_points.csv",
    "score_ks.csv",
    "feature_correlation.csv",
    "v_null_blocks.csv",
    "psi_by_week.csv",
    "learning_curve.csv",
    "diagnostics_summary.json",
)


@pytest.fixture(scope="module")
def diagnostics_run(tmp_path_factory):
    """Pipeline first, diagnostics on its output. One shared run: it is the cost."""
    out = tmp_path_factory.mktemp("diagnostics")
    cfg = replace(CONFIG, split=replace(CONFIG.split, **SYNTHETIC_SPLIT))
    df = make_synthetic_transactions(n_days=70, txns_per_day=200, seed=7)

    run_pipeline(
        df,
        cfg,
        reports_dir=out / "reports",
        models_dir=out / "models" / "artifacts",
        valid_len=10,
        holdout_days=3,
    )
    summary = run_diagnostics(
        out / "reports",
        out / "models" / "artifacts",
        cfg,
        df_raw=make_synthetic_transactions(n_days=70, txns_per_day=200, seed=7),
        learning_curve=True,
        valid_len=10,
        top_n=8,
    )
    return summary, out, cfg


def test_every_report_is_written(diagnostics_run):
    _, out, _ = diagnostics_run
    for name in _EXPECTED_REPORTS:
        assert (out / "reports" / name).exists(), name


def test_the_rebuilt_scoring_reproduces_the_persisted_one(diagnostics_run):
    """Guard 1. Exact and not approximate: the same booster over the same features
    is deterministic, so anything above zero here is a real divergence."""
    summary, _, _ = diagnostics_run
    assert summary["score_reproduction_max_abs_diff"] == 0.0


def test_the_retrained_folds_recover_the_published_tree_count(diagnostics_run):
    """Guard 2. `run_learning_curve` raises SystemExit when these disagree; this
    asserts the healthy branch, so a change that made them differ would be caught
    by the failure of the fixture itself."""
    import lightgbm as lgb

    summary, out, _ = diagnostics_run
    booster = lgb.Booster(model_file=str(out / "models" / "artifacts" / "model.txt"))
    assert summary["n_estimators"] == booster.num_trees()


def test_the_diagnostics_do_not_touch_the_pipeline_outputs(diagnostics_run):
    """Nothing published may move. The driver only ever writes new files, and the
    real run depends on that: `reports/` holds output that costs 14.6 minutes."""
    _, out, _ = diagnostics_run
    reports = out / "reports"
    for name in ("scored_calib.parquet", "scored_test.parquet", "policy_comparison.csv"):
        assert (reports / name).exists()

    scored = pd.read_parquet(reports / "scored_test.parquet")
    assert {"TransactionID", "day", "TransactionAmt", "isFraud", "score_raw", "p"} <= set(
        scored.columns
    )


def test_the_three_operating_points_share_one_budget(diagnostics_run):
    _, out, _ = diagnostics_run
    points = pd.read_csv(out / "reports" / "operating_points.csv").set_index("queue")
    assert list(points.index) == [
        "global_topk_by_score",
        "daily_topk_by_score",
        "daily_topk_by_value",
    ]
    assert points["capacity"].nunique() == 1


def test_the_learning_curve_covers_every_fold(diagnostics_run):
    _, out, cfg = diagnostics_run
    curve = pd.read_csv(out / "reports" / "learning_curve.csv")
    assert curve["fold"].nunique() == cfg.split.n_cv_folds
    assert np.allclose(curve["gap"], curve["fit_ap"] - curve["valid_ap"])


def test_the_summary_carries_what_the_report_quotes(diagnostics_run):
    summary, _, _ = diagnostics_run
    for key in (
        "features_with_gain",
        "top_feature",
        "queue_capacity",
        "ks_calib_vs_test",
        "v_null_blocks",
        "psi_max",
        "overtraining_gap_mean",
    ):
        assert key in summary, key


def test_the_cheap_phases_run_without_anything_to_rebuild_from(tmp_path):
    """The `--cheap-only` path: phases 1 to 3 need only `reports/` and the
    artefacts, which is what makes them usable before committing to a full run."""
    cfg = replace(CONFIG, split=replace(CONFIG.split, **SYNTHETIC_SPLIT))
    run_pipeline(
        make_synthetic_transactions(n_days=70, txns_per_day=200, seed=7),
        cfg,
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
        valid_len=10,
        holdout_days=3,
    )
    summary = run_diagnostics(tmp_path / "reports", tmp_path / "models", cfg, df_raw=None)

    assert "features_with_gain" in summary
    assert "psi_max" not in summary
    assert not (tmp_path / "reports" / "psi_by_week.csv").exists()
