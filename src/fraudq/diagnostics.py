"""Diagnostics driver: from the persisted artefacts to the model reports.

    python -m fraudq.diagnostics                    # phases 1 to 5
    python -m fraudq.diagnostics --learning-curve   # and the overtraining curve
    python -m fraudq.diagnostics --cheap-only       # phases 1 to 3, in seconds

Sibling of `fraudq.analysis`, which reports on the DECISION layer. This one
reports on the MODEL underneath it, and writes:

    reports/feature_importance.csv    gain and split, per feature
    reports/roc_curve.csv             the full ROC over test
    reports/pr_curve.csv              the full precision-recall curve
    reports/operating_points.csv      where the queues sit on them
    reports/score_ks.csv              KS between partitions, by class
    reports/feature_correlation.csv   Spearman over the top features
    reports/v_null_blocks.csv         the Vesta blocks, by null pattern
    reports/psi_by_week.csv           feature drift, train as the reference
    reports/learning_curve.csv        PR-AUC per round, fit against valid
    reports/diagnostics_summary.json  the figures the report quotes

## Three phases, and why they are separated

Phases 1 to 3 read only what is already on disk and take about ten seconds.
Phases 4 and 5 rebuild the 412 features from the raw data, which is the
expensive part and the one that peaks in memory, because the persisted scoring
carries identifiers, amount, label and probability but no feature values: PSI
and correlation are impossible without recomputing them. Phase 6 retrains the
cross-validation folds and is opt-in.

**Nothing here regenerates a model.** The booster and the calibrator are loaded,
never fitted, and every output is a new file: no published number can move.

## The single-look rule

Rebuilding features and re-scoring test with the persisted booster REPRODUCES a
prediction that was already made; it decides nothing. That is the same thing
`fraudq.pipeline` already declares legitimate about being rerun over the same
data (`docs/design.md` §9, invariant 5). Phase 4 turns it into an assertion:
the reconstructed `score_raw` must equal the persisted one to floating point,
and the run aborts if it does not. If those two disagree, either the feature
pipeline is not deterministic or the artefacts are stale, and every number
downstream would be describing a model that no longer exists.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraudq.analysis import DRIFT_BIN_DAYS
from fraudq.config import CONFIG, DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, Config
from fraudq.data.split import expanding_window_folds
from fraudq.data.synthetic import SYNTHETIC_SPLIT, make_synthetic_transactions
from fraudq.evaluate.diagnostics import (
    correlation_matrix,
    importance_table,
    learning_curve_folds,
    null_pattern_blocks,
    operating_points_table,
    overtraining_summary,
    pr_points,
    roc_points,
    score_ks_table,
    top_features,
)
from fraudq.evaluate.drift import psi_by_month
from fraudq.models.persist import load_artifacts
from fraudq.models.train import predict_scores
from fraudq.pipeline import lgbm_params, load_processed, prepare

REPORTS_DIR = FIGURES_DIR.parent
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

#: How many features the correlation matrix and the PSI report cover. The
#: heatmaps stop being readable well before the 412 the model has, and the tail
#: of the gain ranking carries almost nothing: see feature_importance.csv.
TOP_N_FEATURES = 25

#: The anonymised Vesta columns, whose null patterns recover their blocks (§5.4).
V_COLUMN_PREFIX = "V"


def _load(reports_dir: Path, name: str) -> pd.DataFrame:
    path = reports_dir / name
    if not path.exists():
        raise SystemExit(
            f"Cannot find {path}.\n"
            "Run the pipeline first:  python -m fraudq.pipeline\n"
            "Or without real data:    python -m fraudq.pipeline --synthetic"
        )
    return pd.read_parquet(path)


def _write(df: pd.DataFrame, reports_dir: Path, name: str, index: bool = False) -> None:
    df.to_csv(reports_dir / name, index=index)


# ---------------------------------------------------------------------------
# Phases 1 to 3: everything the persisted artefacts already support
# ---------------------------------------------------------------------------


def run_cheap_phases(reports_dir: Path, models_dir: Path, cfg: Config) -> dict:
    """Importance, curves, operating points and the drift KS. About ten seconds."""
    booster, _calibrator, feature_cols, cost_cfg = load_artifacts(models_dir)

    print("[1/6] Feature importance, from the persisted booster")
    importance = importance_table(booster)
    _write(importance, reports_dir, "feature_importance.csv")
    used = int((importance["gain"] > 0).sum())
    print(f"  {len(importance)} features, {used} with gain above zero")
    print(importance.head(10).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("[2/6] ROC and PR curves over test, and the queue operating points")
    scored_test = _load(reports_dir, "scored_test.parquet")
    y_test = scored_test["isFraud"].to_numpy()
    _write(roc_points(y_test, scored_test["p"]), reports_dir, "roc_curve.csv")
    _write(pr_points(y_test, scored_test["p"]), reports_dir, "pr_curve.csv")

    # The costs come from the artefact metadata, not from CONFIG: they are the
    # ones the published evaluation was measured under (persist.py).
    points = operating_points_table(scored_test, cost_cfg, cfg.policy.daily_capacity_pct)
    _write(points, reports_dir, "operating_points.csv")
    print(points.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("[3/6] Score distributions: KS between the partitions already scored")
    scored_calib = _load(reports_dir, "scored_calib.parquet")
    ks_cheap = score_ks_table({"calib": scored_calib, "test": scored_test})
    print(ks_cheap.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    return {
        "booster": booster,
        "feature_cols": feature_cols,
        "cost_cfg": cost_cfg,
        "importance": importance,
        "operating_points": points,
        "scored_calib": scored_calib,
        "scored_test": scored_test,
        "ks_cheap": ks_cheap,
    }


# ---------------------------------------------------------------------------
# Phases 4 and 5: the feature rebuild, which is what costs
# ---------------------------------------------------------------------------


def _check_scoring_reproduces(rebuilt: np.ndarray, persisted: pd.Series) -> float:
    """Abort unless the reconstructed test scores match the persisted ones.

    The rebuild has to land on the same numbers, or the diagnostics would be
    describing a model other than the published one. Returns the largest
    absolute difference so the run can report it rather than only assert it.
    """
    persisted_values = persisted.to_numpy(dtype=float)
    if len(rebuilt) != len(persisted_values):
        raise SystemExit(
            f"The rebuild produced {len(rebuilt):,} test rows against "
            f"{len(persisted_values):,} persisted. The artefacts and the data "
            "do not correspond; rerun the pipeline."
        )
    np.testing.assert_allclose(
        rebuilt,
        persisted_values,
        rtol=1e-9,
        atol=1e-12,
        err_msg=(
            "The reconstructed test scores differ from reports/scored_test.parquet. "
            "Either the feature pipeline is not deterministic or the artefacts in "
            "models/artifacts/ are stale. Every diagnostic below would describe a "
            "model that is not the published one."
        ),
    )
    return float(np.max(np.abs(rebuilt - persisted_values)))


def run_feature_phases(
    reports_dir: Path,
    cheap: dict,
    cfg: Config,
    df_raw: pd.DataFrame,
    top_n: int = TOP_N_FEATURES,
    keep_train: bool = False,
) -> tuple[dict, pd.DataFrame | None, list[str]]:
    """Rebuild the features, re-score train and test, and report drift and redundancy.

    Returns the summary fragment, the train frame (only when `keep_train`, for
    the learning curve) and the feature columns.
    """
    booster = cheap["booster"]
    feature_cols = cheap["feature_cols"]

    print("[4/6] Rebuilding the features and re-scoring")
    parts, rebuilt_cols = prepare(df_raw, cfg)
    del df_raw
    gc.collect()

    if rebuilt_cols != feature_cols:
        raise SystemExit(
            f"The rebuild selected {len(rebuilt_cols)} features and the artefacts "
            f"were trained on {len(feature_cols)}. The feature definition moved "
            "since the model was fitted; rerun the pipeline."
        )

    # calib and the embargo are not needed: the calib scoring is already
    # persisted and the embargo is never used for anything by design.
    for unused in ("embargo", "calib"):
        parts.pop(unused, None)
    gc.collect()

    test_scores = predict_scores(booster, parts["test"], feature_cols)
    max_diff = _check_scoring_reproduces(test_scores, cheap["scored_test"]["score_raw"])
    print(f"  Test scoring reproduced exactly, max abs difference {max_diff:.3e}")

    train_scores = predict_scores(booster, parts["train"], feature_cols)
    scored_train = parts["train"][["day", "isFraud"]].copy()
    scored_train["score_raw"] = train_scores
    print(
        f"  train scored: {len(scored_train):,} rows, days "
        f"{parts['train']['day'].min()}-{parts['train']['day'].max()}"
    )

    print("[5/6] Redundancy, feature drift, and the KS that mixes both effects")
    ranked = top_features(cheap["importance"], top_n)

    corr = correlation_matrix(parts["train"], ranked)
    _write(corr, reports_dir, "feature_correlation.csv", index=True)
    off_diagonal = corr.where(~np.eye(len(corr), dtype=bool))
    strong = int((off_diagonal.abs() > 0.9).sum().sum() // 2)
    print(f"  Spearman over the top {top_n}: {strong} pairs above 0.9 in absolute value")

    v_cols = [c for c in feature_cols if c.startswith(V_COLUMN_PREFIX) and c[1:].isdigit()]
    blocks = null_pattern_blocks(parts["train"], v_cols)
    _write(blocks, reports_dir, "v_null_blocks.csv")
    print(f"  {len(v_cols)} V-columns fall into {len(blocks)} null-pattern blocks")

    psi = psi_by_month(parts["train"], parts["test"], ranked, days_per_month=DRIFT_BIN_DAYS)
    psi.index.name = "week"
    _write(psi, reports_dir, "psi_by_week.csv", index=True)
    print(f"  PSI over {DRIFT_BIN_DAYS} day bins, worst feature-week: {psi.max().max():.4f}")

    ks_full = score_ks_table(
        {
            "train": scored_train,
            "calib": cheap["scored_calib"],
            "test": cheap["scored_test"],
        }
    )
    _write(ks_full, reports_dir, "score_ks.csv")
    print(ks_full.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    train = parts["train"] if keep_train else None
    del parts
    gc.collect()

    summary = {
        "n_features": len(feature_cols),
        "top_n_features": top_n,
        "score_reproduction_max_abs_diff": max_diff,
        "correlated_pairs_above_0_9": strong,
        "v_columns": len(v_cols),
        "v_null_blocks": len(blocks),
        "psi_max": float(psi.max().max()),
        "psi_features_above_moderate": int((psi.max(axis=0) > 0.25).sum()),
    }
    return summary, train, feature_cols


# ---------------------------------------------------------------------------
# Phase 6: the clean overtraining measurement
# ---------------------------------------------------------------------------


def run_learning_curve(
    reports_dir: Path,
    train: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config,
    booster,
    valid_len: int,
) -> dict:
    """Retrain the CV folds recording both series, and check they reproduce the model."""
    print("[6/6] Learning curve: retraining the folds to record both series")
    folds = expanding_window_folds(
        cfg.split.train_end_day, n_folds=cfg.split.n_cv_folds, valid_len=valid_len
    )
    curve = learning_curve_folds(train, feature_cols, folds, lgbm_params(cfg.model))
    _write(curve, reports_dir, "learning_curve.csv")

    summary = overtraining_summary(curve)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # The folds that were just retrained must choose the same number of trees
    # the published booster carries, or this curve describes a different model.
    n_estimators = int(np.median(summary["best_iteration"]))
    published = booster.num_trees()
    print(f"  n_estimators from the median best iteration: {n_estimators} (booster: {published})")
    if n_estimators != published:
        raise SystemExit(
            f"The retrained folds give n_estimators={n_estimators} and the persisted "
            f"booster has {published} trees. The curve is not describing the published "
            "model; do not report it."
        )

    return {
        "cv_mean_valid_ap": float(summary["valid_ap_at_best"].mean()),
        "cv_mean_fit_ap": float(summary["fit_ap_at_best"].mean()),
        "overtraining_gap_mean": float(summary["gap_at_best"].mean()),
        "n_estimators": n_estimators,
    }


# ---------------------------------------------------------------------------


def run_diagnostics(
    reports_dir: Path,
    models_dir: Path,
    cfg: Config,
    df_raw: pd.DataFrame | None = None,
    learning_curve: bool = False,
    valid_len: int = 20,
    top_n: int = TOP_N_FEATURES,
) -> dict:
    """Run the phases the inputs allow, write every report, and return the summary."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    cheap = run_cheap_phases(reports_dir, models_dir, cfg)

    points = cheap["operating_points"].set_index("queue")
    summary = {
        "features_with_gain": int((cheap["importance"]["gain"] > 0).sum()),
        "top_feature": str(cheap["importance"].iloc[0]["feature"]),
        "top_feature_gain_pct": float(cheap["importance"].iloc[0]["gain_pct"]),
        "queue_capacity": int(points.loc["daily_topk_by_score", "capacity"]),
        "recall_at_capacity_by_score": float(points.loc["daily_topk_by_score", "tpr"]),
        "recall_at_capacity_by_value": float(points.loc["daily_topk_by_value", "tpr"]),
        "ks_calib_vs_test": float(
            cheap["ks_cheap"]
            .set_index(["pair", "subset"])
            .loc[("calib_vs_test", "all"), "statistic"]
        ),
    }

    if df_raw is not None:
        feature_summary, train, feature_cols = run_feature_phases(
            reports_dir, cheap, cfg, df_raw, top_n=top_n, keep_train=learning_curve
        )
        summary |= feature_summary
        if learning_curve and train is not None:
            summary |= run_learning_curve(
                reports_dir, train, feature_cols, cfg, cheap["booster"], valid_len
            )
    else:
        print("  Skipping the feature phases: nothing to rebuild them from.")

    (reports_dir / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n  {json.dumps(summary, indent=2)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cheap-only",
        action="store_true",
        help="Phases 1 to 3 only: no feature rebuild, no data needed beyond reports/.",
    )
    parser.add_argument(
        "--learning-curve",
        action="store_true",
        help="Also retrain the CV folds to measure overtraining. Adds several minutes.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Rebuild features from the synthetic dataset instead of data/processed.",
    )
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--top-n", type=int, default=TOP_N_FEATURES)
    args = parser.parse_args()

    cfg = CONFIG
    valid_len = 20
    df_raw = None

    if not args.cheap_only:
        if args.synthetic:
            print("SYNTHETIC dataset: it proves the chain runs, it produces no results.\n")
            df_raw = make_synthetic_transactions()
            cfg = dataclasses.replace(cfg, split=dataclasses.replace(cfg.split, **SYNTHETIC_SPLIT))
            valid_len = 10
        else:
            df_raw = load_processed(args.processed_dir)
            print(f"{len(df_raw):,} transactions from {args.processed_dir}\n")

    run_diagnostics(
        args.reports_dir,
        args.models_dir,
        cfg,
        df_raw=df_raw,
        learning_curve=args.learning_curve,
        valid_len=valid_len,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
