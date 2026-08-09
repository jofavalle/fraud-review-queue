"""Pipeline driver: from the ingested parquet files to the policy table.

    python -m fraudq.pipeline                 # real data, data/processed/
    python -m fraudq.pipeline --synthetic     # synthetic dataset, no Kaggle

It chains what the modules under `src/fraudq/` do separately:

    features -> temporal split -> CV -> training -> calibration
             -> scoring -> threshold on calib -> policy comparison on test

and leaves on disk the artefacts consumed by the results notebook, the queue
simulator and the API:

    reports/scored_calib.parquet     scores and p over the calibration partition
    reports/scored_test.parquet      the same over test, after the single look
    reports/policy_comparison.csv    the table of the four policies
    models/artifacts/                booster, calibrator and metadata

On the test partition: it is touched ONCE, at the end, and the result is
persisted immediately. The sensitivity analysis and the drift report work off
that parquet and never look at it again. Rerunning this script over the same
data is legitimate; using the test number to pick a hyperparameter, a threshold
or a cost assumption is not (`docs/design.md` §9, invariant 5).
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from fraudq.config import CONFIG, DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, Config
from fraudq.data.split import expanding_window_folds, split_by_day
from fraudq.data.synthetic import SYNTHETIC_SPLIT, make_synthetic_transactions
from fraudq.evaluate.metrics import brier_score, ece, pr_auc, roc_auc
from fraudq.evaluate.policies import compare_policies, fit_single_threshold, headline_savings
from fraudq.features.build import FrequencyEncoder, add_base_features, build_features
from fraudq.models.calibrate import (
    compare_calibrators,
    fit_isotonic,
    fit_platt,
    temporal_calibration_split,
)
from fraudq.models.persist import save_artifacts
from fraudq.models.train import cv_lightgbm, predict_scores, train_final_lgbm

#: Pipeline outputs. `config.py` defines FIGURES_DIR = reports/figures, so
#: reports/ is its parent; the model artefacts hang off models/.
REPORTS_DIR = FIGURES_DIR.parent
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

#: Columns that are never features: identifiers, the target, and the ones that
#: only exist to partition. `uid` is left out deliberately: it is a grouping
#: key, and feeding it in as a number would hand the model a raw customer
#: identifier.
_NOT_FEATURES = frozenset({"TransactionID", "isFraud", "TransactionDT", "day", "uid"})

_LGBM_PARAM_FIELDS = (
    "learning_rate",
    "num_leaves",
    "min_child_samples",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "reg_lambda",
)


def lgbm_params(model_cfg) -> dict:
    """Translate `ModelConfig` into the parameters LightGBM understands.

    They are passed explicitly rather than through `asdict`: `n_estimators`,
    `metric` and `calibration_method` govern the flow, not the booster, and
    slipping them into the dictionary would have LightGBM ignore them silently.
    """
    params = {f: getattr(model_cfg, f) for f in _LGBM_PARAM_FIELDS}
    params["seed"] = model_cfg.random_state
    params["num_threads"] = model_cfg.n_jobs
    return params


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Every numeric column that is neither an identifier nor the target."""
    return [
        c for c in df.columns if c not in _NOT_FEATURES and pd.api.types.is_numeric_dtype(df[c])
    ]


def prepare(df_raw: pd.DataFrame, cfg: Config) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Features and temporal split.

    The frequency encoder is fitted on train ONLY and applied to all four
    partitions: that is what stops the calibration and test distributions from
    leaking into the representation the model is trained on.
    """
    feats = build_features(df_raw)
    df = df_raw.merge(
        feats.drop(columns=["TransactionDT", "TransactionAmt"]),
        on="TransactionID",
        how="left",
    )
    df = add_base_features(df)

    parts = split_by_day(df, cfg.split)

    encoder = FrequencyEncoder().fit(parts["train"])
    parts = {name: encoder.transform(part) for name, part in parts.items()}

    feature_cols = select_feature_columns(parts["train"])
    return parts, feature_cols


def train(parts: dict[str, pd.DataFrame], feature_cols: list[str], cfg: Config, valid_len: int):
    """Expanding-window CV to fix n_estimators, then the final fit."""
    folds = expanding_window_folds(
        cfg.split.train_end_day, n_folds=cfg.split.n_cv_folds, valid_len=valid_len
    )
    params = lgbm_params(cfg.model)
    cv = cv_lightgbm(parts["train"], feature_cols, folds, params)
    print(f"  CV: {cv.summary()}")

    booster = train_final_lgbm(parts["train"], feature_cols, params, cv.n_estimators)
    return booster, cv


def calibrate(booster, parts: dict[str, pd.DataFrame], feature_cols: list[str], holdout_days: int):
    """Pick a calibrator on a temporal holdout of calib, then refit it on all of calib.

    The calibrator is fitted only on data the model never saw
    (`docs/design.md` §4.3). The whole cost layer depends on ``p`` being a real
    probability: with an overconfident model, the "optimal" policy stops being
    optimal.
    """
    fit_df, hold_df = temporal_calibration_split(parts["calib"], holdout_days=holdout_days)
    fit_scores = predict_scores(booster, fit_df, feature_cols)
    hold_scores = predict_scores(booster, hold_df, feature_cols)

    table = compare_calibrators(fit_scores, fit_df["isFraud"], hold_scores, hold_df["isFraud"])
    print(table.to_string(float_format=lambda v: f"{v:.5f}"))

    # The comparison runs on the holdout and the winner is picked by Brier,
    # which penalises calibration and discrimination at once.
    winner = str(table["brier"].idxmin()) if "brier" in table.columns else "isotonic"
    if winner == "raw":
        # This can happen and is not a failure: it means the raw score was
        # already well calibrated. Isotonic is used anyway so that downstream
        # code gets an object with the same interface, and the output says so.
        print("  Raw score wins on Brier; using isotonic for a uniform interface.")
        winner = "isotonic"
    print(f"  Calibrator chosen: {winner}")

    calib_scores = predict_scores(booster, parts["calib"], feature_cols)
    fit_fn = fit_platt if winner == "platt" else fit_isotonic
    calibrator = fit_fn(calib_scores, parts["calib"]["isFraud"])
    return calibrator, calib_scores, table


def _scored_frame(df: pd.DataFrame, scores, p) -> pd.DataFrame:
    cols = ["TransactionID", "day", "TransactionAmt", "isFraud"]
    out = df[cols].copy()
    out["score_raw"] = scores
    out["p"] = p
    return out


def run_pipeline(
    df_raw: pd.DataFrame,
    cfg: Config,
    reports_dir: Path,
    models_dir: Path,
    valid_len: int = 20,
    holdout_days: int = 6,
) -> dict:
    """Run the whole chain and return the headline figures."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Features and temporal split")
    parts, feature_cols = prepare(df_raw, cfg)
    for name, part in parts.items():
        print(f"  {name:>8}: {len(part):>7,} rows, days {part['day'].min()}-{part['day'].max()}")
    print(f"  {len(feature_cols)} features")

    print("[2/6] Training")
    booster, cv = train(parts, feature_cols, cfg, valid_len)

    print("[3/6] Calibration")
    calibrator, calib_scores, calib_table = calibrate(booster, parts, feature_cols, holdout_days)

    print("[4/6] Scoring the calibration partition")
    p_calib = calibrator.predict(calib_scores)
    scored_calib = _scored_frame(parts["calib"], calib_scores, p_calib)
    scored_calib.to_parquet(reports_dir / "scored_calib.parquet", index=False)

    threshold = fit_single_threshold(scored_calib, cfg.cost)
    print(f"  Single-threshold policy, threshold fitted on calib: {threshold:.4f}")

    print("[5/6] Evaluation on test (the single look)")
    test_scores = predict_scores(booster, parts["test"], feature_cols)
    p_test = calibrator.predict(test_scores)
    scored_test = _scored_frame(parts["test"], test_scores, p_test)
    scored_test.to_parquet(reports_dir / "scored_test.parquet", index=False)

    y_test = scored_test["isFraud"].to_numpy()
    model_metrics = {
        "pr_auc": pr_auc(y_test, p_test),
        "roc_auc": roc_auc(y_test, p_test),
        "brier": brier_score(y_test, p_test),
        "ece": ece(y_test, p_test),
    }
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in model_metrics.items()))

    comparison = compare_policies(scored_test, cfg.cost, cfg.policy.daily_capacity_pct, threshold)
    comparison.to_csv(reports_dir / "policy_comparison.csv")
    print(comparison.to_string(float_format=lambda v: f"{v:.4f}"))

    savings = headline_savings(comparison)
    print(
        f"  Saving from ranking by value rather than by score: "
        f"{savings['savings_per_1k']:.4f} per $1,000"
    )

    print("[6/6] Artefacts")
    artifacts = save_artifacts(models_dir, booster, calibrator, feature_cols, cfg.cost)
    print(f"  {artifacts}")

    return {
        "cv": cv,
        "calibration": calib_table,
        "threshold": threshold,
        "model_metrics": model_metrics,
        "comparison": comparison,
        "savings": savings,
        "feature_cols": feature_cols,
    }


def load_processed(processed_dir: Path) -> pd.DataFrame:
    """Load the ingested parquet, joined with identity if it is there."""
    transactions = processed_dir / "transactions.parquet"
    if not transactions.exists():
        raise SystemExit(
            f"{transactions} not found.\n"
            "Run first:  ./scripts/download_data.sh  &&  python -m fraudq.data.ingest\n"
            "Or try the chain without real data:  python -m fraudq.pipeline --synthetic"
        )
    df = pd.read_parquet(transactions)

    identity = processed_dir / "identity.parquet"
    if identity.exists():
        # Left join, nulls kept: the absence of identity data is itself a
        # signal, and LightGBM handles NaN natively.
        df = df.merge(pd.read_parquet(identity), on="TransactionID", how="left")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic transactions instead of the parquet files in data/processed.",
    )
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args()

    cfg = CONFIG
    valid_len = 20

    if args.synthetic:
        print("SYNTHETIC dataset: it proves the chain runs, it produces no results.\n")
        df_raw = make_synthetic_transactions()
        # The production split does not fit a dataset of a handful of days.
        cfg = dataclasses.replace(cfg, split=dataclasses.replace(cfg.split, **SYNTHETIC_SPLIT))
        valid_len = 10
    else:
        df_raw = load_processed(args.processed_dir)
        print(f"{len(df_raw):,} transactions from {args.processed_dir}\n")

    run_pipeline(
        df_raw,
        cfg,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        valid_len=valid_len,
        holdout_days=3 if args.synthetic else 6,
    )


if __name__ == "__main__":
    main()
