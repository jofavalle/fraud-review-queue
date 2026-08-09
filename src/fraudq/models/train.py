"""Baselines: logistic regression and LightGBM, with NO resampling.

## The counter-cultural rule (design.md §6.2), encoded rather than just written

> No SMOTE. No undersampling. No `scale_pos_weight`. No `is_unbalance`.

All of those distort the predicted probabilities, and the entire cost layer
depends on `p` being a real probability. Here the rule is not a comment:
`_validate_params` **raises** if a resampling parameter turns up. The imbalance
is handled through the metric, PR-AUC, not by corrupting the training.

## A single source of hyperparameters

The hyperparameters live in `config.py` (`ModelConfig`), not here. These
functions take `params: dict`, passed in from that config. Note the invariant
the config itself carries: an explicit `subsample_freq = 1`, because LightGBM
ignores `subsample` when `subsample_freq = 0`, without a warning.

## What gets evaluated, and what is NOT touched

The expanding-window CV runs INSIDE train (days 0 to 119) and reports PR-AUC
per fold. **Neither calib nor test is touched**: calib belongs to the
calibrator, and test is looked at ONCE, at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Parameters that break probability calibration. Forbidden by design.
_FORBIDDEN_PARAMS = (
    "scale_pos_weight",
    "is_unbalance",
    "class_weight",
    "pos_bagging_fraction",
    "neg_bagging_fraction",
)


def _validate_params(params: dict) -> None:
    """Encode the invariant of §6.2: no resampling, full stop."""
    bad = [k for k in _FORBIDDEN_PARAMS if k in params]
    if bad:
        raise ValueError(
            f"Resampling parameters forbidden by design: {bad}. "
            "They distort p and bring down the cost layer (design.md §6.2)."
        )
    if params.get("subsample") is not None and not params.get("subsample_freq"):
        raise ValueError(
            "subsample without subsample_freq >= 1: LightGBM IGNORES it silently. "
            "Declare subsample_freq=1 explicitly."
        )


@dataclass
class CVResult:
    """The result of the expanding-window cross-validation."""

    fold_ap: list[float] = field(default_factory=list)  # PR-AUC (AP) per fold
    best_iters: list[int] = field(default_factory=list)  # best iteration per fold

    @property
    def n_estimators(self) -> int:
        """Trees for the final model: the median of the best iterations."""
        return int(np.median(self.best_iters))

    def summary(self) -> str:
        aps = ", ".join(f"{a:.4f}" for a in self.fold_ap)
        return (
            f"PR-AUC per fold: [{aps}] | mean={np.mean(self.fold_ap):.4f} "
            f"| n_estimators (median)={self.n_estimators}"
        )


def cv_lightgbm(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[tuple[int, int], tuple[int, int]]],
    params: dict,
    target: str = "isFraud",
    day_col: str = "day",
    num_boost_round: int = 5000,
    early_stopping_rounds: int = 200,
) -> CVResult:
    """Expanding-window CV (design.md §4.4) with early stopping per fold.

    `folds` comes from `fraudq.data.split.expanding_window_folds`. The purpose
    is twofold: an honest out-of-sample PR-AUC inside train, and a choice of
    n_estimators for the final fit without touching calib or test.
    """
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score

    _validate_params(params)
    result = CVResult()

    day = df_train[day_col]
    for (t_lo, t_hi), (v_lo, v_hi) in folds:
        fit = df_train[(day >= t_lo) & (day <= t_hi)]
        valid = df_train[(day >= v_lo) & (day <= v_hi)]

        dtrain = lgb.Dataset(fit[feature_cols], label=fit[target])
        dvalid = lgb.Dataset(valid[feature_cols], label=valid[target], reference=dtrain)

        booster = lgb.train(
            {**params, "objective": "binary", "metric": "average_precision", "verbosity": -1},
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        scores = booster.predict(valid[feature_cols], num_iteration=booster.best_iteration)
        result.fold_ap.append(float(average_precision_score(valid[target], scores)))
        result.best_iters.append(int(booster.best_iteration))

    return result


def train_final_lgbm(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    n_estimators: int,
    target: str = "isFraud",
):
    """Final fit over ALL of train (0 to 119) with n_estimators fixed by the CV.

    No early stopping here: there is no legitimate validation set that is not
    from the future, and settling that was exactly the CV's job.
    """
    import lightgbm as lgb

    _validate_params(params)
    dtrain = lgb.Dataset(df_train[feature_cols], label=df_train[target])
    return lgb.train(
        {**params, "objective": "binary", "verbosity": -1},
        dtrain,
        num_boost_round=n_estimators,
    )


def train_logistic_baseline(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    target: str = "isFraud",
):
    """Logistic regression: the honest baseline (design.md §6.1).

    A pipeline of median imputation, then scaling, then the logistic model.
    `class_weight=None` is explicit: the same no-resampling principle applies to
    the baseline, or the calibration comparison would not be fair.

    Use it with a SMALL set of numeric features. The point is a reference
    point, not competing with LightGBM.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight=None)),
        ]
    )
    pipe.fit(df_train[feature_cols], df_train[target])
    return pipe


def predict_scores(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Scores in [0,1], one uniform interface for both models."""
    if hasattr(model, "predict_proba"):  # sklearn Pipeline
        return model.predict_proba(df[feature_cols])[:, 1]
    return model.predict(df[feature_cols])  # lgb.Booster
