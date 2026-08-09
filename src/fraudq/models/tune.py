"""LightGBM tuning with Optuna, TIMEBOXED to three hours of wall clock.

## Where tuning sits in this project (read before running anything)

Tuning is the FIRST thing to cut. Nothing here is won by lifting PR-AUC from
0.85 to 0.86; the project is won in the decision layer. Hence:

- `timeout_s` defaults to three hours. When it fires, it is over, and the best
  trial so far wins. It does not get extended "because it was still improving".
- The objective is the MEAN PR-AUC across the expanding-window folds, the same
  CV as `cv_lightgbm`. Neither calib nor test is touched.
- The anti-resampling guard (`_validate_params`) runs inside `cv_lightgbm` here
  too: Optuna cannot propose `scale_pos_weight` even if asked to.

The search space is deliberately SMALL: the six or seven parameters that move
the needle in a GBDT, over sensible ranges. A huge space in three hours is
noise.

optuna and lightgbm are imported inside the function, so the module imports
without them, and optuna is not a dependency of the rest of the package.
"""

from __future__ import annotations

import pandas as pd

from fraudq.models.train import CVResult, cv_lightgbm


def tune_lightgbm(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[tuple[int, int], tuple[int, int]]],
    base_params: dict,
    timeout_s: int = 3 * 3600,
    n_trials: int = 60,
    seed: int = 7,
) -> tuple[dict, object]:
    """Search hyperparameters maximising the mean PR-AUC over the CV folds.

    Parameters
    ----------
    base_params:
        The parameters of `ModelConfig`, the single source. Whatever Optuna
        does not touch, such as `subsample_freq=1`, is inherited from here.
    timeout_s / n_trials:
        Whichever comes first ends the study. The timeout IS the timebox.

    Returns
    -------
    (best_params, study):
        `best_params` is base_params merged with the best trial's, ready for
        `train_final_lgbm`. The `study` comes back for inspection.
    """
    import optuna

    def objective(trial: optuna.Trial) -> float:
        params = {
            **base_params,
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            # The invariant: LightGBM IGNORES subsample when subsample_freq is
            # 0, without a warning. Explicit, always.
            "subsample_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        cv: CVResult = cv_lightgbm(df_train, feature_cols, folds, params)
        return float(pd.Series(cv.fold_ap).mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)

    best = {**base_params, **study.best_params, "subsample_freq": 1}
    return best, study
