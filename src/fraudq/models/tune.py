"""Tuning de LightGBM con Optuna — TIMEBOXED a 3 horas de reloj (H5).

Va en: fraud-review-queue/src/fraudq/models/tune.py

## El lugar del tuning en este proyecto (léelo antes de correr nada)

El orden de sacrificio (plan §13) dice: **el tuning es lo PRIMERO que se
corta.** Nadie te contrata por subir el PR-AUC de 0.85 a 0.86; el proyecto se
gana en la capa de decisión de mañana. Por eso:

- `timeout_s` por defecto = 3 horas. Cuando suena, se acaba — el mejor trial
  hasta ese momento gana. No se extiende "porque iba mejorando".
- El objetivo es la MEDIA de PR-AUC en los folds de ventana expansiva, la
  misma CV del Día 4 (`cv_lightgbm`). Ni calib ni test se tocan.
- La guarda anti-rebalanceo del Día 4 (`_validate_params`) corre dentro de
  `cv_lightgbm` también aquí: Optuna no puede proponer `scale_pos_weight` ni
  aunque se lo pidieras.

Espacio de búsqueda deliberadamente CHICO: los 6-7 parámetros que mueven la
aguja en GBDT, con rangos razonables. Un espacio gigante en 3 horas es ruido.

optuna y lightgbm se importan dentro de la función: el módulo es importable
sin ellos (y optuna no es dependencia del resto del paquete).
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
    """Busca hiperparámetros maximizando la media de PR-AUC en la CV del Día 4.

    Parameters
    ----------
    base_params:
        Los parámetros de tu `ModelConfig` (fuente única, design.md §9.1).
        Lo que Optuna no toca (p. ej. `subsample_freq=1`) se hereda de aquí.
    timeout_s / n_trials:
        Lo que ocurra primero corta el estudio. El timeout ES el timebox de H5.

    Returns
    -------
    (best_params, study):
        `best_params` = base_params ∪ los del mejor trial — listos para
        `train_final_lgbm`. El `study` se devuelve para inspección; el registro
        de la decisión final va en la bitácora, no en un dashboard.
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
            # El invariante del Día 1: LightGBM IGNORA subsample si
            # subsample_freq = 0, sin advertencia. Explícito, siempre.
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
