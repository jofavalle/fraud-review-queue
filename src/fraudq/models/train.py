"""Baselines del Día 4: regresión logística + LightGBM, SIN rebalanceo.

Va en: fraud-review-queue/src/fraudq/models/train.py

## La regla contracultural (design.md §7.2) — codificada, no solo escrita

> No SMOTE. No undersampling. No `scale_pos_weight`. No `is_unbalance`.

Todas esas técnicas distorsionan las probabilidades predichas, y toda la capa
de costos depende de que `p` sea una probabilidad real. Aquí la regla no es un
comentario: `_validate_params` **lanza** si aparece un parámetro de rebalanceo.
El desbalance se maneja con la métrica (PR-AUC), no corrompiendo el
entrenamiento.

## Fuente única de hiperparámetros

Los hiperparámetros viven en `config.py` (`ModelConfig`, design.md §9.1) — no
aquí. Estas funciones reciben `params: dict`; pásalos desde tu config
(`dataclasses.asdict(...)` o como lo tengas modelado). Recuerda el invariante
de tu propio config: `subsample_freq = 1` explícito — LightGBM ignora
`subsample` si `subsample_freq = 0`, sin advertencia.

## Qué se evalúa hoy (y qué NO se toca)

La CV de ventana expansiva corre DENTRO de train (días 0-119). Hoy se reporta
PR-AUC por fold. **Ni calib ni test se tocan**: calib es del calibrador (Día 5)
y test se mira UNA vez (Día 6, H6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Parámetros que rompen la calibración de probabilidades. Prohibidos por diseño.
_FORBIDDEN_PARAMS = ("scale_pos_weight", "is_unbalance", "class_weight",
                     "pos_bagging_fraction", "neg_bagging_fraction")


def _validate_params(params: dict) -> None:
    """Codifica el invariante del §7.2: sin rebalanceo, punto."""
    bad = [k for k in _FORBIDDEN_PARAMS if k in params]
    if bad:
        raise ValueError(
            f"Parámetros de rebalanceo prohibidos por diseño: {bad}. "
            "Distorsionan p y derrumban la capa de costos (design.md §7.2)."
        )
    if params.get("subsample") is not None and not params.get("subsample_freq"):
        raise ValueError(
            "subsample sin subsample_freq >= 1: LightGBM lo IGNORA en silencio. "
            "Declara subsample_freq=1 explícito (decisión del Día 1)."
        )


@dataclass
class CVResult:
    """Resultado de la validación cruzada de ventana expansiva."""
    fold_ap: list[float] = field(default_factory=list)   # PR-AUC (AP) por fold
    best_iters: list[int] = field(default_factory=list)  # mejor iteración por fold

    @property
    def n_estimators(self) -> int:
        """Nº de árboles para el modelo final: mediana de las mejores iteraciones."""
        return int(np.median(self.best_iters))

    def summary(self) -> str:
        aps = ", ".join(f"{a:.4f}" for a in self.fold_ap)
        return (f"PR-AUC por fold: [{aps}] | media={np.mean(self.fold_ap):.4f} "
                f"| n_estimators (mediana)={self.n_estimators}")


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
    """CV de ventana expansiva (design.md §4.3) con early stopping por fold.

    `folds` viene de `fraudq.data.split.expanding_window_folds`. El propósito es
    doble: (a) PR-AUC honesto fuera de muestra dentro de train, (b) elegir
    n_estimators para el ajuste final sin tocar calib/test.
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
            {**params, "objective": "binary", "metric": "average_precision",
             "verbosity": -1},
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
    """Ajuste final sobre TODO train (0-119) con n_estimators fijado por la CV.

    Sin early stopping aquí: no hay conjunto de validación legítimo que no sea
    del futuro, y ese fue exactamente el trabajo de la CV.
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
    """Regresión logística: el baseline honesto (design.md §7.1).

    Pipeline imputación (mediana) -> escalado -> logística. `class_weight=None`
    explícito: el mismo principio de no-rebalanceo aplica al baseline, o la
    comparación de calibraciones del Día 5 no sería justa.

    Úsala con un conjunto PEQUEÑO de features numéricas; el punto es tener una
    referencia, no competir con LightGBM.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight=None)),
    ])
    pipe.fit(df_train[feature_cols], df_train[target])
    return pipe


def predict_scores(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Scores en [0,1] con interfaz uniforme para ambos modelos."""
    if hasattr(model, "predict_proba"):          # sklearn Pipeline
        return model.predict_proba(df[feature_cols])[:, 1]
    return model.predict(df[feature_cols])       # lgb.Booster
