"""Calibración de probabilidades — Platt vs. isotónica, SOLO sobre calib.

Va en: fraud-review-queue/src/fraudq/models/calibrate.py

## Por qué existe este módulo (design.md §7.3, decisión A5)

Toda la capa de costos del Día 6 consume `p` como probabilidad REAL. El score
crudo de un GBDT rankea bien pero no está calibrado; y calibrar sobre train
daría probabilidades sobreconfiadas (el modelo ya vio esas etiquetas). Regla 3
del split (§4.2): el calibrador se ajusta EXCLUSIVAMENTE en la partición de
calibración (días 130-155), que el modelo nunca vio.

## Los dos métodos (§7.3)

- **Platt**: una sigmoide sobre el LOG-ODDS del score. Dos parámetros, robusto,
  funciona con pocos positivos. Supone que la distorsión es logística.
- **Isotónica**: no paramétrica; solo asume monotonía. Más flexible, pero con
  pocos datos sobreajusta (escalones). Con ~26 días de calib × ~3.5 % de
  positivos hay señal suficiente — la comparación decide, no el dogma.

Ambos exponen la MISMA interfaz: `.predict(scores) -> p en [0,1]`, monotónica
no-decreciente respecto del score crudo. Esa interfaz es un contrato: la
vigila `tests/test_calibrated_probs_valid.py`.

## Cómo elegir sin hacerse trampa

La elección Platt vs. isotónica usa un holdout TEMPORAL dentro de calib
(`temporal_calibration_split`): se ajusta en los primeros días, se compara
Brier/ECE en los últimos. Comparar sobre los mismos datos del ajuste
favorecería siempre a la isotónica (es más flexible). El test set no aparece
por ningún lado: se mira UNA vez, el Día 6.

sklearn se importa dentro de las funciones: el módulo es importable sin él.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Clip para el log-odds: evita ±inf en scores exactamente 0 o 1.
_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class PlattCalibrator:
    """p = sigmoid(a * logit(score) + b). Monótona por construcción (a >= 0)."""

    coef_: float
    intercept_: float

    def predict(self, scores) -> np.ndarray:
        return _sigmoid(self.coef_ * _logit(scores) + self.intercept_)


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Envuelve una IsotonicRegression ajustada. Monótona por definición."""

    _iso: object

    def predict(self, scores) -> np.ndarray:
        return np.asarray(self._iso.predict(np.asarray(scores, dtype=float)), dtype=float)


def fit_platt(scores, y) -> PlattCalibrator:
    """Platt scaling sobre el log-odds del score.

    Se ajusta sobre logit(s) y no sobre s directamente: el score de un GBDT ya
    vive en (0,1) y la distorsión típica es aproximadamente lineal en log-odds.
    `C` grande = sin regularización efectiva (dos parámetros no necesitan
    prior, y regularizar sesgaría el intercept hacia 0.5).
    """
    from sklearn.linear_model import LogisticRegression

    z = _logit(scores).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z, np.asarray(y, dtype=int))
    coef = float(lr.coef_[0][0])
    if coef < 0:
        # Una pendiente negativa invertiría el ranking del modelo: score alto
        # -> p baja. Eso no es calibrar, es enmascarar un modelo roto.
        raise ValueError(
            f"Platt produjo pendiente negativa ({coef:.4f}): el score no "
            "rankea. Revisa el modelo antes de calibrar."
        )
    return PlattCalibrator(coef_=coef, intercept_=float(lr.intercept_[0]))


def fit_isotonic(scores, y) -> IsotonicCalibrator:
    """Regresión isotónica creciente, acotada a [0,1], clip fuera de rango."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(np.asarray(scores, dtype=float), np.asarray(y, dtype=float))
    return IsotonicCalibrator(_iso=iso)


def temporal_calibration_split(
    df_calib: pd.DataFrame,
    holdout_days: int = 6,
    day_col: str = "day",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parte calib en (ajuste, holdout) POR TIEMPO, no al azar.

    Los últimos `holdout_days` días de calib quedan como holdout para comparar
    Platt vs. isotónica. Misma religión que el resto del proyecto: hasta la
    elección del calibrador respeta la flecha del tiempo. Un split aleatorio
    aquí no sería un pecado grave (el calibrador es un mapa score->p), pero el
    temporal es igual de barato y no abre la puerta a discusiones.
    """
    if holdout_days < 1:
        raise ValueError(f"holdout_days debe ser >= 1, recibido {holdout_days}.")
    cut = int(df_calib[day_col].max()) - holdout_days
    fit_df = df_calib[df_calib[day_col] <= cut]
    hold_df = df_calib[df_calib[day_col] > cut]
    if fit_df.empty or hold_df.empty:
        raise ValueError(
            f"Split de calibración degenerado (corte en día {cut}): "
            f"{len(fit_df)} filas de ajuste, {len(hold_df)} de holdout."
        )
    return fit_df.reset_index(drop=True), hold_df.reset_index(drop=True)


def evaluate_calibrator(calibrator, scores, y_true, n_bins: int = 10) -> dict:
    """Brier y ECE de un calibrador sobre (scores, y). `None` = score crudo."""
    from fraudq.evaluate.metrics import brier_score, ece

    p = np.asarray(scores, dtype=float) if calibrator is None else calibrator.predict(scores)
    return {"brier": brier_score(y_true, p), "ece": ece(y_true, p, n_bins=n_bins)}


def compare_calibrators(fit_scores, fit_y, eval_scores, eval_y, n_bins: int = 10) -> pd.DataFrame:
    """Tabla raw / platt / isotonic con Brier y ECE sobre el conjunto de eval.

    `raw` es la fila "antes" de la curva antes/después del README (§7.2): la
    evidencia empírica de que el score crudo no era una probabilidad.
    """
    rows = {
        "raw": None,
        "platt": fit_platt(fit_scores, fit_y),
        "isotonic": fit_isotonic(fit_scores, fit_y),
    }
    table = pd.DataFrame(
        {
            name: evaluate_calibrator(cal, eval_scores, eval_y, n_bins=n_bins)
            for name, cal in rows.items()
        }
    ).T
    table.index.name = "calibrator"
    return table
