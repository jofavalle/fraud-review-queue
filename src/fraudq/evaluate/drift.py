"""Drift: PSI mensual y degradación de rendimiento — completo (delegable §5.3).

Va en: fraud-review-queue/src/fraudq/evaluate/drift.py

La distribución del fraude se mueve (design.md §8.4). Dos diagnósticos baratos
y muy realistas:

1. **PSI** (Population Stability Index) por feature y por mes: ¿cuánto se
   movió la DISTRIBUCIÓN de las entradas respecto de train?
2. **Degradación por mes**: PR-AUC (y lo que quieras añadir) mes a mes sobre
   el test — ¿cuánto pierde el MODELO con el paso del tiempo?

La lectura conjunta es la frase del README: "el rendimiento cae X % entre el
primer y el último mes del test, lo que sugiere una cadencia de reentrenamiento
de N semanas".

Convención de "mes": bloques de 30 días RELATIVOS al inicio de cada partición
(`day` es un offset, no calendario — §3.3). Regla de oro del PSI (industria,
no teorema): < 0.10 estable, 0.10-0.25 movimiento moderado, > 0.25 drift serio.

Este módulo no importa nada de policy/: mide el modelo y los datos, no la
política. sklearn se importa dentro de la función que lo usa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Suavizado de proporciones vacías: evita log(0) y división por cero. El valor
# exacto no importa mientras sea pequeño y CONSTANTE entre comparaciones.
_EPS = 1e-6

PSI_THRESHOLDS = {"stable": 0.10, "moderate": 0.25}


def psi(expected, actual, n_bins: int = 10) -> float:
    """PSI de `actual` respecto de `expected` (la referencia, típicamente train).

    Bins por CUANTILES de la referencia (así cada bin de `expected` pesa ~1/n y
    el PSI no lo domina un bin arbitrario de una distribución sesgada). Los NaN
    se excluyen en ambos lados; si la tasa de nulos también deriva, eso se ve
    aparte en el conteo (y LightGBM los consume nativamente de todos modos).

    PSI = sum( (a_i - e_i) * ln(a_i / e_i) )  sobre las proporciones por bin.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    # Cuantiles de la referencia; edges únicos (features discretas colapsan bins).
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:          # feature ~constante: no hay distribución que comparar
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_prop = np.clip(e_counts / e_counts.sum(), _EPS, None)
    a_prop = np.clip(a_counts / a_counts.sum(), _EPS, None)
    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))


def add_relative_month(df: pd.DataFrame, day_col: str = "day",
                       days_per_month: int = 30) -> pd.DataFrame:
    """Copia de `df` con la columna `month`: bloques de 30 días desde SU inicio."""
    out = df.copy()
    out["month"] = (out[day_col] - out[day_col].min()) // days_per_month
    return out


def psi_by_month(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    features: list[str],
    day_col: str = "day",
    days_per_month: int = 30,
    n_bins: int = 10,
) -> pd.DataFrame:
    """PSI de cada feature, mes a mes de `df_current` contra TODO `df_reference`.

    La referencia es train completo (lo que el modelo aprendió); cada mes del
    test se compara contra ella. Devuelve un DataFrame ancho: filas = mes,
    columnas = features, valores = PSI. Ideal para un heatmap en el notebook.
    """
    missing = [c for c in features if c not in df_reference.columns
               or c not in df_current.columns]
    if missing:
        raise KeyError(f"Features ausentes en referencia o actual: {missing}.")

    current = add_relative_month(df_current, day_col, days_per_month)
    rows = {}
    for month, g in current.groupby("month", sort=True):
        rows[int(month)] = {
            f: psi(df_reference[f].to_numpy(), g[f].to_numpy(), n_bins=n_bins)
            for f in features
        }
    out = pd.DataFrame(rows).T
    out.index.name = "month"
    return out


def performance_by_month(
    df: pd.DataFrame,
    p_col: str = "p",
    target: str = "isFraud",
    day_col: str = "day",
    days_per_month: int = 30,
) -> pd.DataFrame:
    """PR-AUC y tasa de fraude por mes relativo del test (design.md §8.4).

    Consume el scoring PERSISTIDO del Día 6 (reports/scored_test.parquet) — no
    re-scorea nada ni re-mira el test para tomar decisiones: describe, mes a
    mes, la evaluación que ya ocurrió. La caída porcentual entre el primer y el
    último mes es la frase de cadencia de reentrenamiento del README.
    """
    from sklearn.metrics import average_precision_score

    monthly = add_relative_month(df, day_col, days_per_month)
    rows = []
    for month, g in monthly.groupby("month", sort=True):
        y = g[target].to_numpy()
        rows.append({
            "month": int(month),
            "n": len(g),
            "fraud_rate": float(y.mean()),
            "pr_auc": float(average_precision_score(y, g[p_col].to_numpy()))
            if 0 < y.sum() < len(g) else float("nan"),
        })
    return pd.DataFrame(rows)
