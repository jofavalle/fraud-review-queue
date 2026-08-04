"""Métricas del modelo — las que el diseño dice que importan (design.md §7.4).

Va en: fraud-review-queue/src/fraudq/evaluate/metrics.py
**REEMPLAZA al metrics.py del Día 4.** Cambio aditivo: se suman Brier, ECE,
la tabla de fiabilidad y la calibración por decil (lo que el propio módulo
anunciaba para el Día 5). Nada de lo existente cambia de firma.

| Métrica                | Papel                                                |
|------------------------|------------------------------------------------------|
| PR-AUC (AP)            | La métrica de ranking correcta bajo desbalance.      |
| ROC-AUC                | Se reporta, no lidera.                               |
| precision@K / recall@K | La métrica OPERATIVA: K = capacidad de revisión.     |
| Brier                  | Error cuadrático de las probabilidades.              |
| ECE                    | Desvío promedio ponderado respecto de la diagonal.   |
| calibración por decil  | El promedio esconde el error donde más importa.      |

Accuracy no existe aquí a propósito. El costo por $1,000 llega con la capa de
decisión (Día 6). Los imports de sklearn viven dentro de las funciones que los
usan: todo lo de calibración es numpy/pandas puro y el módulo es importable
(y testeable) sin sklearn instalado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pr_auc(y_true, scores) -> float:
    """Average precision (área bajo la curva precision-recall)."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, scores))


def _top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Máscara booleana del top-K por score, con desempate DETERMINISTA.

    `kind="stable"` fija el orden entre scores empatados (por posición). Sin
    eso, dos corridas podrían reportar precision@K distinta con los mismos
    datos — el mismo problema de los empates de `TransactionDT` en las
    ventanas, en versión numpy.
    """
    if k <= 0:
        raise ValueError(f"k debe ser positivo, recibido {k}.")
    scores = np.asarray(scores)
    k = min(k, len(scores))
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:k]] = True
    return mask


def precision_at_k(y_true, scores, k: int) -> float:
    """Fracción de fraude real dentro del top-K por score.

    Con K = capacidad diaria de analistas, es "de lo que mandé a revisar,
    cuánto era fraude".
    """
    y_true = np.asarray(y_true)
    mask = _top_k_mask(scores, k)
    return float(y_true[mask].mean())


def recall_at_k(y_true, scores, k: int) -> float:
    """Fracción del fraude total capturada por el top-K.

    "De todo el fraude que había, cuánto cayó en la cola de revisión."
    Si no hay positivos, devuelve NaN (mejor un NaN visible que un 0 falso).
    """
    y_true = np.asarray(y_true)
    total_pos = y_true.sum()
    if total_pos == 0:
        return float("nan")
    mask = _top_k_mask(scores, k)
    return float(y_true[mask].sum() / total_pos)


# ---------------------------------------------------------------------------
# Calibración (Día 5). Numpy/pandas puro — sin sklearn.
# ---------------------------------------------------------------------------

def brier_score(y_true, p) -> float:
    """Error cuadrático medio de las probabilidades: mean((p - y)^2).

    Descomponible en calibración + refinamiento (design.md §7.3). Sensible a
    AMBAS cosas; por eso acompaña al ECE (solo calibración) y no lo sustituye.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    if y_true.shape != p.shape:
        raise ValueError(f"Shapes distintos: y {y_true.shape} vs p {p.shape}.")
    return float(np.mean((p - y_true) ** 2))


def _bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin de ancho fijo en [0,1] por probabilidad predicha. p=1 cae en el último."""
    idx = np.floor(np.asarray(p, dtype=float) * n_bins).astype(int)
    return np.clip(idx, 0, n_bins - 1)


def ece(y_true, p, n_bins: int = 10) -> float:
    """Expected Calibration Error con bins de ancho fijo.

    Suma sobre bins de weight_b * |frac_pos_b - mean_p_b|: el desvío promedio
    ponderado respecto de la diagonal del reliability plot. 0 = perfectamente
    calibrado EN PROMEDIO — que es exactamente por lo que también se reporta
    por decil (`calibration_by_decile`): el agregado esconde el top del score.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = _bin_index(p, n_bins)
    total = 0.0
    n = len(p)
    if n == 0:
        raise ValueError("ece: recibido un array vacío.")
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        w = mask.sum() / n
        total += w * abs(y_true[mask].mean() - p[mask].mean())
    return float(total)


def reliability_table(y_true, p, n_bins: int = 10) -> pd.DataFrame:
    """Los datos del reliability plot (design.md §7.3, diagnóstico 1).

    Una fila por bin NO vacío: `mean_p` (eje x), `frac_pos` (eje y), `count` y
    `weight`. El plot en sí vive en el notebook de resultados — este módulo
    produce números, no figuras.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = _bin_index(p, n_bins)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin": b,
            "p_lo": b / n_bins,
            "p_hi": (b + 1) / n_bins,
            "mean_p": p[mask].mean(),
            "frac_pos": y_true[mask].mean(),
            "count": int(mask.sum()),
            "weight": mask.sum() / len(p),
        })
    return pd.DataFrame(rows)


def calibration_by_decile(y_true, p, scores=None) -> pd.DataFrame:
    """Calibración POR DECIL de score — el detalle que casi nadie hace (§7.3).

    Un modelo puede estar perfectamente calibrado en promedio y ser un
    desastre en el decil superior — justo donde se toman las decisiones caras.
    Deciles por cuantiles del SCORE CRUDO (si se pasa `scores`) o de `p`:
    así el decil 9 es "el 10 % que el modelo considera más sospechoso",
    independiente de cómo el calibrador haya movido los valores.

    Columnas: mean_p, frac_pos, gap (|mean_p - frac_pos|), count.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    ranking = p if scores is None else np.asarray(scores, dtype=float)
    decile = pd.qcut(pd.Series(ranking).rank(method="first"), 10, labels=False)
    df = pd.DataFrame({"decile": decile, "p": p, "y": y_true})
    out = (
        df.groupby("decile")
        .agg(mean_p=("p", "mean"), frac_pos=("y", "mean"), count=("y", "size"))
        .reset_index()
    )
    out["gap"] = (out["mean_p"] - out["frac_pos"]).abs()
    return out
