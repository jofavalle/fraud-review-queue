"""Model metrics: the ones the design says matter (design.md §6.4).

| Metric                 | Role                                                |
|------------------------|-----------------------------------------------------|
| PR-AUC (AP)            | The correct ranking metric under imbalance.         |
| ROC-AUC                | Reported, but it does not lead.                     |
| precision@K / recall@K | The OPERATIONAL metric: K = review capacity.        |
| Brier                  | Squared error of the probabilities.                 |
| ECE                    | Weighted mean deviation from the diagonal.          |
| calibration by decile  | The average hides the error where it matters most.  |

Accuracy is absent on purpose. Cost per $1,000 arrives with the decision layer.
The sklearn imports live inside the functions that use them: everything about
calibration is pure numpy and pandas, so the module imports, and is testable,
without sklearn installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pr_auc(y_true, scores) -> float:
    """Average precision, the area under the precision-recall curve."""
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, scores))


def roc_auc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, scores))


def _top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the top-K by score, with a DETERMINISTIC tie-break.

    `kind="stable"` fixes the order among tied scores, by position. Without it,
    two runs could report a different precision@K on the same data: the same
    problem as the `TransactionDT` ties in the windows, in numpy form.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    scores = np.asarray(scores)
    k = min(k, len(scores))
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:k]] = True
    return mask


def precision_at_k(y_true, scores, k: int) -> float:
    """Fraction of real fraud inside the top-K by score.

    With K set to the daily analyst capacity, it answers "of what I sent to
    review, how much was fraud".
    """
    y_true = np.asarray(y_true)
    mask = _top_k_mask(scores, k)
    return float(y_true[mask].mean())


def recall_at_k(y_true, scores, k: int) -> float:
    """Fraction of all fraud captured by the top-K.

    "Of all the fraud there was, how much landed in the review queue." With no
    positives it returns NaN: a visible NaN beats a false 0.
    """
    y_true = np.asarray(y_true)
    total_pos = y_true.sum()
    if total_pos == 0:
        return float("nan")
    mask = _top_k_mask(scores, k)
    return float(y_true[mask].sum() / total_pos)


# ---------------------------------------------------------------------------
# Calibration. Pure numpy and pandas, no sklearn.
# ---------------------------------------------------------------------------


def brier_score(y_true, p) -> float:
    """Mean squared error of the probabilities: mean((p - y)^2).

    It decomposes into calibration plus refinement (design.md §6.3), and is
    sensitive to BOTH. That is why it accompanies ECE, which sees calibration
    only, rather than replacing it.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    if y_true.shape != p.shape:
        raise ValueError(f"Different shapes: y {y_true.shape} against p {p.shape}.")
    return float(np.mean((p - y_true) ** 2))


def _bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Fixed-width bin in [0,1] by predicted probability. p=1 falls in the last."""
    idx = np.floor(np.asarray(p, dtype=float) * n_bins).astype(int)
    return np.clip(idx, 0, n_bins - 1)


def ece(y_true, p, n_bins: int = 10) -> float:
    """Expected Calibration Error with fixed-width bins.

    A sum over bins of weight_b * |frac_pos_b - mean_p_b|: the weighted mean
    deviation from the diagonal of the reliability plot. Zero means perfectly
    calibrated ON AVERAGE, which is exactly why it is also reported by decile
    in `calibration_by_decile`: the aggregate hides the top of the score.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = _bin_index(p, n_bins)
    total = 0.0
    n = len(p)
    if n == 0:
        raise ValueError("ece: got an empty array.")
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        w = mask.sum() / n
        total += w * abs(y_true[mask].mean() - p[mask].mean())
    return float(total)


def reliability_table(y_true, p, n_bins: int = 10) -> pd.DataFrame:
    """The data behind the reliability plot (design.md §6.3).

    One row per NON-empty bin: `mean_p` on the x axis, `frac_pos` on the y
    axis, plus `count` and `weight`. The plot itself lives in the results
    notebook; this module produces numbers, not figures.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = _bin_index(p, n_bins)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "p_lo": b / n_bins,
                "p_hi": (b + 1) / n_bins,
                "mean_p": p[mask].mean(),
                "frac_pos": y_true[mask].mean(),
                "count": int(mask.sum()),
                "weight": mask.sum() / len(p),
            }
        )
    return pd.DataFrame(rows)


def calibration_by_decile(y_true, p, scores=None) -> pd.DataFrame:
    """Calibration BY SCORE DECILE, the detail almost nobody reports (§6.3).

    A model can be perfectly calibrated on average and a disaster in the top
    decile, which is precisely where the expensive decisions are made. Deciles
    come from quantiles of the RAW SCORE when `scores` is passed, or of `p`
    otherwise, so decile 9 is "the 10 % the model finds most suspicious"
    regardless of how the calibrator moved the values.

    Columns: mean_p, frac_pos, gap (|mean_p - frac_pos|), count.
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
