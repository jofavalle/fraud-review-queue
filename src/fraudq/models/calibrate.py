"""Probability calibration: Platt against isotonic, on calib ONLY.

## Why this module exists (design.md §6.3)

The whole cost layer consumes `p` as a REAL probability. The raw score of a
GBDT ranks well but is not calibrated, and calibrating on train would give
overconfident probabilities, since the model has already seen those labels.
Rule 3 of the split (§4.3): the calibrator is fitted EXCLUSIVELY on the
calibration partition (days 130 to 155), which the model never saw.

## The two methods (§6.3)

- **Platt**: a sigmoid over the LOG-ODDS of the score. Two parameters, robust,
  works with few positives. It assumes the distortion is logistic.
- **Isotonic**: non-parametric, assuming only monotonicity. More flexible, but
  with little data it overfits into steps. With around 26 days of calib at
  roughly 3.5 % positives there is enough signal, and the comparison decides
  rather than the dogma.

Both expose the SAME interface: `.predict(scores) -> p in [0,1]`,
non-decreasing in the raw score. That interface is a contract, and
`tests/test_calibrated_probs_valid.py` guards it.

## How to choose without cheating

The choice between Platt and isotonic uses a TEMPORAL holdout inside calib
(`temporal_calibration_split`): fit on the first days, compare Brier and ECE on
the last. Comparing on the same data used to fit would always favour isotonic,
which is more flexible. The test set appears nowhere: it is looked at ONCE, at
the end.

sklearn is imported inside the functions, so the module imports without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Clip for the log-odds: it avoids infinities at scores of exactly 0 or 1.
_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class PlattCalibrator:
    """p = sigmoid(a * logit(score) + b). Monotonic by construction, a >= 0."""

    coef_: float
    intercept_: float

    def predict(self, scores) -> np.ndarray:
        return _sigmoid(self.coef_ * _logit(scores) + self.intercept_)


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Wraps a fitted IsotonicRegression. Monotonic by definition."""

    _iso: object

    def predict(self, scores) -> np.ndarray:
        return np.asarray(self._iso.predict(np.asarray(scores, dtype=float)), dtype=float)


def fit_platt(scores, y) -> PlattCalibrator:
    """Platt scaling over the log-odds of the score.

    It is fitted on logit(s) rather than on s directly: a GBDT score already
    lives in (0,1) and the typical distortion is roughly linear in log-odds. A
    large `C` means no effective regularisation, since two parameters need no
    prior and regularising would bias the intercept towards 0.5.
    """
    from sklearn.linear_model import LogisticRegression

    z = _logit(scores).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z, np.asarray(y, dtype=int))
    coef = float(lr.coef_[0][0])
    if coef < 0:
        # A negative slope would invert the model's ranking: high score, low p.
        # That is not calibrating, it is masking a broken model.
        raise ValueError(
            f"Platt produced a negative slope ({coef:.4f}): the score does not "
            "rank. Check the model before calibrating."
        )
    return PlattCalibrator(coef_=coef, intercept_=float(lr.intercept_[0]))


def fit_isotonic(scores, y) -> IsotonicCalibrator:
    """Increasing isotonic regression, bounded to [0,1], clipped out of range."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(np.asarray(scores, dtype=float), np.asarray(y, dtype=float))
    return IsotonicCalibrator(_iso=iso)


def temporal_calibration_split(
    df_calib: pd.DataFrame,
    holdout_days: int = 6,
    day_col: str = "day",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut calib into (fit, holdout) BY TIME, not at random.

    The last `holdout_days` days of calib become the holdout on which Platt and
    isotonic are compared. Same religion as the rest of the project: even the
    choice of calibrator respects the arrow of time. A random split here would
    not be a grave sin, since the calibrator is a score-to-p map, but the
    temporal one is just as cheap and opens no argument.
    """
    if holdout_days < 1:
        raise ValueError(f"holdout_days must be >= 1, got {holdout_days}.")
    cut = int(df_calib[day_col].max()) - holdout_days
    fit_df = df_calib[df_calib[day_col] <= cut]
    hold_df = df_calib[df_calib[day_col] > cut]
    if fit_df.empty or hold_df.empty:
        raise ValueError(
            f"Degenerate calibration split (cut at day {cut}): "
            f"{len(fit_df)} fit rows, {len(hold_df)} holdout rows."
        )
    return fit_df.reset_index(drop=True), hold_df.reset_index(drop=True)


def evaluate_calibrator(calibrator, scores, y_true, n_bins: int = 10) -> dict:
    """Brier and ECE of a calibrator over (scores, y). `None` means raw score."""
    from fraudq.evaluate.metrics import brier_score, ece

    p = np.asarray(scores, dtype=float) if calibrator is None else calibrator.predict(scores)
    return {"brier": brier_score(y_true, p), "ece": ece(y_true, p, n_bins=n_bins)}


def compare_calibrators(fit_scores, fit_y, eval_scores, eval_y, n_bins: int = 10) -> pd.DataFrame:
    """A raw / platt / isotonic table with Brier and ECE over the eval set.

    `raw` is the "before" row of the before-and-after curve in the README: the
    empirical evidence that the raw score was not a probability.
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
