"""Calibration test: valid probabilities, and monotonic in the raw score.

Probabilities must land in [0,1] and stay monotonic with respect to the raw
score, and the calibration metrics are checked against cases worked out by
hand. Monotonicity is the assertion worth understanding: a calibrator that
reorders the scores is not calibrating, it is changing the model's ranking.

The synthetic data encodes the real scenario: a score that RANKS well but is
badly calibrated, being overconfident. Calibrating must improve Brier without
touching the order.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudq.evaluate.metrics import (
    brier_score,
    calibration_by_decile,
    ece,
    reliability_table,
)

sklearn = pytest.importorskip(
    "sklearn", reason="the calibrators use sklearn, which is in the repo environment"
)

from fraudq.models.calibrate import fit_isotonic, fit_platt  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: an overconfident score with a correct ranking.
# The REAL probability is sigmoid(z); the reported score is sigmoid(2.5 z + 1):
# the same information on a distorted scale, the signature of an uncalibrated
# model.
# ---------------------------------------------------------------------------


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@pytest.fixture(scope="module")
def miscalibrated():
    rng = np.random.default_rng(42)
    z = rng.normal(-2.0, 1.5, size=6000)  # mostly legitimate cases
    p_true = _sigmoid(z)
    y = (rng.random(6000) < p_true).astype(int)
    scores = _sigmoid(2.5 * z + 1.0)
    # Half to fit the calibrator, half to evaluate it.
    return {
        "fit": (scores[:3000], y[:3000]),
        "eval": (scores[3000:], y[3000:]),
    }


# ------------------------------------------------------------------ contract


def test_probabilities_live_in_unit_interval(miscalibrated):
    s_fit, y_fit = miscalibrated["fit"]
    s_eval, _ = miscalibrated["eval"]
    for cal in (fit_platt(s_fit, y_fit), fit_isotonic(s_fit, y_fit)):
        p = cal.predict(s_eval)
        assert np.all(p >= 0.0) and np.all(p <= 1.0)
        assert not np.any(np.isnan(p))


def test_calibrators_are_monotone_wrt_raw_score(miscalibrated):
    """Calibrating must NOT reorder: score_i >= score_j => p_i >= p_j.

    If this fails, the "calibrator" is changing the model's ranking, and
    precision@K before and after calibrating would differ, which is another way
    of saying it is no longer the same model.
    """
    s_fit, y_fit = miscalibrated["fit"]
    grid = np.linspace(0.001, 0.999, 500)
    for cal in (fit_platt(s_fit, y_fit), fit_isotonic(s_fit, y_fit)):
        p = cal.predict(grid)
        assert np.all(np.diff(p) >= -1e-12)


def test_calibration_improves_brier_out_of_sample(miscalibrated):
    """On a genuinely miscalibrated score, calibrating must pay off.

    It is evaluated OUTSIDE the fitting data, on the eval half: the same
    discipline as the temporal holdout inside calib.
    """
    s_fit, y_fit = miscalibrated["fit"]
    s_eval, y_eval = miscalibrated["eval"]
    raw = brier_score(y_eval, s_eval)
    for cal in (fit_platt(s_fit, y_fit), fit_isotonic(s_fit, y_fit)):
        assert brier_score(y_eval, cal.predict(s_eval)) < raw


# ------------------------------------------- the metrics, on hand-made cases


def test_brier_hand_cases():
    assert brier_score([0, 1], [0.0, 1.0]) == 0.0
    assert brier_score([0, 1], [0.5, 0.5]) == pytest.approx(0.25)
    assert brier_score([0], [1.0]) == pytest.approx(1.0)


def test_ece_hand_case():
    """Two groups, worked out with a pencil.

    Bin [0, 0.1): 4 cases at p=0.05, 1 positive -> |0.25 - 0.05| = 0.20, w=0.4
    Bin [0.9, 1): 6 cases at p=0.95, 5 positive -> |5/6 - 0.95| = 0.1167, w=0.6
    ECE = 0.4 * 0.20 + 0.6 * 0.1167 = 0.15
    """
    p = np.array([0.05] * 4 + [0.95] * 6)
    y = np.array([1, 0, 0, 0] + [1, 1, 1, 1, 1, 0])
    assert ece(y, p, n_bins=10) == pytest.approx(0.15, abs=1e-9)


def test_ece_zero_when_perfectly_calibrated_bins():
    # In each bin, frac_pos == mean_p exactly.
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0] + [1, 1, 1, 0])
    assert ece(y, p, n_bins=4) == pytest.approx(0.0, abs=1e-12)


def test_reliability_table_accounts_for_everything(miscalibrated):
    s_eval, y_eval = miscalibrated["eval"]
    table = reliability_table(y_eval, s_eval, n_bins=10)
    assert table["count"].sum() == len(s_eval)
    assert table["weight"].sum() == pytest.approx(1.0)
    assert ((table["frac_pos"] >= 0) & (table["frac_pos"] <= 1)).all()


def test_calibration_by_decile_shape_and_gap(miscalibrated):
    s_eval, y_eval = miscalibrated["eval"]
    table = calibration_by_decile(y_eval, s_eval)
    assert len(table) == 10
    # count splits into tenths; rank method="first" breaks ties evenly
    assert table["count"].sum() == len(s_eval)
    assert (table["gap"] >= 0).all()
    # The fixture's score is overconfident at the high end, so the top decile
    # must show a clearly POSITIVE gap, mean_p > frac_pos. That is the reason
    # to report by decile: this error is invisible in the aggregate.
    top = table.iloc[-1]
    assert top["mean_p"] > top["frac_pos"]
