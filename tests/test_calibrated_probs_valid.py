"""Test de calibración — completo (columna delegable del plan §5.3).

Va en: fraud-review-queue/tests/test_calibrated_probs_valid.py

Es el quinto test de la tabla de design.md §10: probabilidades válidas
(en [0,1]) y monotónicas respecto del score crudo, más las métricas de
calibración verificadas contra casos calculados a mano. Como el test del
split: mecánico, pero léelo y sé capaz de explicar cada aserción — en
particular POR QUÉ la monotonía importa (un calibrador que reordena los
scores ya no calibra: cambia el ranking del modelo).

Los datos sintéticos codifican el escenario real: un score que RANKEA bien
pero está mal calibrado (sobreconfiado). Calibrar debe mejorar Brier sin
tocar el orden.
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
    "sklearn", reason="los calibradores usan sklearn (está en el entorno del repo)"
)

from fraudq.models.calibrate import fit_isotonic, fit_platt  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: score sobreconfiado con ranking correcto.
# La probabilidad REAL es sigmoid(z); el score reportado es sigmoid(2.5 z + 1):
# misma información, escala distorsionada — la firma de un modelo sin calibrar.
# ---------------------------------------------------------------------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@pytest.fixture(scope="module")
def miscalibrated():
    rng = np.random.default_rng(42)
    z = rng.normal(-2.0, 1.5, size=6000)     # mayoría de casos "legítimos"
    p_true = _sigmoid(z)
    y = (rng.random(6000) < p_true).astype(int)
    scores = _sigmoid(2.5 * z + 1.0)
    # Mitad para ajustar el calibrador, mitad para evaluarlo.
    return {
        "fit": (scores[:3000], y[:3000]),
        "eval": (scores[3000:], y[3000:]),
    }


# ------------------------------------------------------------------ contrato

def test_probabilities_live_in_unit_interval(miscalibrated):
    s_fit, y_fit = miscalibrated["fit"]
    s_eval, _ = miscalibrated["eval"]
    for cal in (fit_platt(s_fit, y_fit), fit_isotonic(s_fit, y_fit)):
        p = cal.predict(s_eval)
        assert np.all(p >= 0.0) and np.all(p <= 1.0)
        assert not np.any(np.isnan(p))


def test_calibrators_are_monotone_wrt_raw_score(miscalibrated):
    """Calibrar NO puede reordenar: score_i >= score_j => p_i >= p_j.

    Si esto falla, el "calibrador" está cambiando el ranking del modelo — y
    entonces precision@K antes y después de calibrar serían distintas, que es
    otra forma de decir que ya no es el mismo modelo.
    """
    s_fit, y_fit = miscalibrated["fit"]
    grid = np.linspace(0.001, 0.999, 500)
    for cal in (fit_platt(s_fit, y_fit), fit_isotonic(s_fit, y_fit)):
        p = cal.predict(grid)
        assert np.all(np.diff(p) >= -1e-12)


def test_calibration_improves_brier_out_of_sample(miscalibrated):
    """Sobre un score genuinamente descalibrado, calibrar debe pagar.

    Se evalúa FUERA de los datos de ajuste (mitad eval): la misma disciplina
    que el holdout temporal dentro de calib.
    """
    s_fit, y_fit = miscalibrated["fit"]
    s_eval, y_eval = miscalibrated["eval"]
    raw = brier_score(y_eval, s_eval)
    for cal in (fit_platt(s_fit, y_fit), fit_isotonic(s_fit, y_fit)):
        assert brier_score(y_eval, cal.predict(s_eval)) < raw


# ------------------------------------------- métricas: casos hechos a mano

def test_brier_hand_cases():
    assert brier_score([0, 1], [0.0, 1.0]) == 0.0
    assert brier_score([0, 1], [0.5, 0.5]) == pytest.approx(0.25)
    assert brier_score([0], [1.0]) == pytest.approx(1.0)


def test_ece_hand_case():
    """Dos grupos, calculado con lápiz.

    Bin [0, 0.1): 4 casos con p=0.05, 1 positivo -> |0.25 - 0.05| = 0.20, w=0.4
    Bin [0.9, 1): 6 casos con p=0.95, 5 positivos -> |5/6 - 0.95| ≈ 0.1167, w=0.6
    ECE = 0.4 * 0.20 + 0.6 * 0.1167 = 0.15
    """
    p = np.array([0.05] * 4 + [0.95] * 6)
    y = np.array([1, 0, 0, 0] + [1, 1, 1, 1, 1, 0])
    assert ece(y, p, n_bins=10) == pytest.approx(0.15, abs=1e-9)


def test_ece_zero_when_perfectly_calibrated_bins():
    # En cada bin, frac_pos == mean_p exactamente.
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
    # count debe repartirse en décimas (rank method="first" desempata parejo)
    assert table["count"].sum() == len(s_eval)
    assert (table["gap"] >= 0).all()
    # El score de la fixture es sobreconfiado en el extremo alto: el decil
    # superior debe mostrar un gap POSITIVO claro (mean_p > frac_pos). Es la
    # razón de reportar por decil: este error es invisible en el agregado.
    top = table.iloc[-1]
    assert top["mean_p"] > top["frac_pos"]
