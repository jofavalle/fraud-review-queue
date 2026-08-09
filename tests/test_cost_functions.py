"""Especificación ejecutable de costs.py — casos calculados A MANO (§10).

Va en: fraud-review-queue/tests/test_cost_functions.py

Este test se commitea ANTES de implementar costs.py (TDD, como el Día 3).
Cada número de aquí salió de un lápiz, no de correr el código: si tu
implementación discrepa, la que está mal es la implementación. No edites el
test para que pase — edítalo solo si encuentras un error DE LÁPIZ, y en ese
caso documenta el hallazgo en la bitácora.

Parámetros de referencia (los del diseño §2.3):
F=20, m=0.25, phi=10, r=2.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fraudq.policy.costs import (
    cost_approve,
    cost_block,
    realized_cost,
    value_of_review,
)

CFG = SimpleNamespace(F=20.0, m=0.25, phi=10.0, r=2.0)


# ----------------------------------------------------- extremos (p = 0, p = 1)


def test_certain_fraud_makes_blocking_free():
    """p=1: bloquear no pierde nada (no había venta legítima que perder)."""
    assert cost_block(1.0, 100.0, CFG) == pytest.approx(0.0)
    # ...y aprobar cuesta el monto completo más el chargeback.
    assert cost_approve(1.0, 100.0, CFG) == pytest.approx(120.0)


def test_certain_legit_makes_approving_free():
    """p=0: aprobar es gratis; bloquear cuesta margen + fricción."""
    assert cost_approve(0.0, 100.0, CFG) == pytest.approx(0.0)
    assert cost_block(0.0, 100.0, CFG) == pytest.approx(0.25 * 100.0 + 10.0)  # 35


# ----------------------------------------------------------- casos con lápiz


def test_hand_computed_moderate_case():
    """p=0.2, amt=100: approve = 0.2·120 = 24; block = 0.8·35 = 28; V = 24-2 = 22."""
    assert cost_approve(0.2, 100.0, CFG) == pytest.approx(24.0)
    assert cost_block(0.2, 100.0, CFG) == pytest.approx(28.0)
    assert value_of_review(0.2, 100.0, CFG) == pytest.approx(22.0)


def test_hand_computed_small_amount():
    """p=0.5, amt=10: approve = 0.5·30 = 15; block = 0.5·12.5 = 6.25; V = 4.25."""
    assert cost_approve(0.5, 10.0, CFG) == pytest.approx(15.0)
    assert cost_block(0.5, 10.0, CFG) == pytest.approx(6.25)
    assert value_of_review(0.5, 10.0, CFG) == pytest.approx(4.25)


def test_value_is_negative_when_the_case_is_already_decided():
    """p=0.99, amt=10: block = 0.01·12.5 = 0.125 < r  =>  V = -1.875 < 0.

    La tesis en miniatura: en un caso ya claro, el humano no aporta — revisar
    destruye valor.
    """
    assert value_of_review(0.99, 10.0, CFG) == pytest.approx(0.125 - 2.0)


def test_value_peaks_at_moderate_probability():
    """V(·, amt) es máximo en p* = (m·a+φ)/((a+F)+(m·a+φ)) — §2.6.

    Para amt=100: p* = 35/155 ≈ 0.2258, y ahí ambos costos valen
    0.2258·120 = 0.7742·35 ≈ 27.10. Verificación NUMÉRICA (no exige que
    implementes p_star): V en p* supera a V en p*±0.1, y los dos costos se
    cruzan en p*.
    """
    amt = 100.0
    p_star = (0.25 * amt + 10.0) / ((amt + 20.0) + (0.25 * amt + 10.0))
    assert p_star == pytest.approx(35.0 / 155.0)
    assert cost_approve(p_star, amt, CFG) == pytest.approx(cost_block(p_star, amt, CFG))
    v_peak = value_of_review(p_star, amt, CFG)
    assert v_peak > value_of_review(p_star - 0.1, amt, CFG)
    assert v_peak > value_of_review(p_star + 0.1, amt, CFG)
    assert v_peak == pytest.approx(p_star * (amt + 20.0) - 2.0)


# ------------------------------------------------------------- vectorización


def test_vectorized_matches_scalar():
    """El contrato exige numpy: arrays entran, arrays salen, sin bucles."""
    p = np.array([0.0, 0.2, 0.5, 0.99, 1.0])
    amt = np.array([100.0, 100.0, 10.0, 10.0, 100.0])
    va = cost_approve(p, amt, CFG)
    vb = cost_block(p, amt, CFG)
    vv = value_of_review(p, amt, CFG)
    for i in range(len(p)):
        assert va[i] == pytest.approx(cost_approve(float(p[i]), float(amt[i]), CFG))
        assert vb[i] == pytest.approx(cost_block(float(p[i]), float(amt[i]), CFG))
        assert vv[i] == pytest.approx(value_of_review(float(p[i]), float(amt[i]), CFG))


# ------------------------------------------------------------ costo realizado


def test_realized_cost_hand_table():
    """La tabla contable completa, caso por caso (docstring de realized_cost)."""
    actions = np.array(["approve", "approve", "block", "block", "review", "review"], dtype=object)
    is_fraud = np.array([1, 0, 0, 1, 1, 0])
    amt = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    out = np.asarray(realized_cost(actions, is_fraud, amt, CFG), dtype=float)
    #        fraude aprobado, legítima aprobada, legítima bloqueada,
    #        fraude bloqueado, revisión (siempre r)
    assert out == pytest.approx([120.0, 0.0, 35.0, 0.0, 2.0, 2.0])


def test_expected_cost_is_probability_weighted_realized_cost():
    """Coherencia interna: E[realizado] == esperado, para cada acción.

    approve: p·(amt+F) + (1-p)·0 == cost_approve
    block:   p·0 + (1-p)·(m·amt+φ) == cost_block
    Si esto falla, tus dos "contabilidades" se contradicen y la simulación
    no mide lo que la política optimiza.
    """
    p, amt = 0.3, 80.0
    exp_approve = p * float(
        realized_cost(np.array(["approve"], dtype=object), np.array([1]), np.array([amt]), CFG)[0]
    ) + (1 - p) * float(
        realized_cost(np.array(["approve"], dtype=object), np.array([0]), np.array([amt]), CFG)[0]
    )
    exp_block = p * float(
        realized_cost(np.array(["block"], dtype=object), np.array([1]), np.array([amt]), CFG)[0]
    ) + (1 - p) * float(
        realized_cost(np.array(["block"], dtype=object), np.array([0]), np.array([amt]), CFG)[0]
    )
    assert exp_approve == pytest.approx(cost_approve(p, amt, CFG))
    assert exp_block == pytest.approx(cost_block(p, amt, CFG))
