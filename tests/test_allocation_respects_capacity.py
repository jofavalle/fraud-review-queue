"""Especificación ejecutable de allocate.py — incluida LA TESIS (§10).

Va en: fraud-review-queue/tests/test_allocation_respects_capacity.py

Como test_cost_functions: se commitea ANTES de implementar, los números están
hechos a mano, y no se edita para que pase. El caso `test_thesis_in_one_case`
es el proyecto entero en cuatro líneas: si tu asignación lo pasa, tu cola ya
no es la ingenua.

Parámetros de referencia: F=20, m=0.25, phi=10, r=2.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fraudq.policy.allocate import allocate_day
from fraudq.policy.costs import cost_approve, cost_block, value_of_review

CFG = SimpleNamespace(F=20.0, m=0.25, phi=10.0, r=2.0)


def _rng_case(n: int = 500, seed: int = 11):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.5, 6.0, size=n)               # mayoría de p bajas, cola alta
    amt = np.exp(rng.normal(3.5, 1.2, size=n))   # montos log-normales
    return p, amt


# ----------------------------------------------------------------- la tesis

def test_thesis_in_one_case():
    """Capacidad 1: el score alto NO se revisa; el ambiguo de monto alto SÍ.

    A: p=0.95, amt=50  -> approve=66.5, block=1.125, V=-0.875 < 0. Ya está
       decidido: bloquear casi no arriesga nada; un humano no aporta.
    B: p=0.25, amt=200 -> approve=55, block=45, V=43. Genuinamente ambiguo
       y caro: exactamente lo que vale la pena poner frente a un analista.

    El top-1 POR SCORE revisaría A. El top-1 POR VALOR revisa B.
    """
    p = np.array([0.95, 0.25])
    amt = np.array([50.0, 200.0])
    actions = allocate_day(p, amt, capacity=1, cfg=CFG)
    assert actions[1] == "review"
    assert actions[0] == "block"      # su acción automática más barata


# ------------------------------------------------------------- el contrato

def test_output_contract():
    p, amt = _rng_case()
    actions = np.asarray(allocate_day(p, amt, 25, CFG))
    assert actions.shape == p.shape
    assert set(np.unique(actions)) <= {"approve", "review", "block"}


def test_capacity_is_never_exceeded():
    p, amt = _rng_case()
    for capacity in (0, 1, 7, 50, 10_000):
        actions = np.asarray(allocate_day(p, amt, capacity, CFG))
        assert (actions == "review").sum() <= capacity


def test_reviews_are_exactly_the_top_value_candidates():
    """Greedy de verdad: las revisadas son las de mayor V entre las V>0.

    Con desempate estable (por posición), el conjunto esperado es calculable
    sin ambigüedad: los primeros `capacity` índices de
    np.argsort(-v, kind="stable") restringido a v > 0.
    """
    p, amt = _rng_case()
    capacity = 25
    v = value_of_review(p, amt, CFG)
    eligible_sorted = [i for i in np.argsort(-v, kind="stable") if v[i] > 0]
    expected = set(eligible_sorted[:capacity])

    actions = np.asarray(allocate_day(p, amt, capacity, CFG))
    got = set(np.flatnonzero(actions == "review"))
    assert got == expected


def test_spare_capacity_is_not_wasted_on_nonpositive_value():
    """V <= 0 no se revisa NI CON CAPACIDAD SOBRANTE: revisar lo decidido
    es pagar r por nada (§2.5, punto 3)."""
    # Todos los casos claros: p altísima o bajísima con montos bajos.
    p = np.array([0.99, 0.995, 0.001, 0.002])
    amt = np.array([10.0, 15.0, 20.0, 25.0])
    assert np.all(value_of_review(p, amt, CFG) <= 0)   # premisa del caso
    actions = np.asarray(allocate_day(p, amt, capacity=10, cfg=CFG))
    assert (actions == "review").sum() == 0


def test_non_reviewed_get_the_cheaper_auto_action():
    p, amt = _rng_case()
    actions = np.asarray(allocate_day(p, amt, 25, CFG))
    approve_cost = cost_approve(p, amt, CFG)
    block_cost = cost_block(p, amt, CFG)
    for i in np.flatnonzero(actions != "review"):
        if approve_cost[i] <= block_cost[i]:      # empate -> approve (contrato)
            assert actions[i] == "approve"
        else:
            assert actions[i] == "block"


def test_capacity_zero_means_fully_automatic():
    p, amt = _rng_case(n=50)
    actions = np.asarray(allocate_day(p, amt, 0, CFG))
    assert (actions == "review").sum() == 0


def test_negative_capacity_raises():
    p, amt = _rng_case(n=10)
    with pytest.raises(ValueError):
        allocate_day(p, amt, -1, CFG)


def test_deterministic_under_ties():
    """Dos transacciones idénticas y cupo para una: gana la PRIMERA (orden
    estable), y dos corridas dan lo mismo. El mismo principio que el
    desempate por TransactionID en las ventanas del Día 3."""
    p = np.array([0.25, 0.25, 0.25])
    amt = np.array([200.0, 200.0, 200.0])
    first = np.asarray(allocate_day(p, amt, 1, CFG))
    second = np.asarray(allocate_day(p, amt, 1, CFG))
    assert (first == second).all()
    assert first[0] == "review"
    assert (first == "review").sum() == 1
