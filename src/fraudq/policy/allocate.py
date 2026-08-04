"""Asignación de la cola de revisión bajo capacidad finita.

Un día de transacciones entra; salen tres acciones posibles por transacción. La
capacidad se renueva cada jornada, y agrupar por día es responsabilidad de
`simulate.py`: aquí llega un día ya agrupado.

El contrato, que asumen `simulate.py` y las políticas comparadas:

1. Se revisan a lo sumo ``capacity`` transacciones, y TODAS con ``V > 0``. La
   capacidad sobrante no se gasta: revisar un caso sin valor es pagar ``r`` por
   nada.
2. Las revisadas son las de mayor ``V``. El greedy es óptimo aquí porque todas
   las revisiones cuestan lo mismo: es una mochila con pesos unitarios
   (`docs/design.md` §2.5).
3. El resto recibe su acción automática más barata. Ante empate, "approve".
4. Determinista ante empates de ``V``: orden estable por posición, el mismo
   principio que el desempate por `TransactionID` en las ventanas retrospectivas.

La especificación ejecutable es `tests/test_allocation_respects_capacity.py`,
incluido el caso que codifica la tesis del proyecto: con capacidad para uno, una
transacción de score 0.95 y monto bajo se bloquea, y una de score 0.25 y monto
alto se revisa. Rankear por score haría lo contrario.
"""

from __future__ import annotations

import numpy as np

from fraudq.policy.costs import cost_approve, cost_block, value_of_review


def allocate_day(p, amt, capacity, cfg):
    """Asigna approve / review / block a un día de transacciones.

    Parameters
    ----------
    p:
        Probabilidades CALIBRADAS. Sin calibración ``V`` no significa nada y esta
        función optimiza una ficción (`docs/design.md` §4.3).
    amt:
        Montos (``TransactionAmt``), alineados con ``p``.
    capacity:
        Cupo de revisión del día, entero >= 0. Con 0, todo es automático.
    cfg:
        Duck-typed con ``F``, ``m``, ``phi`` y ``r``.

    Returns
    -------
    np.ndarray
        dtype object, del mismo largo que ``p``, con valores exactos "approve",
        "review" o "block".

    Raises
    ------
    ValueError
        Si ``capacity`` es negativa.
    """
    capacity = int(capacity)
    if capacity < 0:
        raise ValueError(f"capacity debe ser >= 0, se recibió {capacity}")

    p = np.asarray(p, dtype=float)
    amt = np.asarray(amt, dtype=float)

    # Acción automática de partida: la más barata de las dos, empate a "approve".
    actions = np.where(
        cost_approve(p, amt, cfg) <= cost_block(p, amt, cfg), "approve", "block"
    ).astype(object)

    if capacity == 0:
        return actions

    v = value_of_review(p, amt, cfg)

    # Orden estable: ante V idénticos gana la posición menor, y dos corridas
    # sobre la misma entrada dan el mismo resultado.
    order = np.argsort(-v, kind="stable")
    to_review = order[v[order] > 0][:capacity]
    actions[to_review] = "review"

    return actions
