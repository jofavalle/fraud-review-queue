"""Fixtures compartidas de los tests.

Va en: fraud-review-queue/tests/conftest.py

`sample_df` es un dataset sintético pequeño, diseñado a propósito para el test de
leakage: varios UIDs con varias transacciones cada uno, repartidas en el tiempo,
**un empate de `TransactionDT`** dentro de un UID (verifica que el desempate por
`TransactionID` hace el cálculo determinista) y **una fila con `addr1` nulo**
(verifica que una transacción sin identidad no recibe ni fabrica historia).

Construir fixtures sintéticas es boilerplate (columna delegable del plan §5.3). Lo
que NO es delegable —y por eso está en un scaffold aparte— es la aserción del test
que usa esta fixture.
"""

from __future__ import annotations

import pandas as pd
import pytest

SECONDS_PER_DAY = 86_400


def _row(txn_id: int, dt: int, card1: int, addr1: int, d1n: int, amt: float, fraud: int) -> dict:
    """Crea una fila con `day` y `D1` derivados de un `d1n` objetivo constante.

    Elegimos `D1 = day - d1n` para que `D1n = day - D1 == d1n` sea constante por
    cliente, que es justo el supuesto del UID (uid.py / design.md §5.2).
    """
    day = dt // SECONDS_PER_DAY
    return {
        "TransactionID": txn_id,
        "TransactionDT": dt,
        "day": day,
        "card1": card1,
        "addr1": addr1,
        "D1": day - d1n,
        "TransactionAmt": amt,
        "isFraud": fraud,  # presente pero las features NUNCA deben usarlo
    }


@pytest.fixture
def sample_df() -> pd.DataFrame:
    # Cliente A: card1=1000, addr1=50, D1n=5. Cuatro transacciones; dos comparten
    # TransactionDT (empate deliberado) para probar el determinismo.
    # Cliente B: card1=2000, addr1=60, D1n=8. Tres transacciones.
    rows = [
        _row(1, 1 * SECONDS_PER_DAY + 100, 1000, 50, 5, 50.0, 0),
        _row(2, 3 * SECONDS_PER_DAY + 200, 1000, 50, 5, 70.0, 0),
        _row(3, 3 * SECONDS_PER_DAY + 200, 1000, 50, 5, 30.0, 1),  # empate DT con id=2
        _row(4, 9 * SECONDS_PER_DAY + 500, 1000, 50, 5, 120.0, 0),
        _row(5, 2 * SECONDS_PER_DAY + 300, 2000, 60, 8, 200.0, 0),
        _row(6, 6 * SECONDS_PER_DAY + 100, 2000, 60, 8, 210.0, 0),
        _row(7, 8 * SECONDS_PER_DAY + 900, 2000, 60, 8, 400.0, 1),
    ]
    # Fila sin identidad: addr1 nulo -> uid NULL -> las 5 features deben ser NULL.
    # (groupby("uid") de pandas descarta NaN por defecto; el segundo test no la ve,
    # y así debe ser: no es "la primera txn de un uid", es una txn SIN uid.)
    rows.append(_row(8, 4 * SECONDS_PER_DAY + 400, 4000, None, 3, 99.0, 0))
    # Orden de entrada deliberadamente "desordenado" para no depender de él.
    df = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
    return df
