"""Construcción del UID — entity resolution, NO leakage.

Va en: fraud-review-queue/src/fraudq/features/uid.py

## La decisión (design.md §5.2, registro de decisiones A3)

IEEE-CIS no trae un identificador de cliente, pero se puede **reconstruir** uno
aproximado:

    D1n = day - D1

`D1` es "días desde que la tarjeta fue vista por primera vez". Restándolo del día
actual se obtiene una aproximación de la **fecha de registro de la tarjeta**, que
es (aproximadamente) **constante para esa tarjeta**. Combinada con `card1` y
`addr1`, identifica un cliente:

    uid = card1 + '_' + addr1 + '_' + D1n

### ¿Esto es leakage? No.

- **Construir el UID es *entity resolution*.** En un sistema de fraude real **sí**
  existe un identificador persistente de cliente en el momento del scoring.
  Reconstruirlo desde columnas anonimizadas recupera información que en producción
  estaría disponible. Es legítimo.
- **Lo que SÍ sería leakage** es agregar el *target* (`isFraud`) sobre el UID
  usando el dataset completo — eso mira etiquetas del futuro. Este módulo **no
  toca `isFraud`**; las agregaciones (retrospectivas) viven en `build.py` y las
  vigila `tests/test_no_future_leakage.py`.

Requiere que `df` ya tenga la columna `day` (la deriva la ingesta desde
`TransactionDT`; ver data/ingest.py).
"""

from __future__ import annotations

import pandas as pd

UID_COLS = ("card1", "addr1", "D1")


def add_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de `df` con las columnas `D1n` y `uid` añadidas.

    - `D1n = day - D1` (proxy de la fecha de registro de la tarjeta), como
      entero nullable (`Int64`). El casteo importa: si `D1` es float (tiene
      NaN), sin él `D1n` sería '5.0' en un subset y '5' en otro, y el uid
      cambiaría según qué filas estén presentes.
    - `uid` = 'card1_addr1_D1n' como string.

    No modifica `df` in situ.

    **Nulos:** si CUALQUIER componente es nulo, el `uid` completo queda **NULL**
    (pd.NA se propaga en la concatenación de dtype 'string'). Esto es
    deliberado y `build.py` lo respeta: las features de una fila sin uid son
    NULL. La alternativa —agrupar los nulos bajo un uid literal común— sería
    un error grave: `PARTITION BY` juntaría transacciones de clientes sin
    relación en una sola "identidad" gigante y los agregados fabricarían
    historia falsa.
    """
    missing = [c for c in ("day", *UID_COLS) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Faltan columnas para construir el UID: {missing}. "
            "¿Corriste la ingesta (deriva `day`) antes de las features?"
        )

    out = df.copy()
    # round() antes de Int64: estas columnas llegan como float (nullable: los
    # nulos fuerzan float64 en pandas) y el casteo directo falla o es inseguro
    # ante artefactos de coma flotante. Sin la normalizacion, addr1=50.0 daria
    # uid '1000_50.0_5' en vez de '1000_50_5'.
    def _int_str(series: pd.Series) -> pd.Series:
        return series.round().astype("Int64").astype("string")

    out["D1n"] = (out["day"] - out["D1"]).round().astype("Int64")
    out["uid"] = (
        _int_str(out["card1"])
        + "_"
        + _int_str(out["addr1"])
        + "_"
        + out["D1n"].astype("string")
    )
    return out
