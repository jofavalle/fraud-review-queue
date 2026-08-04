"""Agregados retrospectivos sobre el UID — el invariante anti-leakage en SQL.

Va en: fraud-review-queue/src/fraudq/features/build.py

## El invariante (design.md §5.3)

> **Todo agregado sobre el UID es estrictamente retrospectivo.** Para la fila `i`,
> el valor se calcula usando **solo transacciones anteriores del mismo UID** —
> jamás la fila actual ni ninguna futura.

En SQL eso es el frame:

    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING

Ese `1 PRECEDING` **es** el `.shift(1)` de pandas: excluye la fila actual. Es,
literalmente, el invariante anti-leakage expresado en window functions
(plan de SQL §4.6). `LAG(...)` cumple el mismo rol para el gap temporal.

## El contrato (lo que asume tests/test_no_future_leakage.py)

`build_features(df)`:

- Requiere en `df`: `TransactionID`, `TransactionDT`, `TransactionAmt`, y las
  columnas para el UID (`day`, `card1`, `addr1`, `D1`). No usa `isFraud`.
- Devuelve un `DataFrame` **ordenado por (`TransactionDT`, `TransactionID`)** con
  el índice reseteado, columnas: las de identidad más las features.
- **Filas sin uid** (algún componente nulo): sus cinco features son **NULL**.
  Sin identidad no hay historia; agrupar los nulos entre sí fabricaría historia
  cruzando clientes sin relación.
- **Propiedad clave (la que verifica el test):** para cualquier corte temporal
  `t`, las features de las filas con `TransactionDT <= t` son **idénticas** se
  calcule sobre el histórico completo o sobre el histórico truncado en `t`.
  Se cumple porque cada feature depende solo de filas estrictamente anteriores.

El desempate por `TransactionID` en el `ORDER BY` de la ventana hace el cálculo
**determinista** ante empates de `TransactionDT` — sin él, el reparto entre filas
de igual instante quedaría a criterio del motor y el test podría parpadear.

## Por qué DuckDB y no pandas

Son "queries de DuckDB" (columna delegable del plan §5.3): expresa el invariante
de forma declarativa, corre sobre el `DataFrame` sin copiarlo a otra estructura, y
es la misma práctica de window functions que necesitas para entrevistas. DuckDB
consulta un `DataFrame` de pandas directamente por su nombre de variable
registrado.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from fraudq.features.uid import add_uid

# Columnas de identidad que se conservan tal cual en la salida.
_ID_COLS = ("TransactionID", "uid", "TransactionDT", "TransactionAmt")

# Agregados estrictamente retrospectivos. Toda ventana ordena por
# (TransactionDT, TransactionID) para ser determinista; el frame excluye la fila
# actual con `1 PRECEDING`.
_FEATURE_SQL = """
SELECT
    TransactionID,
    uid,
    TransactionDT,
    TransactionAmt,

    -- Toda feature es NULL si uid IS NULL: sin identidad no hay historia que
    -- mirar, y NO se agrupan los nulos entre sí (sería historia fabricada).

    -- Nº de transacciones ANTERIORES del mismo uid (0 en la primera).
    CASE WHEN uid IS NULL THEN NULL
         ELSE COUNT(*) OVER w_prior END AS uid_txn_count_prior,

    -- Segundos desde la transacción anterior del uid (NULL en la primera).
    CASE WHEN uid IS NULL THEN NULL
         ELSE TransactionDT - LAG(TransactionDT) OVER w_order END AS uid_seconds_since_last,

    -- Media de montos PREVIOS (NULL en la primera).
    CASE WHEN uid IS NULL THEN NULL
         ELSE AVG(TransactionAmt) OVER w_prior END AS uid_amt_prior_mean,

    -- Monto actual / media previa. NULLIF evita dividir por 0.
    CASE WHEN uid IS NULL THEN NULL
         ELSE TransactionAmt / NULLIF(AVG(TransactionAmt) OVER w_prior, 0)
         END AS uid_amt_ratio,

    -- z-score contra la distribución PREVIA del uid.
    -- STDDEV_SAMP necesita >= 2 datos previos -> NULL en la 1ª y la 2ª txn.
    CASE WHEN uid IS NULL THEN NULL
         ELSE (TransactionAmt - AVG(TransactionAmt) OVER w_prior)
              / NULLIF(STDDEV_SAMP(TransactionAmt) OVER w_prior, 0)
         END AS uid_amt_zscore

FROM t
WINDOW
    w_order AS (PARTITION BY uid ORDER BY TransactionDT, TransactionID),
    w_prior AS (PARTITION BY uid ORDER BY TransactionDT, TransactionID
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
ORDER BY TransactionDT, TransactionID
"""

FEATURE_COLUMNS = (
    "uid_txn_count_prior",
    "uid_seconds_since_last",
    "uid_amt_prior_mean",
    "uid_amt_ratio",
    "uid_amt_zscore",
)


def build_features(
    df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Calcula los agregados retrospectivos del UID. Ver el contrato en el módulo.

    Parameters
    ----------
    df:
        Transacciones crudas (con las columnas del UID). No se modifica in situ.
    con:
        Conexión DuckDB opcional (para reutilizar). Si es None, se crea una
        efímera y se cierra al terminar.

    Returns
    -------
    DataFrame ordenado por (TransactionDT, TransactionID), índice reseteado.
    """
    with_uid = add_uid(df)

    owns_con = con is None
    con = con or duckdb.connect()
    try:
        con.register("t", with_uid)
        out = con.execute(_FEATURE_SQL).df()
        con.unregister("t")
    finally:
        if owns_con:
            con.close()

    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Features base (design.md §5.1)
# --------------------------------------------------------------------------
#
# Los agregados del UID de arriba son la parte cara y delicada. Estas son la
# parte barata, y no estaban en los entregables aunque el diseño las especifica:
# sin ellas el modelo se entrenaría solo con las cinco columnas del UID.
#
# Se separan en dos por una razón que no es estética: las de `add_base_features`
# dependen únicamente de la propia fila, así que son inmunes al leakage y se
# pueden calcular sobre el dataset entero. Las de `FrequencyEncoder` resumen la
# distribución de una columna, así que **se ajustan solo en train** y se aplican
# al resto; calcularlas sobre todo el dataset dejaría que calibración y test
# influyeran en la representación de las filas de entrenamiento.

BASE_FEATURE_COLUMNS = ("amt_log", "amt_decimal", "hour")

#: Categóricas de alta cardinalidad que se codifican por frecuencia.
FREQ_ENCODED_COLS = ("card1", "addr1", "P_emaildomain", "R_emaildomain")

SECONDS_PER_HOUR = 3_600
HOURS_PER_DAY = 24


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de §5.1 que dependen solo de la fila. No requieren ajuste.

    - ``amt_log``: ``log1p`` del monto, que comprime la cola larga.
    - ``amt_decimal``: la parte decimal del monto. Los importes convertidos de
      otra divisa o generados por un programa dejan firmas raras aquí, mientras
      que una compra humana tiende a terminar en .00 o .99.
    - ``hour``: hora del día. La deriva la ingesta; se recalcula si falta, para
      que la función sirva también sobre un DataFrame crudo.

    Los dominios de correo se reducen a su proveedor base (``gmail.com`` ->
    ``gmail``) y se dejan como categóricas: quien las convierte en número es
    `FrequencyEncoder`.
    """
    out = df.copy()

    amt = out["TransactionAmt"].astype(float)
    out["amt_log"] = np.log1p(amt)
    out["amt_decimal"] = amt % 1.0

    if "hour" not in out.columns:
        out["hour"] = (out["TransactionDT"] // SECONDS_PER_HOUR) % HOURS_PER_DAY

    for col in ("P_emaildomain", "R_emaildomain"):
        if col in out.columns:
            out[col] = out[col].astype("string").str.split(".").str[0]

    return out


class FrequencyEncoder:
    """Codificación por frecuencia de categóricas de alta cardinalidad (§5.1).

    Ajustada SOLO en train, que es lo que la hace legítima: la frecuencia de un
    `card1` es un resumen de la distribución, y calcularlo sobre el dataset
    completo filtraría al entrenamiento información de las particiones futuras.

    Una categoría no vista en train recibe frecuencia 0, que es la respuesta
    correcta: en el momento de entrenar, ese valor no existía.
    """

    def __init__(self, cols: tuple[str, ...] = FREQ_ENCODED_COLS) -> None:
        self.cols = cols
        self.freqs_: dict[str, pd.Series] = {}

    @property
    def feature_names(self) -> list[str]:
        return [f"{c}_freq" for c in self.freqs_]

    def fit(self, df_train: pd.DataFrame) -> FrequencyEncoder:
        self.freqs_ = {
            col: df_train[col].value_counts(normalize=True)
            for col in self.cols
            if col in df_train.columns
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.freqs_:
            raise RuntimeError("FrequencyEncoder sin ajustar: llama a fit() primero.")
        out = df.copy()
        for col, freq in self.freqs_.items():
            out[f"{col}_freq"] = out[col].map(freq).astype(float).fillna(0.0)
        return out
