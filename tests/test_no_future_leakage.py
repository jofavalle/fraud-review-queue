"""
================================================================================
  ESTE ARCHIVO LO ESCRIBES TÚ.  (plan §5.3 — columna "Escribo yo")
================================================================================

El test de leakage es tu **credencial de rigor** y material de entrevista directo
("¿cómo verificas que tus features no miran el futuro?"). Por eso NO viene resuelto:
las secciones marcadas `TODO (tú)` son tuyas. Escríbelas, entiéndelas y sé capaz
de defender cada línea en voz alta.

**Orden (design.md §6, hito H3):** este test se escribe y se commitea *ANTES* que
`features/build.py`. Es TDD. La primera vez que lo corras contra unas features
ingenuas **debe fallar** — esa falla es la que te ahorra el proyecto. Solo cuando
`build.py` calcula todo de forma estrictamente retrospectiva, pasa.

Va en: fraud-review-queue/tests/test_no_future_leakage.py

--------------------------------------------------------------------------------
La idea (design.md §6.1)
--------------------------------------------------------------------------------
Si una feature es honesta (solo mira hacia atrás), calcularla sobre el histórico
COMPLETO o sobre el histórico TRUNCADO en un tiempo `t` debe dar EXACTAMENTE el
mismo resultado para todas las filas anteriores a `t`. Si no coinciden, alguna
feature está mirando el presente o el futuro.

Contrato de `build_features` (ver features/build.py): devuelve un DataFrame
ordenado por (TransactionDT, TransactionID), índice reseteado, con las columnas
de identidad más `FEATURE_COLUMNS`.
"""

from __future__ import annotations

import pandas as pd

from fraudq.features.build import FEATURE_COLUMNS, build_features


def test_features_do_not_look_forward(sample_df: pd.DataFrame) -> None:
    """Feature honesta = invariante ante truncamiento temporal."""
    cutoff = sample_df["TransactionDT"].quantile(0.6)

    # Features sobre TODO el histórico y sobre el histórico truncado en `cutoff`.
    full = build_features(sample_df)
    truncated = build_features(sample_df[sample_df["TransactionDT"] <= cutoff])

    # `full` está ordenado por (TransactionDT, TransactionID). Nos quedamos con las
    # filas de `full` anteriores o iguales al corte, para compararlas con `truncated`.
    full_past = full[full["TransactionDT"] <= cutoff].reset_index(drop=True)

    # Se compara `TransactionID` junto a las features: si el orden se rompiera, la
    # columna de identidad lo delataría en vez de dejar pasar una comparación de
    # filas distintas que por azar coincidieran. Ambas tablas vienen ordenadas por
    # (TransactionDT, TransactionID) con el índice reseteado, así que ya están
    # alineadas fila a fila.
    cols = ["TransactionID", *FEATURE_COLUMNS]

    # NULL == NULL cuenta como igualdad, que es justo lo que hace
    # `assert_frame_equal` con los NaN. Aquí el nulo no es un dato que falte: es el
    # valor CORRECTO de una feature sin historia previa (la primera transacción de
    # un uid, o una fila sin uid). Tratarlo como desigual haría fallar al test por
    # el comportamiento que precisamente exige el contrato.
    #
    # `check_dtype=False` porque truncar puede cambiar el tipo inferido de una
    # columna entera a float en cuanto aparece un nulo, sin que el valor cambie.
    pd.testing.assert_frame_equal(full_past[cols], truncated[cols], check_dtype=False)


def test_first_txn_per_uid_has_no_history(sample_df: pd.DataFrame) -> None:
    """La PRIMERA transacción de cada UID no tiene pasado que mirar.

    Un segundo guardia, más fino: si una feature retrospectiva tuviera historia en
    la primera transacción de un UID, estaría inventando pasado (o mirando la fila
    actual). Esperado: `uid_txn_count_prior == 0`, y las medias/ratios/z-score son
    NULL (NaN) en esa primera fila.
    """
    feats = build_features(sample_df)
    first_per_uid = feats.sort_values(["TransactionDT", "TransactionID"]).groupby("uid").head(1)

    assert not first_per_uid.empty, "la fixture debe traer al menos un uid"

    # Contar hacia atrás desde la primera transacción da cero, no nulo: hay una
    # respuesta y es "ninguna". Un NaN aquí escondería la diferencia entre
    # "no tiene pasado" y "no se pudo calcular".
    assert (first_per_uid["uid_txn_count_prior"] == 0).all()

    # Las agregaciones sobre un conjunto vacío sí son NULL: no existe la media de
    # cero montos, ni el cociente contra ella.
    assert first_per_uid["uid_amt_prior_mean"].isna().all()
    assert first_per_uid["uid_amt_ratio"].isna().all()

    # `uid_seconds_since_last` viene de LAG sobre la partición del uid: en la
    # primera fila no hay anterior, así que NULL. Un 0 sería peor que inútil,
    # afirmaría que la transacción previa ocurrió en el mismo instante.
    assert first_per_uid["uid_seconds_since_last"].isna().all()

    # `uid_amt_zscore` divide por STDDEV_SAMP del histórico previo, que necesita
    # dos observaciones: NULL en la primera transacción y también en la segunda.
    # Está documentado en build.py y es correcto, no un defecto.
    assert first_per_uid["uid_amt_zscore"].isna().all()
