"""Generador de transacciones sintéticas con la forma del esquema real.

Existe para una sola cosa: poder correr el pipeline entero sin haber descargado
los 590k registros de IEEE-CIS. No sustituye a los datos, y ningún resultado que
salga de aquí es un resultado del proyecto. Lo que verifica es que la cadena
--ingesta, features, split, entrenamiento, calibración, asignación y comparación
de políticas-- no se rompe, que es lo que no conviene descubrir en la máquina
donde el entrenamiento cuesta horas.

Reproduce las propiedades del esquema de las que depende el código:

- ``TransactionDT`` en segundos desde un origen sin especificar, no una fecha.
- ``D1`` construido como ``day - D1n`` con ``D1n`` constante por tarjeta, que es
  el supuesto sobre el que `features/uid.py` reconstruye el cliente.
- Fraude poco frecuente y dependiente del monto, para que exista algo que
  aprender y para que la capa de costos tenga un caso no degenerado.
- Filas sin ``addr1``, porque en los datos reales las hay y las features del UID
  tienen que salir nulas ahí.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400

#: Split reducido para el dataset sintético. El de producción (train hasta el día
#: 119, calibración hasta el 155) no cabe en un dataset de pocos días, y forzarlo
#: daría particiones vacías.
SYNTHETIC_SPLIT = {
    "train_end_day": 39,
    "embargo_days": 3,
    "calib_end_day": 59,
    "n_cv_folds": 2,
}


def make_synthetic_transactions(
    n_days: int = 75,
    # El volumen diario no es cosmético: la capacidad de revisión es un
    # porcentaje del volumen del día, y con pocas transacciones
    # `int(n * 0.01)` cae a cero, la cola se queda vacía y las políticas 3 y 4
    # degeneran a la misma. El smoke test pasaría sin ejercitar la asignación,
    # que es justo la pieza que hay que probar.
    txns_per_day: int = 400,
    n_cards: int = 300,
    fraud_base_rate: float = 0.035,
    missing_addr_frac: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera un DataFrame con el esquema que espera el pipeline.

    Returns
    -------
    DataFrame con ``TransactionID``, ``TransactionDT``, ``day``, ``hour``,
    ``card1``, ``addr1``, ``D1``, ``TransactionAmt``, ``P_emaildomain``,
    ``C1``, ``C2``, ``V1`` e ``isFraud``, ordenado por ``TransactionDT``.
    """
    rng = np.random.default_rng(seed)
    n = n_days * txns_per_day

    # Un cliente es (card1, addr1, D1n). D1n constante por tarjeta es lo que hace
    # reconstruible el UID.
    card1 = rng.integers(1000, 1000 + n_cards, size=n)
    card_addr = rng.integers(50, 90, size=n_cards)
    card_d1n = rng.integers(-400, 0, size=n_cards)
    addr1 = card_addr[card1 - 1000].astype(float)
    d1n = card_d1n[card1 - 1000]

    day = np.repeat(np.arange(n_days), txns_per_day)
    second_in_day = rng.integers(0, SECONDS_PER_DAY, size=n)
    dt = day * SECONDS_PER_DAY + second_in_day

    amt = np.round(np.exp(rng.normal(3.6, 1.1, size=n)), 2)

    # Fraude dependiente del monto y de una covariable, para que el modelo tenga
    # señal y la calibración tenga algo que corregir.
    v1 = rng.normal(0.0, 1.0, size=n)
    logit = (
        np.log(fraud_base_rate / (1 - fraud_base_rate))
        + 0.55 * (np.log1p(amt) - np.log1p(amt).mean())
        + 0.9 * v1
    )
    is_fraud = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))

    # Filas sin identidad: sus features del UID deben salir nulas.
    addr1[rng.random(n) < missing_addr_frac] = np.nan

    df = pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1),
            "TransactionDT": dt,
            "day": day,
            "hour": (dt // 3_600) % 24,
            "card1": card1,
            "addr1": addr1,
            "D1": day - d1n,
            "TransactionAmt": amt,
            "P_emaildomain": rng.choice(
                ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com"],
                size=n,
                p=[0.5, 0.25, 0.15, 0.10],
            ),
            "C1": rng.poisson(3.0, size=n).astype(float),
            "C2": rng.poisson(1.5, size=n).astype(float),
            "V1": v1,
            "isFraud": is_fraud,
        }
    )
    return df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
