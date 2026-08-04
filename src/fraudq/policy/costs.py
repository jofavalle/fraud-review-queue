"""La capa de decisión: costo esperado de cada acción y valor de la revisión.

Aquí se convierten en numpy las derivaciones de `docs/design.md` §2. Es el núcleo
del proyecto: todo lo demás (la asignación, la simulación, la API, el simulador de
cola) consume estas cinco funciones.

Contrato, asumido por `allocate.py`, `simulate.py`, `policies.py`, la API y el
Streamlit:

- `cfg` es duck-typed con los atributos `F`, `m`, `phi` y `r`. `CostConfig` de
  `config.py` los expone como alias de sus campos; en los tests basta un
  `SimpleNamespace`.
- Todas las funciones aceptan escalares o arrays de numpy y devuelven lo mismo.
  Vectorizadas con `np.minimum` y `np.where`, sin bucles.
- `cfg` llega SIEMPRE como argumento, nunca de un global. Es el invariante 8 de
  `docs/design.md`: una función que alcanza `CONFIG` por su cuenta ignora en
  silencio el barrido del análisis de sensibilidad, y falla sin levantar nada.

La especificación ejecutable es `tests/test_cost_functions.py`, cuyos casos están
calculados a mano.
"""

from __future__ import annotations

import numpy as np


def cost_approve(p, amt, cfg):
    """Costo esperado de APROBAR la transacción (design.md §2.2, fila 1).

    Si era fraude (probabilidad ``p``), el emisor revierte el cargo: se pierde el
    monto completo más la comisión fija de chargeback. Si era legítima, no cuesta
    nada.

        E[costo | approve] = p * (amt + F)
    """
    return p * (amt + cfg.F)


def cost_block(p, amt, cfg):
    """Costo esperado de BLOQUEAR la transacción (design.md §2.2, fila 2).

    Si era legítima (probabilidad ``1 - p``), se pierde el margen bruto de esa
    venta más el costo de fricción: soporte, y la parte del churn que provoca. Si
    era fraude, bloquear no cuesta nada: se evitó la pérdida.

        E[costo | block] = (1 - p) * (m * amt + phi)
    """
    return (1.0 - p) * (cfg.m * amt + cfg.phi)


def value_of_review(p, amt, cfg):
    """Valor de mandar la transacción a un humano (design.md §2.5).

    La reducción de costo esperado respecto de la MEJOR acción automática,
    asumiendo que la revisión resuelve el caso correctamente a un costo ``r``:

        V = min( cost_approve, cost_block ) - r

    Puede ser NEGATIVO, y ese signo es la tesis entera del proyecto en una
    desigualdad: revisar un caso que el sistema ya tiene decidido cuesta ``r`` y no
    compra nada. ``V`` se maximiza en probabilidades MODERADAS, en ``p_star``, y
    crece con el monto.
    """
    return np.minimum(cost_approve(p, amt, cfg), cost_block(p, amt, cfg)) - cfg.r


def realized_cost(actions, is_fraud, amt, cfg):
    """Costo REALIZADO de cada acción, dado el desenlace verdadero.

    Es la contraparte contable de las funciones de arriba: las esperadas deciden,
    esta liquida. La usa `simulate.py` para evaluar políticas contra etiquetas
    reales (en calibración para fijar umbrales; en test una sola vez).

    | acción    | fraude (y=1)    | legítima (y=0)     |
    |-----------|-----------------|--------------------|
    | approve   | amt + F         | 0                  |
    | block     | 0               | m*amt + phi        |
    | review    | r               | r                  |

    Revisar cuesta ``r`` pase lo que pase: el supuesto de §2.2 es que el analista
    resuelve el caso, así que no se incurre en la pérdida de ninguna de las dos
    columnas.

    Parameters
    ----------
    actions:
        Array de strings con valores exactos "approve", "review" o "block".
    is_fraud:
        Etiqueta verdadera, 0 o 1, alineada con ``actions``.
    amt:
        Montos, alineados con ``actions``.

    Returns
    -------
    Array de floats del mismo largo.

    Raises
    ------
    ValueError
        Si aparece una acción que no es una de las tres. Un typo silencioso aquí
        contaminaría la contabilidad de una política entera.
    """
    actions = np.asarray(actions, dtype=object)
    y = np.asarray(is_fraud, dtype=float)
    amt = np.asarray(amt, dtype=float)

    is_approve = actions == "approve"
    is_block = actions == "block"
    is_review = actions == "review"

    unknown = ~(is_approve | is_block | is_review)
    if np.any(unknown):
        raise ValueError(f"Acciones desconocidas: {sorted(set(np.asarray(actions)[unknown]))}")

    return np.where(
        is_approve,
        y * (amt + cfg.F),
        np.where(is_block, (1.0 - y) * (cfg.m * amt + cfg.phi), float(cfg.r)),
    )


def p_star(amt, cfg):
    """La probabilidad donde ``value_of_review`` es máxima (design.md §2.6).

    ``V`` es el mínimo entre una recta creciente en ``p`` (aprobar) y una
    decreciente (bloquear), menos una constante. El mínimo de las dos se maximiza
    justo donde se cruzan:

        p * (a + F) = (1 - p) * (m*a + phi)
        p_star = (m*a + phi) / ( (a + F) + (m*a + phi) )

    Ronda 0.2-0.3 y depende poco del monto, que es la razón por la que rankear la
    cola por score concentra a los analistas donde menos aportan. Ningún consumidor
    de producción la necesita: la usa el notebook de resultados para anotar la
    figura de regiones de decisión.
    """
    blocked = cfg.m * amt + cfg.phi
    return blocked / ((amt + cfg.F) + blocked)
