"""
================================================================================
  EL NÚCLEO DE ESTE ARCHIVO LO ESCRIBES TÚ.  (plan §5.3 — "Escribo yo")
================================================================================

El análisis de sensibilidad es TU diferenciador (design.md §8.3): vienes de un
campo donde un número sin barra de error es inaceptable. Por eso el barrido —
la parte que convierte supuestos en incertidumbre — es tuyo. Lo que te dejo
hecho es lo delegable: la figura (tornado) y la extracción del ahorro.

Va en: fraud-review-queue/src/fraudq/evaluate/sensitivity.py

--------------------------------------------------------------------------------
Las dos reglas que hacen que esto sea un análisis y no una justificación
--------------------------------------------------------------------------------
1. **Los rangos se leen de `SENSITIVITY_RANGES` en config.py** — fijados ANTES
   de ver los resultados (§13.5). Elegir rangos después de conocer el número
   convierte el tornado en marketing.
2. **El config es frozen; las variantes se crean con `dataclasses.replace`.**
   Tu propia respuesta de entrevista del Día 1: si los parámetros no fluyeran
   como argumento, el barrido daría resultados idénticos para todos los valores
   sin lanzar excepción — y creerías que tu conclusión es más robusta de lo
   que es.

--------------------------------------------------------------------------------
El contrato de `tornado_data` (lo que asume plot_tornado y el notebook)
--------------------------------------------------------------------------------
DataFrame con una fila por parámetro barrido:

    param     | low   | high  | savings_at_low | savings_at_high | swing
    "F"       | 10.0  | 40.0  | ...            | ...             | |alto-bajo|

ordenado por `swing` DESCENDENTE (el tornado se lee de arriba hacia abajo).
`savings_*` es el ahorro por $1,000 de la política 4 sobre la 3 con TODOS los
demás parámetros en su valor base.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd


def savings_per_1k(comparison: pd.DataFrame) -> float:
    """Extrae el ahorro por $1,000 (política 4 vs. 3) de una tabla de compare_policies."""
    return float(
        comparison.loc["topk_by_score", "cost_per_1k"]
        - comparison.loc["topk_by_value", "cost_per_1k"]
    )


def tornado_data(base_cfg, ranges: dict, evaluate_fn) -> pd.DataFrame:
    """Barrido uno-a-la-vez sobre los supuestos de costo (design.md §8.3).

    Parameters
    ----------
    base_cfg:
        TU CostConfig congelado (frozen=True), con los valores base.
    ranges:
        `SENSITIVITY_RANGES` de config.py: {nombre_de_campo: (low, high)}.
        Los nombres son los CAMPOS del dataclass (`chargeback_fee`, ...), no
        los alias — `dataclasses.replace` no conoce propiedades.
    evaluate_fn:
        callable(cfg) -> DataFrame de compare_policies sobre el scoring
        PERSISTIDO del Día 6 (reports/scored_test.parquet). Nota bien: variar
        parámetros de COSTO no re-mira el test — las predicciones ya están
        hechas; cambia solo la capa de decisión.

    Returns
    -------
    El DataFrame del contrato de arriba, ordenado por `swing` descendente.
    """
    rows = []
    for param, (low, high) in ranges.items():
        # `replace` sobre un dataclass frozen devuelve una copia: `base_cfg` no se
        # toca, y cada variante mueve UN parámetro dejando los demás en su base.
        # Eso es lo que hace que el swing sea atribuible a ese parámetro.
        cfg_low = replace(base_cfg, **{param: low})
        cfg_high = replace(base_cfg, **{param: high})

        savings_at_low = savings_per_1k(evaluate_fn(cfg_low))
        savings_at_high = savings_per_1k(evaluate_fn(cfg_high))

        rows.append(
            {
                "param": param,
                "low": low,
                "high": high,
                "savings_at_low": savings_at_low,
                "savings_at_high": savings_at_high,
                # Valor absoluto: lo que mide el tornado es cuánto MUEVE el
                # parámetro la conclusión, no en qué dirección la mueve. La
                # dirección sigue legible en las dos columnas anteriores.
                "swing": abs(savings_at_high - savings_at_low),
            }
        )

    # Descendente: el tornado se lee de arriba hacia abajo, y la barra más larga
    # es el parámetro que el negocio debería medir mejor.
    return (
        pd.DataFrame(rows, columns=["param", "low", "high", "savings_at_low",
                                    "savings_at_high", "swing"])
        .sort_values("swing", ascending=False)
        .reset_index(drop=True)
    )


def sensitivity_grid_2d(base_cfg, param_x: str, values_x, param_y: str,
                        values_y, evaluate_fn) -> pd.DataFrame:
    """[OPCIONAL, §8.3 extensión] Malla 2D: ahorro en función de dos parámetros.

    Devuelve un DataFrame largo (param_x, param_y, savings_per_1k) listo para
    un heatmap donde se colorea la región en la que tu política gana. Es una
    figura preciosa y barata — SOLO si sobra tiempo (orden de sacrificio §13).
    """
    rows = [
        {
            param_x: vx,
            param_y: vy,
            "savings_per_1k": savings_per_1k(
                evaluate_fn(replace(base_cfg, **{param_x: vx, param_y: vy}))
            ),
        }
        for vx in values_x
        for vy in values_y
    ]
    return pd.DataFrame(rows, columns=[param_x, param_y, "savings_per_1k"])


def plot_tornado(tornado_df: pd.DataFrame, base_savings: float, ax=None,
                 param_labels: dict | None = None):
    """El tornado plot (delegable: es una figura). Barras horizontales.

    Cada barra va de `savings_at_low` a `savings_at_high`, ordenadas por
    `swing` (la mayor arriba); la línea vertical es el ahorro con los
    parámetros base. Si TODAS las barras viven a la derecha del cero, la
    conclusión sobrevive el rango completo de supuestos — esa frase, tal
    cual, es la que va en el README.

    matplotlib se importa adentro: el módulo es importable sin él.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.6 * len(tornado_df) + 1.5))

    df = tornado_df.iloc[::-1]                      # la de mayor swing arriba
    labels = [
        (param_labels or {}).get(row["param"], row["param"])
        for _, row in df.iterrows()
    ]
    lo = df[["savings_at_low", "savings_at_high"]].min(axis=1)
    hi = df[["savings_at_low", "savings_at_high"]].max(axis=1)

    ax.barh(labels, hi - lo, left=lo, height=0.6, alpha=0.85)
    ax.axvline(base_savings, linestyle="--", linewidth=1.2, label="base")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Savings of value-ranked queue vs. score-ranked, $ per $1,000")
    ax.legend(loc="lower right")
    ax.figure.tight_layout()
    return ax
