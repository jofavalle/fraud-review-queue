"""Las cuatro políticas de comparación (design.md §8.2) — completo (delegable).

Va en: fraud-review-queue/src/fraudq/evaluate/policies.py

| # | Política           | Qué representa                                  |
|---|--------------------|--------------------------------------------------|
| 1 | approve_all        | La pérdida basal. El "no hacer nada".            |
| 2 | single_threshold   | El sistema ingenuo: bloquea si p >= t.           |
| 3 | topk_by_score      | Lo que hace la mayoría: revisa el top-K por p.   |
| 4 | topk_by_value      | TU política (allocate_day): top-K por V.         |

## La decisión de diseño que hay que poder defender

Las políticas 3 y 4 comparten la MISMA regla automática para lo no revisado
(la acción de menor costo esperado). Así la única variable entre ambas es EL
RANKING de la cola — que es la tesis. Si a la 3 se le diera además una regla
automática tonta, la diferencia mezclaría dos efectos y el número del README
estaría inflado. Elegir la comparación conservadora es lo que la hace creíble.
(Va directo a tu registro de decisiones §III.)

El umbral de la política 2 se ajusta en CALIB (§4.1: "fijar umbrales"), jamás
en test. `compare_policies` recibe el umbral ya fijado.

## El protocolo de la única mirada

`compare_policies(parts["test"], ...)` se ejecuta UNA vez, el Día 6, con los
parámetros base de config.py, y el resultado se guarda en disco
(reports/policy_comparison.csv + reports/scored_test.parquet). El tornado del
Día 7 y el Streamlit del Día 8 REUSAN ese scoring persistido: varían los
parámetros de COSTO sobre predicciones ya hechas — no vuelven a mirar el test
para decidir nada del modelo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudq.policy.allocate import allocate_day
from fraudq.policy.costs import cost_approve, cost_block, realized_cost
from fraudq.policy.simulate import QueueResult, simulate_queue

POLICY_ORDER = ("approve_all", "single_threshold", "topk_by_score", "topk_by_value")


# ------------------------------------------------------------- las políticas

def actions_approve_all(p, amt, capacity, cfg) -> np.ndarray:
    """Política 1: aprobar todo. La pérdida basal contra la que todo se mide."""
    return np.full(len(np.asarray(p)), "approve", dtype=object)


def make_actions_single_threshold(threshold: float):
    """Política 2: bloquear si p >= t, aprobar si no. Sin cola de revisión.

    Es el "sistema de un solo umbral sobre el score" del §2.4 — la política
    estructuralmente equivocada porque el umbral óptimo depende del monto.
    """
    def actions_single_threshold(p, amt, capacity, cfg) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return np.where(p >= threshold, "block", "approve").astype(object)

    return actions_single_threshold


def actions_topk_by_score(p, amt, capacity, cfg) -> np.ndarray:
    """Política 3: revisar el top-K por SCORE; el resto, la mejor acción auto.

    La implementación ingenua de la cola: manda a los analistas a p > 0.9,
    donde ya estás seguro y un humano no aporta información (§2.6). No filtra
    por V > 0 — la política ingenua no sabe qué es V; gastar capacidad en
    casos ya decididos ES su defecto, y se simula tal cual.
    """
    p = np.asarray(p, dtype=float)
    amt = np.asarray(amt, dtype=float)
    order = np.argsort(-p, kind="stable")          # determinista ante empates
    to_review = order[: max(int(capacity), 0)]

    approve_cheaper = cost_approve(p, amt, cfg) <= cost_block(p, amt, cfg)
    actions = np.where(approve_cheaper, "approve", "block").astype(object)
    actions[to_review] = "review"
    return actions


def actions_topk_by_value(p, amt, capacity, cfg) -> np.ndarray:
    """Política 4: TU política. Delega en allocate_day (top-K por V, V > 0)."""
    return allocate_day(p, amt, capacity, cfg)


# ------------------------------------------- ajuste del umbral (en CALIB)

def fit_single_threshold(
    df_calib: pd.DataFrame,
    cfg,
    p_col: str = "p",
    amt_col: str = "TransactionAmt",
    target: str = "isFraud",
    grid: np.ndarray | None = None,
) -> float:
    """Elige el umbral de la política 2 minimizando el costo REALIZADO en calib.

    En calib las etiquetas son legítimamente utilizables (§4.1: la partición
    existe para "fijar umbrales y explorar K"). Vectorizado sin bucle por día:
    la política 2 no tiene capacidad, así que el día no importa.
    Empates: gana el umbral más bajo del grid (primero en orden).
    """
    p = df_calib[p_col].to_numpy(dtype=float)
    amt = df_calib[amt_col].to_numpy(dtype=float)
    y = df_calib[target].to_numpy()
    if grid is None:
        grid = np.linspace(0.005, 0.995, 199)

    costs = np.empty(len(grid))
    for i, t in enumerate(grid):
        actions = np.where(p >= t, "block", "approve").astype(object)
        costs[i] = float(np.sum(realized_cost(actions, y, amt, cfg)))
    return float(grid[int(np.argmin(costs))])


# --------------------------------------------------------- la comparación

def compare_policies(
    df: pd.DataFrame,
    cfg,
    capacity_pct: float,
    threshold: float,
    p_col: str = "p",
    amt_col: str = "TransactionAmt",
    day_col: str = "day",
    target: str = "isFraud",
) -> pd.DataFrame:
    """Corre las 4 políticas sobre `df` y devuelve la tabla del README (§8.2).

    Una fila por política: costo total, costo por $1,000 (LA métrica),
    fraudes atrapados/perdidos, legítimas bloqueadas, reviews y utilización.
    """
    policies = {
        "approve_all": actions_approve_all,
        "single_threshold": make_actions_single_threshold(threshold),
        "topk_by_score": actions_topk_by_score,
        "topk_by_value": actions_topk_by_value,
    }
    rows = {}
    for name in POLICY_ORDER:
        result: QueueResult = simulate_queue(
            df, policies[name], cfg, capacity_pct,
            p_col=p_col, amt_col=amt_col, day_col=day_col, target=target,
        )
        rows[name] = result.summary()

    table = pd.DataFrame(rows).T.loc[list(POLICY_ORDER)]
    table.index.name = "policy"
    return table


def headline_savings(comparison: pd.DataFrame) -> dict:
    """EL número del README: política 4 vs. política 3 (design.md §8.2).

    Lo que deja sobre la mesa una organización que rankea su cola por score.
    Positivo = tu política gana. Si sale chico, NO se tocan los parámetros:
    se activa la contingencia §13.5 y cambia la tesis, no los supuestos.
    """
    cost3 = comparison.loc["topk_by_score", "total_cost"]
    cost4 = comparison.loc["topk_by_value", "total_cost"]
    per1k_3 = comparison.loc["topk_by_score", "cost_per_1k"]
    per1k_4 = comparison.loc["topk_by_value", "cost_per_1k"]
    return {
        "savings_total": float(cost3 - cost4),
        "savings_per_1k": float(per1k_3 - per1k_4),
    }
