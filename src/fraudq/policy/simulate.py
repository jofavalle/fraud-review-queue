"""Simulación de la cola por día — completo (columna delegable del plan §5.3).

Va en: fraud-review-queue/src/fraudq/policy/simulate.py

Harness mecánico: agrupa por día, calcula la capacidad de la jornada, delega la
decisión en una política (`choose_actions`) y liquida el costo realizado con
TUS funciones de costs.py. La asignación es POR DÍA porque la capacidad de los
analistas se renueva cada jornada (design.md §8.1) — una cola global usaría
hoy analistas de la semana pasada.

El kernel intelectual NO está aquí: está en costs.py y allocate.py (tuyos).
Este módulo no sabe qué es un fraude — solo contabiliza.

## Interfaz de política

    choose_actions(p, amt, capacity, cfg) -> np.ndarray de
    "approve" | "review" | "block"

`allocate_day` (tuya) satisface esa firma; las políticas de comparación del
Día 6 (evaluate/policies.py) también.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudq.policy.costs import realized_cost

_VALID_ACTIONS = frozenset({"approve", "review", "block"})


def daily_capacity(n_transactions: int, capacity_pct: float) -> int:
    """Cupo de revisión de una jornada: floor(pct * volumen del día).

    `capacity_pct` viene de tu PolicyConfig (0.005-0.02 en el diseño §2.3).
    floor y no round: la capacidad es gente real — no existe medio analista.
    Puede dar 0 en días chicos con pct bajo; eso es correcto, no un bug.
    """
    if not 0.0 <= capacity_pct <= 1.0:
        raise ValueError(f"capacity_pct fuera de [0,1]: {capacity_pct}.")
    return int(np.floor(n_transactions * capacity_pct))


@dataclass
class QueueResult:
    """Resultado de simular una política sobre una partición completa."""

    actions: pd.Series  # acción por transacción, alineada al df de entrada
    per_day: pd.DataFrame  # una fila por día: costos, conteos, capacidad

    # ------------------------------------------------------------- agregados
    @property
    def total_cost(self) -> float:
        return float(self.per_day["cost"].sum())

    @property
    def total_volume(self) -> float:
        return float(self.per_day["volume"].sum())

    @property
    def cost_per_1k(self) -> float:
        """LA métrica principal (§8.2): pérdida por cada $1,000 transaccionados."""
        return self.total_cost / self.total_volume * 1_000.0

    @property
    def frauds_caught(self) -> int:
        return int(self.per_day["frauds_caught"].sum())

    @property
    def frauds_missed(self) -> int:
        return int(self.per_day["frauds_missed"].sum())

    @property
    def legit_blocked(self) -> int:
        return int(self.per_day["legit_blocked"].sum())

    @property
    def reviews(self) -> int:
        return int(self.per_day["reviews"].sum())

    @property
    def capacity(self) -> int:
        return int(self.per_day["capacity"].sum())

    @property
    def utilization(self) -> float:
        """Reviews usadas / capacidad total. NaN si la capacidad fue 0."""
        return self.reviews / self.capacity if self.capacity else float("nan")

    def summary(self) -> dict:
        return {
            "total_cost": self.total_cost,
            "cost_per_1k": self.cost_per_1k,
            "frauds_caught": self.frauds_caught,
            "frauds_missed": self.frauds_missed,
            "legit_blocked": self.legit_blocked,
            "reviews": self.reviews,
            "capacity": self.capacity,
            "utilization": self.utilization,
        }


def simulate_queue(
    df: pd.DataFrame,
    choose_actions,
    cfg,
    capacity_pct: float,
    p_col: str = "p",
    amt_col: str = "TransactionAmt",
    day_col: str = "day",
    target: str = "isFraud",
) -> QueueResult:
    """Corre una política día a día sobre `df` y liquida el costo realizado.

    `df` necesita: `p_col` (probabilidad CALIBRADA del Día 5), `amt_col`,
    `day_col`, `target`. Se recorren los días en orden; la capacidad de cada
    jornada es `daily_capacity(n_día, capacity_pct)`.
    """
    missing = [c for c in (p_col, amt_col, day_col, target) if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas para simular: {missing}.")

    p_all = df[p_col].to_numpy(dtype=float)
    if np.any(np.isnan(p_all)) or p_all.min() < 0.0 or p_all.max() > 1.0:
        # p inválida => V inválido => toda la política es ficción. Mejor morir aquí.
        raise ValueError(
            f"'{p_col}' no es una probabilidad válida (rango "
            f"[{np.nanmin(p_all):.4f}, {np.nanmax(p_all):.4f}], "
            f"NaN={int(np.isnan(p_all).sum())}). ¿Calibraste (Día 5)?"
        )

    actions_all = pd.Series(index=df.index, dtype=object)
    day_rows = []

    for day_value, g in df.groupby(day_col, sort=True):
        p = g[p_col].to_numpy(dtype=float)
        amt = g[amt_col].to_numpy(dtype=float)
        y = g[target].to_numpy()
        capacity = daily_capacity(len(g), capacity_pct)

        actions = np.asarray(choose_actions(p, amt, capacity, cfg))
        if actions.shape != p.shape:
            raise ValueError(
                f"La política devolvió {actions.shape} acciones para "
                f"{p.shape} transacciones (día {day_value})."
            )
        bad = set(np.unique(actions)) - _VALID_ACTIONS
        if bad:
            raise ValueError(f"Acciones desconocidas de la política: {sorted(bad)}.")

        costs = np.asarray(realized_cost(actions, y, amt, cfg), dtype=float)
        is_fraud = y == 1
        reviewed = actions == "review"
        blocked = actions == "block"
        approved = actions == "approve"

        day_rows.append(
            {
                "day": day_value,
                "n": len(g),
                "capacity": capacity,
                "reviews": int(reviewed.sum()),
                "cost": float(costs.sum()),
                "volume": float(amt.sum()),
                # la revisión resuelve el caso correctamente (§2.2): un fraude
                # revisado cuenta como atrapado.
                "frauds_caught": int(((blocked | reviewed) & is_fraud).sum()),
                "frauds_missed": int((approved & is_fraud).sum()),
                "legit_blocked": int((blocked & ~is_fraud).sum()),
            }
        )
        actions_all.loc[g.index] = actions

    return QueueResult(actions=actions_all, per_day=pd.DataFrame(day_rows))
