"""Capa de decisión: costos esperados, asignación bajo capacidad, simulación.

El corazón del proyecto (design.md §8). costs.py y allocate.py los escribes
TÚ (plan §5.3); simulate.py es el harness contable.
"""

from fraudq.policy.allocate import allocate_day
from fraudq.policy.costs import (
    cost_approve,
    cost_block,
    realized_cost,
    value_of_review,
)
from fraudq.policy.simulate import QueueResult, daily_capacity, simulate_queue

__all__ = [
    "allocate_day",
    "cost_approve",
    "cost_block",
    "realized_cost",
    "value_of_review",
    "QueueResult",
    "daily_capacity",
    "simulate_queue",
]
