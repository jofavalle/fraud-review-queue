"""API de scoring (FastAPI) — boilerplate delegable (plan §5.3, design.md §11.1).

Va en: fraud-review-queue/src/fraudq/api/main.py

    uvicorn fraudq.api.main:app --host 0.0.0.0 --port 8000

## El contrato del endpoint /score

Recibe una transacción (campos crudos; los que falten entran como NaN —
LightGBM los maneja nativamente y la ausencia es señal, §3.2). Devuelve:

    probability         — p CALIBRADA (el booster + el calibrador del Día 5)
    value_of_review     — V con los parámetros de costo del artefacto
    recommended_action  — approve / review / block, SIN restricción de capacidad
    queue_position      — null en este endpoint (ver nota)

**Nota de honestidad sobre `queue_position`:** la cola es un concepto de LOTE
(la capacidad se asigna por jornada contra las demás transacciones del día).
Un endpoint de una sola transacción no tiene cohorte contra la cual rankear,
así que devuelve null en vez de un número inventado. La acción recomendada es
la política SIN capacidad (§2.4): review si V > 0, si no la auto más barata.
La asignación bajo capacidad vive en policy/allocate.py y en el simulador.

**Límite conocido (al README):** las features retrospectivas de UID exigen un
feature store en producción; aquí se aceptan como campos opcionales del payload
(p. ej. `uid_txn_count_prior`). Si no vienen, entran como NaN — igual que una
transacción de un cliente nunca visto.

El modelo se carga UNA vez (lifespan), desde $FRAUDQ_MODELS_DIR (default
`models/artifacts`). Si no hay artefactos, la API arranca igual y /score
responde 503 — así los tests inyectan un scorer falso sin tocar disco.
"""

from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from fraudq.policy.costs import cost_approve, cost_block, value_of_review

DEFAULT_MODELS_DIR = "models/artifacts"


class Scorer:
    """Envuelve los artefactos persistidos. Duck-typed para los tests:
    cualquier objeto con `.predict_p(payload) -> float` y `.cost_cfg` sirve."""

    def __init__(self, booster, calibrator, feature_cols, cost_cfg):
        self._booster = booster
        self._calibrator = calibrator
        self._feature_cols = list(feature_cols)
        self.cost_cfg = cost_cfg

    @classmethod
    def from_dir(cls, dir_path) -> "Scorer":
        from fraudq.models.persist import load_artifacts
        return cls(*load_artifacts(dir_path))

    def predict_p(self, payload: dict) -> float:
        import numpy as np
        import pandas as pd

        row = {c: payload.get(c, np.nan) for c in self._feature_cols}
        score = float(self._booster.predict(pd.DataFrame([row]))[0])
        return float(self._calibrator.predict(np.array([score]))[0])


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = os.environ.get("FRAUDQ_MODELS_DIR", DEFAULT_MODELS_DIR)
    try:
        app.state.scorer = Scorer.from_dir(models_dir)
    except (FileNotFoundError, ImportError):
        # Sin artefactos la API vive (para tests y para el health check del
        # contenedor), pero /score lo dice claramente con un 503.
        app.state.scorer = getattr(app.state, "scorer", None)
    yield


app = FastAPI(
    title="fraud-review-queue scoring API",
    description="Calibrated fraud probability + expected-cost recommendation.",
    lifespan=lifespan,
)


class ScoreRequest(BaseModel):
    """La transacción cruda. Campos extra (C1..., V1..., uid_*) se aceptan y
    fluyen al modelo; los ausentes entran como NaN."""

    model_config = ConfigDict(extra="allow")

    TransactionAmt: float = Field(gt=0, description="Monto de la transacción")
    TransactionDT: int | None = Field(default=None, ge=0)


class ScoreResponse(BaseModel):
    probability: float = Field(ge=0, le=1, description="p calibrada")
    value_of_review: float
    recommended_action: str  # approve | review | block
    queue_position: int | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    scorer = getattr(app.state, "scorer", None)
    if scorer is None:
        raise HTTPException(
            status_code=503,
            detail="No model artifacts loaded (set FRAUDQ_MODELS_DIR).",
        )

    payload = req.model_dump()
    p = float(scorer.predict_p(payload))
    if not (0.0 <= p <= 1.0) or math.isnan(p):
        # La misma guarda que simulate_queue: sin p válida, V es ficción.
        raise HTTPException(status_code=500, detail=f"Invalid probability: {p}")

    amt = float(req.TransactionAmt)
    cfg = scorer.cost_cfg
    v = float(value_of_review(p, amt, cfg))
    if v > 0:
        action = "review"
    elif float(cost_approve(p, amt, cfg)) <= float(cost_block(p, amt, cfg)):
        action = "approve"
    else:
        action = "block"

    return ScoreResponse(
        probability=p,
        value_of_review=v,
        recommended_action=action,
        queue_position=None,
    )
