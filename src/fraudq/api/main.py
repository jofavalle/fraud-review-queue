"""Scoring API (FastAPI).

    uvicorn fraudq.api.main:app --host 0.0.0.0 --port 8000

## The contract of the /score endpoint

It takes one transaction (raw fields; whatever is missing enters as NaN, since
LightGBM handles that natively and absence is itself a signal, §3.2). It
returns:

    probability         CALIBRATED p (the booster plus the calibrator)
    value_of_review     V under the cost parameters stored in the artefact
    recommended_action  approve / review / block, with NO capacity constraint
    queue_position      null on this endpoint (see the note)

**An honest note on `queue_position`:** the queue is a BATCH concept, since
capacity is allocated per day against the other transactions of that day. A
single-transaction endpoint has no cohort to rank against, so it returns null
rather than an invented number. The recommended action is the policy WITHOUT
capacity (§2.4): review if V > 0, otherwise the cheaper automatic action.
Allocation under capacity lives in policy/allocate.py and in the simulator.

**A known limit:** the backward-looking UID features need a feature store in
production. Here they are accepted as optional payload fields (say
`uid_txn_count_prior`). If they do not arrive they enter as NaN, exactly like a
transaction from a customer never seen before.

The model is loaded ONCE, on lifespan, from $FRAUDQ_MODELS_DIR (default
`models/artifacts`). With no artefacts the API still starts and /score answers
503, which is how the tests inject a fake scorer without touching disk.
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
    """Wraps the persisted artefacts. Duck-typed for the tests: any object with
    `.predict_p(payload) -> float` and `.cost_cfg` will do."""

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
        # With no artefacts the API still lives, for the tests and for the
        # container health check, but /score says so plainly with a 503.
        app.state.scorer = getattr(app.state, "scorer", None)
    yield


app = FastAPI(
    title="fraud-review-queue scoring API",
    description="Calibrated fraud probability + expected-cost recommendation.",
    lifespan=lifespan,
)


class ScoreRequest(BaseModel):
    """The raw transaction. Extra fields (C1..., V1..., uid_*) are accepted and
    flow through to the model; the missing ones enter as NaN."""

    model_config = ConfigDict(extra="allow")

    TransactionAmt: float = Field(gt=0, description="Transaction amount")
    TransactionDT: int | None = Field(default=None, ge=0)


class ScoreResponse(BaseModel):
    probability: float = Field(ge=0, le=1, description="Calibrated p")
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
        # The same guard as simulate_queue: without a valid p, V is fiction.
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
