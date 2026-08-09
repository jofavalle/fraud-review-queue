"""The API test: /score returns 200 with a valid schema.

A FAKE scorer is injected into app.state, so the test needs neither artefacts
on disk nor LightGBM. What it does use for real is the cost layer: the
recommended action of the central case is worked out by hand with F=20,
m=0.25, phi=10, r=2.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx", reason="the FastAPI TestClient uses httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fraudq.api.main import app  # noqa: E402


class StubScorer:
    """A deterministic scorer: fixed p, cost parameters from design.md §2.1."""

    def __init__(self, p: float):
        self._p = p
        self.cost_cfg = SimpleNamespace(F=20.0, m=0.25, phi=10.0, r=2.0)

    def predict_p(self, payload: dict) -> float:
        return self._p


@pytest.fixture
def client_with(monkeypatch):
    def _make(p: float | None) -> TestClient:
        app.state.scorer = None if p is None else StubScorer(p)
        return TestClient(app)

    yield _make
    app.state.scorer = None


def test_score_returns_200_with_valid_schema(client_with):
    """The central case, by hand: p=0.25, amt=200.

    approve = 0.25*220 = 55; block = 0.75*60 = 45; best automatic = 45;
    V = 45 - 2 = 43 > 0  =>  recommended_action = "review".
    """
    resp = client_with(0.25).post("/score", json={"TransactionAmt": 200.0})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"probability", "value_of_review", "recommended_action", "queue_position"}
    assert body["probability"] == pytest.approx(0.25)
    assert body["value_of_review"] == pytest.approx(43.0)
    assert body["recommended_action"] == "review"
    assert body["queue_position"] is None


def test_certain_low_probability_gets_approved(client_with):
    """p=0.001, amt=20: approve = 0.001*40 = 0.04; block = 14.985; V < 0 -> approve."""
    body = client_with(0.001).post("/score", json={"TransactionAmt": 20.0}).json()
    assert body["recommended_action"] == "approve"
    assert body["value_of_review"] < 0


def test_certain_high_probability_gets_blocked(client_with):
    """p=0.99, amt=10: block = 0.01*12.5 = 0.125; V = -1.875 -> block."""
    body = client_with(0.99).post("/score", json={"TransactionAmt": 10.0}).json()
    assert body["recommended_action"] == "block"


def test_extra_fields_are_accepted(client_with):
    """The real payload carries card1, C1..., uid_*: extra='allow' lets them flow."""
    resp = client_with(0.25).post(
        "/score",
        json={
            "TransactionAmt": 80.0,
            "card1": 1000,
            "P_emaildomain": "gmail.com",
            "uid_txn_count_prior": 3,
        },
    )
    assert resp.status_code == 200


def test_invalid_amount_is_rejected(client_with):
    assert client_with(0.25).post("/score", json={"TransactionAmt": -5.0}).status_code == 422
    assert client_with(0.25).post("/score", json={}).status_code == 422


def test_without_artifacts_returns_503(client_with):
    resp = client_with(None).post("/score", json={"TransactionAmt": 50.0})
    assert resp.status_code == 503


def test_health_endpoint(client_with):
    assert client_with(None).get("/health").json() == {"status": "ok"}
