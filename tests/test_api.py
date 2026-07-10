import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

VALID_PAYLOAD = {
    "LIMIT_BAL": 200000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 50000, "BILL_AMT2": 48000, "BILL_AMT3": 46000,
    "BILL_AMT4": 44000, "BILL_AMT5": 42000, "BILL_AMT6": 40000,
    "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
    "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000,
}


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_returns_valid_response_shape(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["default_prediction"] in (0, 1)
    assert body["risk_tier"] in ("low", "medium", "high", "very_high")


def test_predict_rejects_missing_fields(client):
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "LIMIT_BAL"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_low_risk_profile_scores_lower_than_high_risk_profile(client):
    low_risk = dict(VALID_PAYLOAD)
    low_risk.update({"PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
                      "LIMIT_BAL": 500000})

    high_risk = dict(VALID_PAYLOAD)
    high_risk.update({"PAY_0": 4, "PAY_2": 3, "PAY_3": 3, "PAY_4": 2, "PAY_5": 2, "PAY_6": 2,
                       "LIMIT_BAL": 20000})

    low_prob = client.post("/predict", json=low_risk).json()["default_probability"]
    high_prob = client.post("/predict", json=high_risk).json()["default_probability"]
    assert low_prob < high_prob
