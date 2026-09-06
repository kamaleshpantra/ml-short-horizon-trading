import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "XGBoostMarketPredictor"
    assert "mid_price" in data["feature_list"]
    assert data["status"] == "active"


def test_predict_endpoint_success(client):
    payload = {
        "mid_price": 100.5,
        "spread": 0.02,
        "relative_spread": 0.0002,
        "obi_1": 0.35,
        "obi_3": 0.25,
        "obi_5": 0.20,
        "obi_10": 0.15,
        "microprice": 100.52,
        "bid_depth_5": 50.0,
        "ask_depth_5": 30.0,
        "depth_ratio_5": 1.6667,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in ["UP", "HOLD", "DOWN"]
    assert 0.0 <= data["probability_up"] <= 1.0
    assert 0.0 <= data["probability_hold"] <= 1.0
    assert 0.0 <= data["probability_down"] <= 1.0
    assert pytest.approx(data["probability_up"] + data["probability_hold"] + data["probability_down"]) == 1.0
    assert "model_version" in data


def test_predict_endpoint_validation_error(client):
    # Missing required field mid_price
    payload = {
        "spread": 0.02,
        "obi_1": 0.35,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
