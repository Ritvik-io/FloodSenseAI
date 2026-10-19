import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "FloodSense AI API"
    assert "cuda_available" in data
    assert "compute_device" in data


def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "unet_sar" in data["models"]
    assert "baseline_threshold" in data["models"]
