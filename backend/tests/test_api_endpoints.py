"""
Integration tests for FastAPI inference and metadata endpoints
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_active_model_endpoint():
    response = client.get("/model/active")
    assert response.status_code == 200
    data = response.json()
    assert data["model_architecture"] == "UNet"
    assert "normalization_statistics" in data
    assert "vv_mean" in data["normalization_statistics"]
    assert "vh_mean" in data["normalization_statistics"]
    assert data["checkpoint_epoch"] == 8
    assert "validation_iou" in data


def test_predict_sample_endpoint():
    # Use existing sample chip
    response = client.post("/predict/sample/USA_1010394?threshold=0.5")
    assert response.status_code == 200
    data = response.json()
    assert data["chip_id"] == "USA_1010394"
    assert "inundated_area_km2" in data
    assert "flood_pixel_count" in data
    assert "bounds" in data
    assert "crs" in data
    assert data["valid_pixel_count"] > 0


def test_metrics_reports_endpoint():
    response = client.get("/metrics/reports")
    assert response.status_code == 200
    data = response.json()
    assert "validation_split" in data
    assert "bolivia_heldout_split" in data
    assert data["validation_split"]["metrics"]["iou"] > 0.0
    assert data["bolivia_heldout_split"]["metrics"]["iou"] > 0.0
