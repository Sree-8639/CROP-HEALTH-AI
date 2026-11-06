"""
Unit tests for the three-module architecture:
/api/crop, /api/forest, /api/heat
Verifies:
- Separate FastAPI routers per module under /api/crop, /api/forest, /api/heat
- Common data-access layer sharing
- Deforestation endpoints return 501 Not Implemented stubs
- Heat endpoints return 501 Not Implemented stubs
- Existing crop endpoints continue functioning
"""

import pytest
from fastapi.testclient import TestClient
from main import app
from services.data_access import data_access_layer

client = TestClient(app)


def test_shared_data_access_layer():
    """Verifies that the common data-access layer provides shared resources."""
    assert data_access_layer is not None
    assert hasattr(data_access_layer, "search_sentinel2")
    assert hasattr(data_access_layer, "fetch_sentinel2_bands")
    assert hasattr(data_access_layer, "search_sentinel1")
    assert hasattr(data_access_layer, "fetch_sentinel1_bands")
    assert hasattr(data_access_layer, "get_canonical_grid")
    assert hasattr(data_access_layer, "cache")


def test_crop_module_status():
    """Verifies /api/crop/status returns operational status."""
    response = client.get("/api/crop/status")
    assert response.status_code == 200
    data = response.json()
    assert data["module"] == "crop"
    assert data["status"] == "online"
    assert data["data_access_layer"] == "connected"


def test_crop_overview():
    """Verifies /api/crop/overview returns registered agricultural fields."""
    response = client.get("/api/crop/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "fields" in data
    assert len(data["fields"]) >= 1


def test_deforestation_module_operational():
    """Verifies that deforestation endpoints are operational and return valid GeoJSON inference."""
    # Status
    status_resp = client.get("/api/forest/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["module"] == "forest"
    assert status_resp.json()["status"] == "operational"

    # POST /api/forest/analyze returns 200 with polygons and hectares
    analyze_resp = client.post("/api/forest/analyze", json={
        "bbox": [82.55, 22.75, 82.75, 22.95],
        "year_before": 2020,
        "year_after": 2021,
    })
    assert analyze_resp.status_code == 200
    data = analyze_resp.json()
    assert data["status"] == "success"
    assert data["total_hectares_lost"] > 0
    assert "geojson" in data

    # POST /api/forest/inference returns 200 with visual overlays
    inference_resp = client.post("/api/forest/inference", json={
        "aoi_name": "Hasdeo Arand Forest",
        "bbox": [82.55, 22.75, 82.75, 22.95],
        "year_before": 2020,
        "year_after": 2021,
    })
    assert inference_resp.status_code == 200
    inf_data = inference_resp.json()
    assert "visual_overlays" in inf_data
    assert "before_true_color_base64" in inf_data["visual_overlays"]
    assert "after_true_color_base64" in inf_data["visual_overlays"]
    # GET /api/forest/metrics returns 200 with benchmarks
    metrics_resp = client.get("/api/forest/metrics")
    assert metrics_resp.status_code == 200
    assert "model_performance_test_region" in metrics_resp.json()


def test_heat_module_operational():
    """Verifies that heat endpoints are fully operational and return 200 with real downscaled data."""
    # Status
    status_resp = client.get("/api/heat/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["module"] == "heat"
    assert status_resp.json()["status"] == "operational"
    assert status_resp.json()["implemented"] is True

    # POST /api/heat/analyze must return 200 with downscaled statistics
    analyze_resp = client.post("/api/heat/analyze", json={
        "city_name": "Visakhapatnam",
        "bbox": [83.15, 17.62, 83.35, 17.80],
    })
    assert analyze_resp.status_code == 200
    data = analyze_resp.json()
    assert data["status"] == "success"
    assert "summary_statistics" in data
    assert "downscaling_accuracy" in data

    # POST /api/heat/downscale must return 200
    downscale_resp = client.post("/api/heat/downscale", json={})
    assert downscale_resp.status_code == 200

    # GET /api/heat/metrics must return 200
    metrics_resp = client.get("/api/heat/metrics")
    assert metrics_resp.status_code == 200
    assert "downscaling_evaluation_spatial_split" in metrics_resp.json()


def test_existing_crop_pages_endpoints_preserved():
    """Verifies that existing endpoints remain operational."""
    # Health check
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200

    # Farmer dashboard overview
    farmer_resp = client.get("/api/v1/farmer-dashboard/overview")
    assert farmer_resp.status_code == 200

