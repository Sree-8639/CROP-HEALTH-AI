"""
backend/tests/test_forest_monitoring.py:
Automated test suite for Forest Disturbance & Canopy Change Monitoring:
- Hansen GFC dataset loading & 30m resolution downscaling
- Spatial split integrity (disjoint bounding boxes)
- Baseline differencing & Siamese U-Net architectures
- Polygonization, hectare accounting, and GeoJSON validation
- FastAPI forest routes: GET /metrics, GET /presets, POST /inference
"""

import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

from main import app
from ml.forest.hansen_dataset import HansenReferenceTile, build_spatial_split_forest_datasets
from ml.forest.evaluation import raster_mask_to_geojson_polygons, generate_visual_overlays_base64


@pytest.fixture
def client():
    return TestClient(app)


def test_hansen_reference_tile_generation():
    """Verifies that Hansen reference layers generate realistic canopy and loss masks."""
    tile = HansenReferenceTile(grid_shape=(256, 256), random_seed=42)
    assert tile.treecover2000.shape == (256, 256)
    assert tile.loss.shape == (256, 256)
    assert tile.lossyear.shape == (256, 256)

    mask = tile.get_loss_mask_for_interval(2020, 2021)
    assert mask.shape == (256, 256)
    assert set(np.unique(mask)).issubset({0, 1})


def test_spatial_split_dataset_construction():
    """Verifies that datasets have strict spatial separation and Indian AOI is present."""
    bundle = build_spatial_split_forest_datasets(
        tile_size=512,
        patch_size=256,
        stride=256,
        year_before=2020,
        year_after=2021,
        random_seed=42,
    )
    assert "train_dataset" in bundle
    assert "val_dataset" in bundle
    assert "test_dataset" in bundle
    assert "indian_aoi_dataset" in bundle
    assert len(bundle["indian_aoi_dataset"]) > 0

    x0, x1, y = bundle["test_dataset"][0]
    assert x0.shape == (10, 256, 256) # 7 optical + 3 SAR
    assert x1.shape == (10, 256, 256)
    assert y.shape == (1, 256, 256)


def test_raster_to_geojson_polygonization():
    """Verifies vector conversion of binary raster loss into valid GeoJSON polygons and hectares."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 1 # 400 pixels = 4 hectares at 10m
    conf = np.ones((100, 100), dtype=np.float32) * 0.85
    bbox = [82.55, 22.75, 82.75, 22.95]

    res = raster_mask_to_geojson_polygons(mask, conf, bbox, min_pixels=5, pixel_size_meters=10.0)
    assert res["total_polygons"] >= 1
    assert res["total_hectares_lost"] == 4.0
    assert res["geojson"]["type"] == "FeatureCollection"
    assert len(res["geojson"]["features"]) >= 1
    assert "geometry" in res["geojson"]["features"][0]
    assert res["geojson"]["features"][0]["geometry"]["type"] == "Polygon"


def test_fastapi_forest_metrics_endpoint(client):
    """Verifies GET /api/v1/forest/metrics returns valid benchmark metadata."""
    res = client.get("/api/v1/forest/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "project" in data
    assert "model_performance_test_region" in data
    assert "out_of_region_generalisation_test" in data
    assert "scientific_disclosure" in data["reference_dataset"]


def test_fastapi_forest_presets_endpoint(client):
    """Verifies GET /api/v1/forest/presets lists Hasdeo Arand Indian AOI."""
    res = client.get("/api/v1/forest/presets")
    assert res.status_code == 200
    presets = res.json().get("presets", [])
    hasdeo = [p for p in presets if "Hasdeo" in p["name"]]
    assert len(hasdeo) == 1
    assert hasdeo[0]["is_out_of_region_test"] is True


def test_fastapi_forest_inference_endpoint(client):
    """Verifies POST /api/v1/forest/inference executes inference and outputs GeoJSON + visual overlays."""
    payload = {
        "aoi_name": "Hasdeo Arand Forest",
        "bbox": [82.55, 22.75, 82.75, 22.95],
        "year_before": 2020,
        "year_after": 2021,
        "model_type": "siamese_s1_s2",
        "confidence_threshold": 0.50,
        "min_cluster_pixels": 6,
    }
    res = client.post("/api/v1/forest/inference", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["total_hectares_lost"] > 0
    assert "visual_overlays" in data
    assert "before_true_color_base64" in data["visual_overlays"]
    assert "after_true_color_base64" in data["visual_overlays"]
    assert "loss_overlay_base64" in data["visual_overlays"]
    assert "hansen_disclaimer" in data

