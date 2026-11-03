import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

from services.sentinel1_service import (
    Sentinel1Service,
    sentinel1_service,
    InvalidSARGeometryError,
    InvalidSARDatesError,
    NoSARDataFoundError,
)
from services.fusion_service import (
    MultimodalFusionService,
    multimodal_fusion_service,
    MultimodalFusionPreparationError,
)
from services.raster_utils import RasterAlignmentError
from main import app

client = TestClient(app)

@pytest.fixture
def sample_geometry():
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [81.52, 16.81],
                [81.54, 16.81],
                [81.54, 16.83],
                [81.52, 16.83],
                [81.52, 16.81],
            ]
        ],
    }

# -------------------------------------------------------------------------
# Input Validation Tests
# -------------------------------------------------------------------------

def test_validate_inputs_invalid_geom():
    service = Sentinel1Service()
    with pytest.raises(InvalidSARGeometryError):
        service.validate_inputs({}, "2026-10-01", "2026-10-25")

    with pytest.raises(InvalidSARGeometryError):
        service.validate_inputs({"type": "Point", "coordinates": [81.5, 16.8]}, "2026-10-01", "2026-10-25")

def test_validate_inputs_invalid_dates(sample_geometry):
    service = Sentinel1Service()
    with pytest.raises(InvalidSARDatesError):
        service.validate_inputs(sample_geometry, "invalid-date", "2026-10-25")

    with pytest.raises(InvalidSARDatesError):
        service.validate_inputs(sample_geometry, "2026-10-26", "2026-10-01")

# -------------------------------------------------------------------------
# Sentinel-1 Search & Catalog Tests
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_imagery_demo_catalog(sample_geometry):
    service = Sentinel1Service()
    results = await service.search_imagery(sample_geometry, "2026-09-01", "2026-10-25")

    assert len(results) > 0
    first = results[0]
    assert "scene_id" in first
    assert "acquisition_date" in first
    assert first["product_type"] == "Sentinel-1 GRD"
    assert "VV" in first["polarization"]
    assert "VH" in first["polarization"]
    assert first["is_demo"] is True

# -------------------------------------------------------------------------
# Temporal Matching Tests
# -------------------------------------------------------------------------

def test_temporal_matching_found():
    service = Sentinel1Service()
    catalog = [
        {"acquisition_date": "2026-10-10T00:38:15Z", "scene_id": "S1_1"},
        {"acquisition_date": "2026-10-21T00:38:15Z", "scene_id": "S1_2"},
        {"acquisition_date": "2026-10-29T00:38:15Z", "scene_id": "S1_3"},
    ]

    # Target S2 observation: 2026-10-20 (closest is 2026-10-21, 1 day difference)
    match = service.find_closest_sentinel1_observation("2026-10-20", catalog, max_window_days=7)
    assert match["matched"] is True
    assert match["sentinel2_date"] == "2026-10-20"
    assert match["sentinel1_date"] == "2026-10-21"
    assert match["date_difference_days"] == 1
    assert match["scene_id"] == "S1_2"

def test_temporal_matching_unavailable_when_outside_window():
    service = Sentinel1Service()
    catalog = [
        {"acquisition_date": "2026-08-01T00:38:15Z", "scene_id": "S1_OLD"},
    ]

    # Target S2 observation: 2026-10-20 (> 7 days apart)
    match = service.find_closest_sentinel1_observation("2026-10-20", catalog, max_window_days=7)
    assert match["matched"] is False
    assert match["sentinel1_date"] is None
    assert "unavailable" in match["message"].lower()

# -------------------------------------------------------------------------
# Feature Computation, Backscatter in dB, Ratio, and Stats Tests
# -------------------------------------------------------------------------

def test_compute_sar_features_numerical_correctness():
    service = Sentinel1Service()
    h, w = 32, 32

    # Linear backscatter values: VV = 0.1 (-10 dB), VH = 0.01 (-20 dB)
    vv_linear = np.full((h, w), 0.1, dtype=np.float32)
    vh_linear = np.full((h, w), 0.01, dtype=np.float32)
    valid_mask = np.ones((h, w), dtype=bool)

    features = service.compute_sar_features(vv_linear, vh_linear, valid_mask, total_aoi_area_km2=5.0)

    vv_stats = features["features"]["vv"]["statistics"]
    vh_stats = features["features"]["vh"]["statistics"]
    ratio_stats = features["features"]["vv_vh_ratio"]["statistics"]

    # 10 * log10(0.1) = -10.0 dB
    assert pytest.approx(vv_stats["mean"], 0.1) == -10.0
    # 10 * log10(0.01) = -20.0 dB
    assert pytest.approx(vh_stats["mean"], 0.1) == -20.0
    # VV_dB - VH_dB = -10 - (-20) = 10.0 dB difference
    assert pytest.approx(ratio_stats["mean"], 0.1) == 10.0
    # Valid percentage
    assert features["quality"]["valid_pixel_percentage"] == 100.0
    assert features["quality"]["area_represented_km2"] == 5.0

def test_compute_sar_features_nodata_and_zeros():
    service = Sentinel1Service()
    h, w = 20, 20

    # Test zeros (division by zero and log10(0) safety)
    vv_linear = np.zeros((h, w), dtype=np.float32)
    vh_linear = np.zeros((h, w), dtype=np.float32)
    valid_mask = np.zeros((h, w), dtype=bool)
    valid_mask[0:5, 0:5] = True  # 25 valid pixels

    features = service.compute_sar_features(vv_linear, vh_linear, valid_mask)
    assert features["quality"]["valid_pixels"] == 25
    assert features["quality"]["total_pixels"] == 400

    vv_stats = features["features"]["vv"]["statistics"]
    assert vv_stats["mean"] is not None
    assert not np.isnan(vv_stats["mean"])
    assert not np.isinf(vv_stats["mean"])

# -------------------------------------------------------------------------
# Deterministic Demo Mode Test
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deterministic_demo_sar_reproducibility(sample_geometry):
    service = Sentinel1Service()
    vv1, vh1, mask1, meta1 = service._generate_deterministic_sar_data(sample_geometry, "2026-10-01", "2026-10-25")
    vv2, vh2, mask2, meta2 = service._generate_deterministic_sar_data(sample_geometry, "2026-10-01", "2026-10-25")

    np.testing.assert_array_equal(vv1, vv2)
    np.testing.assert_array_equal(vh1, vh2)
    np.testing.assert_array_equal(mask1, mask2)
    assert meta1["is_demo"] is True
    assert meta2["is_demo"] is True

# -------------------------------------------------------------------------
# Multimodal Fusion Preparation Tests
# -------------------------------------------------------------------------

def test_multimodal_fusion_dataset_preparation():
    service = MultimodalFusionService()

    optical_data = {
        "date": "2026-10-20",
        "scene": {"acquisition_date": "2026-10-20T10:00:00Z"},
        "indices": {
            "ndvi": {"statistics": {"mean": 0.62, "median": 0.64, "min": 0.1, "max": 0.85}},
            "ndmi": {"statistics": {"mean": 0.18, "median": 0.19, "min": -0.1, "max": 0.4}},
            "ndre": {"statistics": {"mean": 0.41, "median": 0.42, "min": 0.05, "max": 0.65}},
            "evi": {"statistics": {"mean": 0.55, "median": 0.57, "min": 0.1, "max": 0.8}},
        },
        "quality": {"valid_percentage": 98.5},
    }

    sar_data = {
        "observation_date": "2026-10-21T00:38:15Z",
        "product_type": "Sentinel-1 GRD",
        "polarization": ["VV", "VH"],
        "sar_features": {
            "vv": {"statistics": {"mean": -10.4, "median": -10.2, "min": -15.0, "max": -6.0, "std": 1.5}},
            "vh": {"statistics": {"mean": -17.2, "median": -17.0, "min": -22.0, "max": -12.0, "std": 1.8}},
            "vv_vh_ratio": {"statistics": {"mean": 6.8, "median": 6.8, "min": 2.0, "max": 10.0, "std": 1.2}},
        },
        "quality": {"valid_pixel_percentage": 99.1},
    }

    fusion = service.prepare_multimodal_dataset(optical_data, sar_data)

    assert "optical" in fusion
    assert fusion["optical"]["ndvi"]["mean"] == 0.62
    assert "sar" in fusion
    assert fusion["sar"]["vv"]["mean"] == -10.4
    assert fusion["quality"]["optical_valid_percentage"] == 98.5
    assert fusion["quality"]["sar_valid_percentage"] == 99.1
    assert fusion["quality"]["fusion_readiness"] is True

def test_align_spatial_rasters():
    service = MultimodalFusionService()

    # Optical grid 64x64, SAR grid 32x32
    opt = {"B08": np.ones((64, 64), dtype=np.float32)}
    sar = {"VV": np.ones((32, 32), dtype=np.float32) * 0.1}

    aligned_opt, aligned_sar = service.align_spatial_rasters(opt, sar, reference_grid="B08")
    assert aligned_opt["B08"].shape == (64, 64)
    assert aligned_sar["VV"].shape == (64, 64)

# -------------------------------------------------------------------------
# API Endpoints Integration Tests
# -------------------------------------------------------------------------

def test_api_sentinel1_search_endpoint(sample_geometry):
    response = client.post(
        "/api/v1/satellite/sentinel1/search",
        json={
            "geometry": sample_geometry,
            "start_date": "2026-09-01",
            "end_date": "2026-10-25",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_scenes_found"] > 0
    assert len(data["scenes"]) > 0

def test_api_sentinel1_analysis_endpoint(sample_geometry):
    response = client.post(
        "/api/v1/analysis/sentinel1",
        json={
            "geometry": sample_geometry,
            "start_date": "2026-10-01",
            "end_date": "2026-10-25",
            "s2_date": "2026-10-20",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "sar_features" in data
    assert "vv" in data["sar_features"]
    assert "vh" in data["sar_features"]
    assert "vv_vh_ratio" in data["sar_features"]
    assert "temporal_match" in data
    assert data["temporal_match"]["matched"] is True

def test_api_multimodal_fusion_endpoint(sample_geometry):
    response = client.post(
        "/api/v1/analysis/fusion/prepare",
        json={
            "geometry": sample_geometry,
            "start_date": "2026-10-01",
            "end_date": "2026-10-25",
            "max_cloud_coverage": 30.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "fusion_dataset" in data
    assert "optical" in data["fusion_dataset"]
    assert "sar" in data["fusion_dataset"]

def test_api_sentinel1_lat_lon_fallback():
    response = client.post(
        "/api/v1/analysis/sentinel1",
        json={
            "latitude": 16.5062,
            "longitude": 80.6480,
            "radius_km": 2.0,
            "start_date": "2026-10-01",
            "end_date": "2026-10-25",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "sar_features" in data

def test_api_sentinel1_missing_geometry_or_lat_lon():
    response = client.post(
        "/api/v1/analysis/sentinel1",
        json={
            "start_date": "2026-10-01",
            "end_date": "2026-10-25",
        },
    )
    assert response.status_code == 400
    assert "Either 'geometry' or 'latitude' and 'longitude' must be provided" in response.json()["detail"]
