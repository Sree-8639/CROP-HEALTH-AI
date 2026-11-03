"""
Tests for Phase 6 - Crop Condition Intelligence Service and Endpoints.
Ensures:
- Heuristic condition score calculation
- Threshold classification into spatial zones (0-5)
- NaN and cloud/invalid pixel masking
- Dynamic weight renormalization if indices are missing
- Stress pattern detection (Patterns A, B, C, D) with non-diagnostic language
- Data quality levels (Excellent, Good, Fair, Poor) and warnings
- Discrete colormap generation
- API endpoints POST and GET /api/v1/analysis/crop-condition
- Edge cases: all pixels invalid, all pixels valid, zero denominator, single index NaN
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app
from services.crop_condition_service import (
    DEFAULT_CONDITION_CONFIG,
    compute_crop_condition,
    generate_condition_colormap,
    normalize_index_array,
)
from services.sentinel2_service import Sentinel2Service

client = TestClient(app)


# ---------------------------------------------------------------------------
# Unit Tests for `normalize_index_array`
# ---------------------------------------------------------------------------

def test_normalize_index_array_basic():
    """Verify piece-wise linear normalization maps expected reference points."""
    # Using min_ref=0.0, low=0.2, mod=0.4, good=0.7, max_ref=0.9
    arr = np.array([0.0, 0.2, 0.4, 0.7, 0.9])
    norm = normalize_index_array(
        arr,
        min_ref=0.0,
        low_ref=0.2,
        mod_ref=0.4,
        good_ref=0.7,
        max_ref=0.9,
    )
    
    # 0.0 -> 0.0
    assert np.isclose(norm[0], 0.0)
    # 0.2 -> 0.30 (low reference point maps to 0.30)
    assert np.isclose(norm[1], 0.30)
    # 0.4 -> 0.55 (moderate reference point maps to 0.55)
    assert np.isclose(norm[2], 0.55)
    # 0.7 -> 0.80 (good reference point maps to 0.80)
    assert np.isclose(norm[3], 0.80)
    # 0.9 -> 1.0 (max reference point maps to 1.0)
    assert np.isclose(norm[4], 1.0)


def test_normalize_index_array_nan_handling():
    """NaNs in input array must remain NaNs without raising warnings."""
    arr = np.array([np.nan, 0.5, np.nan])
    norm = normalize_index_array(arr, min_ref=0.0, low_ref=0.2, mod_ref=0.4, good_ref=0.7, max_ref=0.9)
    assert np.isnan(norm[0])
    assert not np.isnan(norm[1])
    assert np.isnan(norm[2])


# ---------------------------------------------------------------------------
# Unit Tests for `compute_crop_condition`
# ---------------------------------------------------------------------------

def test_compute_crop_condition_all_valid_uniform():
    """Test with uniform synthetic indices."""
    shape = (10, 10)
    # High vigor across all indices: NDVI=0.75, NDMI=0.35, NDRE=0.45, EVI=0.60
    ndvi = np.full(shape, 0.75, dtype=np.float32)
    ndmi = np.full(shape, 0.35, dtype=np.float32)
    ndre = np.full(shape, 0.45, dtype=np.float32)
    evi = np.full(shape, 0.60, dtype=np.float32)

    result = compute_crop_condition(ndvi=ndvi, ndmi=ndmi, ndre=ndre, evi=evi)

    assert result["overall_condition"] in ("Good", "Very Good")
    assert result["condition_score"] >= 70.0
    assert result["quality"]["valid_percentage"] == 100.0
    assert result["quality"]["quality_level"] == "Excellent"
    assert result["zones"]["very_good"] + result["zones"]["good"] == 100.0
    assert result["classified_raster"].shape == shape
    assert set(np.unique(result["classified_raster"])).issubset({4, 5})


def test_compute_crop_condition_stress_pattern_a():
    """Low NDVI + Low NDMI triggers possible moisture stress pattern."""
    shape = (10, 10)
    ndvi = np.full(shape, 0.15, dtype=np.float32)  # Low NDVI
    ndmi = np.full(shape, -0.05, dtype=np.float32)  # Low NDMI
    ndre = np.full(shape, 0.15, dtype=np.float32)
    evi = np.full(shape, 0.15, dtype=np.float32)

    result = compute_crop_condition(ndvi=ndvi, ndmi=ndmi, ndre=ndre, evi=evi)
    pattern_types = [p["type"] for p in result["patterns"]]
    assert "possible_moisture_stress" in pattern_types

    # Ensure scientific non-diagnostic terminology is adhered to
    for p in result["patterns"]:
        assert "diagnos" not in p["explanation"].lower()
        assert "disease" not in p["explanation"].lower()
    
    # Check that possible_moisture_stress includes field verification requirement
    moist_p = next(p for p in result["patterns"] if p["type"] == "possible_moisture_stress")
    assert "field verification" in moist_p["explanation"].lower() or "requires" in moist_p["explanation"].lower()


def test_compute_crop_condition_stress_pattern_b():
    """Low NDVI + Normal NDMI triggers Pattern B."""
    shape = (10, 10)
    ndvi = np.full(shape, 0.18, dtype=np.float32)  # Low
    ndmi = np.full(shape, 0.30, dtype=np.float32)  # Normal/Good
    ndre = np.full(shape, 0.30, dtype=np.float32)
    evi = np.full(shape, 0.30, dtype=np.float32)

    result = compute_crop_condition(ndvi=ndvi, ndmi=ndmi, ndre=ndre, evi=evi)
    pattern_types = [p["type"] for p in result["patterns"]]
    assert "possible_canopy_vigor_reduction" in pattern_types


def test_compute_crop_condition_stress_pattern_c():
    """Normal NDVI + Low NDRE triggers Pattern C."""
    shape = (10, 10)
    ndvi = np.full(shape, 0.65, dtype=np.float32)  # Good
    ndmi = np.full(shape, 0.30, dtype=np.float32)  # Good
    ndre = np.full(shape, 0.12, dtype=np.float32)  # Low NDRE
    evi = np.full(shape, 0.50, dtype=np.float32)

    result = compute_crop_condition(ndvi=ndvi, ndmi=ndmi, ndre=ndre, evi=evi)
    pattern_types = [p["type"] for p in result["patterns"]]
    assert "possible_chlorophyll_transition" in pattern_types


def test_compute_crop_condition_stress_pattern_d():
    """Low EVI + Low NDVI triggers Pattern D."""
    shape = (10, 10)
    ndvi = np.full(shape, 0.18, dtype=np.float32)  # Low
    ndmi = np.full(shape, 0.20, dtype=np.float32)  # Moderate
    ndre = np.full(shape, 0.25, dtype=np.float32)  # Moderate
    evi = np.full(shape, 0.12, dtype=np.float32)  # Low

    result = compute_crop_condition(ndvi=ndvi, ndmi=ndmi, ndre=ndre, evi=evi)
    pattern_types = [p["type"] for p in result["patterns"]]
    assert "concurring_canopy_reduction" in pattern_types


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

def test_compute_crop_condition_all_pixels_invalid():
    """Edge Case: When 100% of pixels are invalid (e.g. obscured by clouds or NaNs)."""
    shape = (8, 8)
    ndvi = np.full(shape, np.nan, dtype=np.float32)
    valid_mask = np.zeros(shape, dtype=bool)  # All masked

    result = compute_crop_condition(ndvi=ndvi, valid_mask=valid_mask)

    assert result["overall_condition"] == "No Data"
    assert result["condition_score"] == 0.0
    assert result["quality"]["valid_pixels"] == 0
    assert result["quality"]["valid_percentage"] == 0.0
    assert result["quality"]["quality_level"] == "Poor"
    assert len(result["warnings"]) > 0
    # All zones should be 0.0
    for v in result["zones"].values():
        assert v == 0.0
    # Raster should be all zeros (class 0 = No Data)
    assert (result["classified_raster"] == 0).all()


def test_compute_crop_condition_missing_optional_indices():
    """Edge Case: Dynamic renormalization when NDRE and EVI are None."""
    shape = (10, 10)
    ndvi = np.full(shape, 0.60, dtype=np.float32)
    ndmi = np.full(shape, 0.30, dtype=np.float32)

    result = compute_crop_condition(ndvi=ndvi, ndmi=ndmi, ndre=None, evi=None)

    assert result["overall_condition"] in ("Good", "Moderate")
    assert result["condition_score"] is not None
    assert "ndre" not in result["analysis_metadata"]["indices_used"]
    assert "evi" not in result["analysis_metadata"]["indices_used"]
    assert "ndvi" in result["analysis_metadata"]["indices_used"]
    assert "ndmi" in result["analysis_metadata"]["indices_used"]


def test_compute_crop_condition_partial_nan_index():
    """Edge Case: One index has NaNs in certain locations."""
    shape = (10, 10)
    ndvi = np.full(shape, 0.70, dtype=np.float32)
    ndmi = np.full(shape, 0.40, dtype=np.float32)
    ndre = np.full(shape, 0.35, dtype=np.float32)
    evi = np.full(shape, 0.50, dtype=np.float32)

    # Corrupt a 2x2 corner in NDMI
    ndmi[:2, :2] = np.nan

    result = compute_crop_condition(ndvi=ndvi, ndmi=ndmi, ndre=ndre, evi=evi)
    # The corrupted 4 pixels should still have a valid condition calculated from NDVI, NDRE, EVI
    assert result["quality"]["valid_pixels"] == 100
    assert result["condition_score"] > 60.0


def test_compute_crop_condition_low_data_quality_warning():
    """When valid pixel percentage is low, a quality warning must be generated."""
    shape = (10, 10)
    ndvi = np.full(shape, 0.60, dtype=np.float32)
    valid_mask = np.zeros(shape, dtype=bool)
    valid_mask[:2, :2] = True  # Only 4% valid

    result = compute_crop_condition(ndvi=ndvi, valid_mask=valid_mask)
    assert result["quality"]["valid_percentage"] == 4.0
    assert result["quality"]["quality_level"] == "Poor"
    assert any("large portions" in w.lower() or "confidence" in w.lower() for w in result["warnings"])


# ---------------------------------------------------------------------------
# Colormap Generation
# ---------------------------------------------------------------------------

def test_generate_condition_colormap():
    """Verify PNG generation from classified raster."""
    raster = np.array([
        [0, 1, 2],
        [3, 4, 5]
    ], dtype=np.uint8)

    base64_png = generate_condition_colormap(raster, is_demo=False)
    assert isinstance(base64_png, str)
    assert len(base64_png) > 50

    # Demo mode banner inclusion
    demo_png = generate_condition_colormap(raster, is_demo=True)
    assert isinstance(demo_png, str)
    assert len(demo_png) > 50


# ---------------------------------------------------------------------------
# API Endpoint Integration Tests
# ---------------------------------------------------------------------------

def test_api_crop_condition_post_demo_mode():
    """Test POST /api/v1/analysis/crop-condition in demo mode."""
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [78.50, 17.40],
                [78.51, 17.40],
                [78.51, 17.41],
                [78.50, 17.41],
                [78.50, 17.40]
            ]]
        },
        "start_date": "2024-05-01",
        "end_date": "2024-05-25",
        "max_cloud_coverage": 30.0
    }

    # Ensure demo mode behavior
    with patch.object(Sentinel2Service, "is_demo_mode", True):
        response = client.post("/api/v1/analysis/crop-condition", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "overall_condition" in data
        assert "condition_score" in data
        assert "zones" in data
        assert "indices" in data
        assert "patterns" in data
        assert "quality" in data
        assert "condition_map" in data
        assert data["scene"]["is_demo"] is True

        # Ensure all 4 indices were evaluated
        for idx in ["ndvi", "ndmi", "ndre", "evi"]:
            assert idx in data["indices"]
            stats = data["indices"][idx]["statistics"]
            assert "mean" in stats
            assert "median" in stats
            assert "min" in stats
            assert "max" in stats


def test_api_crop_condition_get_demo_mode():
    """Test GET /api/v1/analysis/crop-condition with query params."""
    with patch.object(Sentinel2Service, "is_demo_mode", True):
        response = client.get(
            "/api/v1/analysis/crop-condition?lat=17.405&lon=78.505&radius_km=1.0&start_date=2024-05-01&end_date=2024-05-25"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "overall_condition" in data
        assert data["scene"]["is_demo"] is True


def test_api_crop_condition_deterministic_demo():
    """Verify demo mode produces deterministic results for identical input."""
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [78.50, 17.40],
                [78.51, 17.40],
                [78.51, 17.41],
                [78.50, 17.41],
                [78.50, 17.40]
            ]]
        },
        "start_date": "2024-05-01",
        "end_date": "2024-05-25",
        "max_cloud_coverage": 30.0
    }
    with patch.object(Sentinel2Service, "is_demo_mode", True):
        res1 = client.post("/api/v1/analysis/crop-condition", json=payload).json()
        res2 = client.post("/api/v1/analysis/crop-condition", json=payload).json()

        assert res1["condition_score"] == res2["condition_score"]
        assert res1["overall_condition"] == res2["overall_condition"]
        assert res1["zones"] == res2["zones"]

