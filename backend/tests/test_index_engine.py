"""
Unit tests for the Spectral Index Engine and SCL Cloud/Shadow Masking.
Tests NDVI, NDMI, NDRE, EVI formulas, division-by-zero protection,
boundary clipping, SCL mask filtering, and metric statistics extraction.
"""

import pytest
import numpy as np

from services.ndvi_service import (
    calculate_ndvi,
    create_valid_pixel_mask,
    extract_scl_cloud_metrics,
    calculate_ndvi_statistics,
)
from services.ndmi_service import calculate_ndmi
from services.ndre_service import calculate_ndre
from services.evi_service import calculate_evi


def test_ndvi_formula_and_bounds():
    """Validates NDVI formula: (NIR - Red) / (NIR + Red) and [-1, 1] bounds."""
    red = np.array([[0.1, 0.2], [0.05, 0.4]], dtype=np.float32)
    nir = np.array([[0.5, 0.6], [0.45, 0.4]], dtype=np.float32)

    ndvi = calculate_ndvi(red, nir)

    # Expected:
    # (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 = 0.6667
    # (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5000
    # (0.45 - 0.05) / (0.45 + 0.05) = 0.4 / 0.5 = 0.8000
    # (0.4 - 0.4) / (0.4 + 0.4) = 0.0
    assert np.isclose(ndvi[0, 0], 2.0 / 3.0, atol=1e-4)
    assert np.isclose(ndvi[0, 1], 0.5, atol=1e-4)
    assert np.isclose(ndvi[1, 0], 0.8, atol=1e-4)
    assert np.isclose(ndvi[1, 1], 0.0, atol=1e-4)
    assert np.all(ndvi >= -1.0) and np.all(ndvi <= 1.0)


def test_ndvi_division_by_zero_and_negative_reflectance():
    """Validates division by zero protection and non-physical negative reflectance handling."""
    red = np.array([[0.0, -0.1], [np.nan, 0.2]], dtype=np.float32)
    nir = np.array([[0.0, 0.5], [0.6, np.inf]], dtype=np.float32)

    ndvi = calculate_ndvi(red, nir)

    # All these edge cases must evaluate to NaN, never raise unhandled exceptions
    assert np.isnan(ndvi[0, 0])  # Zero division (0 - 0) / (0 + 0)
    assert np.isnan(ndvi[0, 1])  # Negative reflectance
    assert np.isnan(ndvi[1, 0])  # NaN input
    assert np.isnan(ndvi[1, 1])  # Inf input


def test_ndmi_formula_and_bounds():
    """Validates NDMI formula: (NIR - SWIR) / (NIR + SWIR) and resampling."""
    nir = np.array([[0.6, 0.5], [0.7, 0.4]], dtype=np.float32)
    swir = np.array([[0.2, 0.3], [0.1, 0.4]], dtype=np.float32)

    ndmi = calculate_ndmi(nir, swir)

    # (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5
    # (0.4 - 0.4) / (0.4 + 0.4) = 0.0
    assert np.isclose(ndmi[0, 0], 0.5, atol=1e-4)
    assert np.isclose(ndmi[1, 1], 0.0, atol=1e-4)
    assert np.all(ndmi >= -1.0) and np.all(ndmi <= 1.0)


def test_ndre_formula_and_bounds():
    """Validates NDRE formula: (NIR - RedEdge) / (NIR + RedEdge)."""
    nir = np.array([[0.7, 0.6]], dtype=np.float32)
    red_edge = np.array([[0.3, 0.2]], dtype=np.float32)

    ndre = calculate_ndre(nir, red_edge)

    # (0.7 - 0.3) / (0.7 + 0.3) = 0.4 / 1.0 = 0.4
    # (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5
    assert np.isclose(ndre[0, 0], 0.4, atol=1e-4)
    assert np.isclose(ndre[0, 1], 0.5, atol=1e-4)
    assert np.all(ndre >= -1.0) and np.all(ndre <= 1.0)


def test_evi_formula_and_bounds():
    """Validates Enhanced Vegetation Index (EVI) formula."""
    blue = np.array([[0.05, 0.06]], dtype=np.float32)
    red = np.array([[0.10, 0.12]], dtype=np.float32)
    nir = np.array([[0.50, 0.45]], dtype=np.float32)

    evi = calculate_evi(blue, red, nir)

    # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1.0)
    # For pixel (0,0):
    # num = 2.5 * (0.50 - 0.10) = 1.0
    # den = 0.50 + 6*(0.10) - 7.5*(0.05) + 1.0 = 0.50 + 0.60 - 0.375 + 1.0 = 1.725
    # expected = 1.0 / 1.725 = 0.5797
    assert np.isclose(evi[0, 0], 1.0 / 1.725, atol=1e-4)
    assert np.all(evi >= -1.5) and np.all(evi <= 1.5)


def test_scl_cloud_and_shadow_masking():
    """
    Validates Sentinel-2 Scene Classification Layer (SCL) filtering.
    Invalid SCL classes:
      0: NO_DATA, 1: DEFECTIVE, 3: CLOUD_SHADOW,
      8: CLOUD_MEDIUM_PROB, 9: CLOUD_HIGH_PROB, 10: THIN_CIRRUS, 11: SNOW
    Valid SCL classes:
      2: DARK_AREA, 4: VEGETATION, 5: NOT_VEGETATED, 6: WATER, 7: UNCLASSIFIED
    """
    # 12 pixels representing SCL classes 0 through 11
    scl = np.arange(12, dtype=np.uint8).reshape((3, 4))
    red = np.full((3, 4), 0.1, dtype=np.float32)
    nir = np.full((3, 4), 0.5, dtype=np.float32)

    valid_mask = create_valid_pixel_mask(red, nir, scl=scl)

    # Check each class
    expected_valid_classes = {2, 4, 5, 6, 7}
    expected_invalid_classes = {0, 1, 3, 8, 9, 10, 11}

    for cls in range(12):
        row = cls // 4
        col = cls % 4
        if cls in expected_valid_classes:
            assert valid_mask[row, col] is True or valid_mask[row, col] == 1, f"Class {cls} should be valid"
        else:
            assert valid_mask[row, col] is False or valid_mask[row, col] == 0, f"Class {cls} should be invalid"


def test_extract_scl_cloud_metrics():
    """Validates cloud %, shadow %, and valid % calculation from SCL."""
    # 10 pixels:
    # 2 clouds (8, 9) -> 20%
    # 1 shadow (3) -> 10%
    # 1 nodata (0) -> 10%
    # 6 valid (4, 4, 4, 5, 5, 6) -> 60%
    scl = np.array([8, 9, 3, 0, 4, 4, 4, 5, 5, 6], dtype=np.uint8)

    metrics = extract_scl_cloud_metrics(scl)

    assert metrics["cloud_pct"] == 20.0
    assert metrics["shadow_pct"] == 10.0
    assert metrics["valid_pct"] == 60.0


def test_ndvi_statistics_calculation():
    """Validates calculation of mean, median, min, max, std, and valid coverage."""
    ndvi = np.array([[0.2, 0.4], [0.6, np.nan]], dtype=np.float64)
    valid_mask = np.array([[True, True], [True, False]], dtype=bool)

    stats = calculate_ndvi_statistics(ndvi, valid_mask=valid_mask, pixel_size_m=10.0)

    assert stats["valid_pixels"] == 3
    assert stats["total_pixels"] == 4
    assert stats["valid_pixel_percentage"] == 75.0
    assert np.isclose(stats["mean_ndvi"], 0.4, atol=1e-3)
    assert np.isclose(stats["median_ndvi"], 0.4, atol=1e-3)
    assert np.isclose(stats["min_ndvi"], 0.2, atol=1e-3)
    assert np.isclose(stats["max_ndvi"], 0.6, atol=1e-3)
    assert stats["std_dev"] > 0.0
