import numpy as np
import pytest

from services.ndvi_service import (
    calculate_ndvi,
    create_valid_pixel_mask,
    calculate_ndvi_statistics,
    generate_ndvi_colormap,
    generate_demo_spectral_bands,
)

# 1. Basic NDVI calculation
def test_basic_ndvi_calculation():
    red = np.array([0.2], dtype=np.float64)
    nir = np.array([0.6], dtype=np.float64)
    # NDVI = (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5
    ndvi = calculate_ndvi(red, nir)
    assert np.isclose(ndvi[0], 0.5, atol=1e-5)

# 2. Equal RED and NIR values
def test_equal_red_and_nir():
    red = np.array([0.35], dtype=np.float64)
    nir = np.array([0.35], dtype=np.float64)
    ndvi = calculate_ndvi(red, nir)
    assert np.isclose(ndvi[0], 0.0, atol=1e-5)

# 3. Division by zero (Must return NaN, not silent zero)
def test_division_by_zero():
    red = np.array([0.0], dtype=np.float64)
    nir = np.array([0.0], dtype=np.float64)
    ndvi = calculate_ndvi(red, nir)
    assert np.isnan(ndvi[0])
    # Denominator close to zero
    red_tiny = np.array([1e-9], dtype=np.float64)
    nir_tiny = np.array([1e-9], dtype=np.float64)
    ndvi_tiny = calculate_ndvi(red_tiny, nir_tiny)
    assert np.isnan(ndvi_tiny[0])

# 4. Negative NDVI (Typical of water or bare wet soil)
def test_negative_ndvi():
    red = np.array([0.4], dtype=np.float64)
    nir = np.array([0.1], dtype=np.float64)
    # NDVI = (0.1 - 0.4) / (0.1 + 0.4) = -0.3 / 0.5 = -0.6
    ndvi = calculate_ndvi(red, nir)
    assert np.isclose(ndvi[0], -0.6, atol=1e-5)

# 5. High NDVI (Dense healthy green canopy)
def test_high_ndvi():
    red = np.array([0.05], dtype=np.float64)
    nir = np.array([0.80], dtype=np.float64)
    # NDVI = (0.80 - 0.05) / (0.80 + 0.05) = 0.75 / 0.85 ~= 0.88235
    ndvi = calculate_ndvi(red, nir)
    assert np.isclose(ndvi[0], 0.75 / 0.85, atol=1e-4)
    assert ndvi[0] > 0.8

# 6. Nodata pixels (NaN inputs preserved as NaN)
def test_nodata_pixels():
    red = np.array([np.nan, 0.1], dtype=np.float64)
    nir = np.array([0.5, np.nan], dtype=np.float64)
    ndvi = calculate_ndvi(red, nir)
    assert np.isnan(ndvi[0])
    assert np.isnan(ndvi[1])

# 7. Invalid pixels (Negative reflectance values or infinite values)
def test_invalid_pixels():
    red = np.array([-0.05, np.inf], dtype=np.float64)
    nir = np.array([0.4, 0.5], dtype=np.float64)
    ndvi = calculate_ndvi(red, nir)
    assert np.isnan(ndvi[0])
    assert np.isnan(ndvi[1])

# 8. Cloud masking (Cloud contaminated pixels excluded)
def test_cloud_masking():
    # 4 pixels: pixel 0=crop (SCL 4), pixel 1=cloud shadow (SCL 3), pixel 2=cloud (SCL 8), pixel 3=cirrus (SCL 10)
    red = np.array([[0.05, 0.10], [0.50, 0.40]], dtype=np.float64)
    nir = np.array([[0.50, 0.15], [0.55, 0.45]], dtype=np.float64)
    scl = np.array([[4, 3], [8, 10]], dtype=np.uint8) # 4=Veg, 3=Shadow, 8=Cloud, 10=Cirrus
    data_mask = np.ones((2, 2), dtype=np.uint8)

    valid_mask = create_valid_pixel_mask(red, nir, scl=scl, data_mask=data_mask)
    assert valid_mask[0, 0] is True or valid_mask[0, 0] == 1  # Veg
    assert valid_mask[0, 1] is False or valid_mask[0, 1] == 0 # Shadow
    assert valid_mask[1, 0] is False or valid_mask[1, 0] == 0 # Cloud med
    assert valid_mask[1, 1] is False or valid_mask[1, 1] == 0 # Cirrus

    ndvi = calculate_ndvi(red, nir, valid_mask=valid_mask)
    assert not np.isnan(ndvi[0, 0])
    assert np.isnan(ndvi[0, 1])
    assert np.isnan(ndvi[1, 0])
    assert np.isnan(ndvi[1, 1])

# 9. NDVI statistics (min, max, mean, median, std dev, counts, percentages)
def test_ndvi_statistics():
    # 2x2 grid with known NDVI values: 0.2, 0.4, 0.6, NaN
    ndvi_grid = np.array([[0.2, 0.4], [0.6, np.nan]], dtype=np.float64)
    stats = calculate_ndvi_statistics(ndvi_grid, pixel_size_m=10.0, total_aoi_area_km2=10.0)

    assert stats["total_pixels"] == 4
    assert stats["valid_pixels"] == 3
    assert np.isclose(stats["valid_pixel_percentage"], 75.0, atol=1e-2)
    assert np.isclose(stats["min_ndvi"], 0.2, atol=1e-4)
    assert np.isclose(stats["max_ndvi"], 0.6, atol=1e-4)
    assert np.isclose(stats["mean_ndvi"], 0.4, atol=1e-4)
    assert np.isclose(stats["median_ndvi"], 0.4, atol=1e-4)
    assert stats["std_dev"] > 0
    assert stats["area_represented_km2"] == 7.5 # 75% of 10 km²

# 10. Demo-mode NDVI generation (deterministic arrays executed through real pipeline)
def test_demo_spectral_data_and_pipeline():
    b04, b08, scl, data_mask = generate_demo_spectral_bands("2026-10-01", "2026-10-26", shape=(32, 32))
    assert b04.shape == (32, 32)
    assert b08.shape == (32, 32)

    valid_mask = create_valid_pixel_mask(b04, b08, scl=scl, data_mask=data_mask)
    ndvi = calculate_ndvi(b04, b08, valid_mask=valid_mask)
    stats = calculate_ndvi_statistics(ndvi, valid_mask=valid_mask, total_aoi_area_km2=5.0)

    assert stats["valid_pixels"] > 0
    assert stats["mean_ndvi"] is not None
    assert stats["min_ndvi"] <= stats["max_ndvi"]

    # Colormap generation
    color_map_data = generate_ndvi_colormap(ndvi, valid_mask=valid_mask, is_demo=True, output_size=(64, 64))
    assert color_map_data.startswith("data:image/png;base64,")
