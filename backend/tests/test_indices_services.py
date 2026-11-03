import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app
from services.raster_utils import resample_band, align_bands, validate_band_shapes
from services.ndmi_service import (
    calculate_ndmi,
    calculate_ndmi_statistics,
    generate_ndmi_colormap,
)
from services.ndre_service import (
    calculate_ndre,
    calculate_ndre_statistics,
    generate_ndre_colormap,
)
from services.evi_service import (
    calculate_evi,
    calculate_evi_statistics,
    generate_evi_colormap,
)
from services.ndvi_service import (
    create_valid_pixel_mask,
    generate_all_demo_spectral_bands,
)

client = TestClient(app)

SAMPLE_AOI = {
    "type": "Polygon",
    "coordinates": [
        [
            [77.58, 12.96],
            [77.60, 12.96],
            [77.60, 12.98],
            [77.58, 12.98],
            [77.58, 12.96],
        ]
    ],
}


# =====================================================================
# 1. RASTER UTILS & BAND RESOLUTION TESTS
# =====================================================================

def test_raster_utils_alignment_same_shape():
    b08 = np.ones((10, 10), dtype=np.float32) * 0.5
    b11 = np.ones((10, 10), dtype=np.float32) * 0.2
    aligned = align_bands({"B08": b08, "B11": b11}, target_band="B08")
    assert aligned["B08"].shape == (10, 10)
    assert aligned["B11"].shape == (10, 10)
    assert np.allclose(aligned["B11"], 0.2)


def test_raster_utils_resample_20m_to_10m():
    # 20m band is 5x5, 10m target is 10x10
    b05_20m = np.full((5, 5), 0.3, dtype=np.float32)
    b08_10m = np.full((10, 10), 0.6, dtype=np.float32)
    aligned = align_bands({"B08": b08_10m, "B05": b05_20m}, target_band="B08")
    assert aligned["B05"].shape == (10, 10)
    assert np.allclose(aligned["B05"], 0.3, atol=1e-3)


def test_raster_utils_nan_preservation():
    arr = np.array([[0.2, np.nan], [0.4, 0.6]], dtype=np.float32)
    resampled = resample_band(arr, (4, 4), method="bilinear")
    assert resampled.shape == (4, 4)
    # The top-right quadrant should maintain NaN values
    assert np.isnan(resampled[0, 3])


def test_raster_utils_shape_validation():
    arr1 = np.zeros((10, 10))
    arr2 = np.zeros((10, 10))
    validate_band_shapes(arr1, arr2)

    arr3 = np.zeros((5, 5))
    with pytest.raises(ValueError, match="Band shape mismatch"):
        validate_band_shapes(arr1, arr3)


# =====================================================================
# 2. NDMI TESTS: (NIR - SWIR) / (NIR + SWIR)
# =====================================================================

def test_ndmi_normal_calculation():
    # B08 (NIR) = 0.6, B11 (SWIR) = 0.2
    # NDMI = (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5
    b08 = np.array([0.6], dtype=np.float32)
    b11 = np.array([0.2], dtype=np.float32)
    ndmi = calculate_ndmi(b08, b11)
    assert np.isclose(ndmi[0], 0.5, atol=1e-5)


def test_ndmi_zero_denominator():
    b08 = np.array([0.0, 1e-9], dtype=np.float32)
    b11 = np.array([0.0, 1e-9], dtype=np.float32)
    ndmi = calculate_ndmi(b08, b11)
    assert np.isnan(ndmi[0])
    assert np.isnan(ndmi[1])


def test_ndmi_negative_values():
    # Moisture-stressed / dry bare soil: SWIR > NIR
    # B08 = 0.15, B11 = 0.45
    # NDMI = (0.15 - 0.45) / (0.15 + 0.45) = -0.3 / 0.6 = -0.5
    b08 = np.array([0.15], dtype=np.float32)
    b11 = np.array([0.45], dtype=np.float32)
    ndmi = calculate_ndmi(b08, b11)
    assert np.isclose(ndmi[0], -0.5, atol=1e-5)


def test_ndmi_nodata_and_valid_mask():
    b08 = np.array([[0.5, np.nan], [0.4, 0.6]], dtype=np.float32)
    b11 = np.array([[0.2, 0.2], [np.nan, 0.3]], dtype=np.float32)
    mask = np.array([[True, True], [True, False]])  # pixel [1,1] masked out
    ndmi = calculate_ndmi(b08, b11, valid_mask=mask)

    assert not np.isnan(ndmi[0, 0])
    assert np.isnan(ndmi[0, 1])  # input NaN
    assert np.isnan(ndmi[1, 0])  # input NaN
    assert np.isnan(ndmi[1, 1])  # masked out


def test_ndmi_statistics():
    grid = np.array([[0.1, 0.3], [0.5, np.nan]], dtype=np.float32)
    stats = calculate_ndmi_statistics(grid)
    assert stats["total_pixels"] == 4
    assert stats["valid_pixels"] == 3
    assert np.isclose(stats["valid_pixel_percentage"], 75.0)
    assert np.isclose(stats["min"], 0.1)
    assert np.isclose(stats["max"], 0.5)
    assert np.isclose(stats["mean"], 0.3)
    assert np.isclose(stats["median"], 0.3)
    assert "moisture_signal_distribution" in stats


def test_ndmi_colormap():
    grid = np.array([[0.1, -0.2], [0.4, np.nan]], dtype=np.float32)
    cmap = generate_ndmi_colormap(grid, is_demo=True)
    assert isinstance(cmap, str)
    assert cmap.startswith("data:image/png;base64,")


# =====================================================================
# 3. NDRE TESTS: (NIR - RedEdge) / (NIR + RedEdge)
# =====================================================================

def test_ndre_normal_calculation():
    # B08 (NIR) = 0.5, B05 (RedEdge) = 0.3
    # NDRE = (0.5 - 0.3) / (0.5 + 0.3) = 0.2 / 0.8 = 0.25
    b08 = np.array([0.5], dtype=np.float32)
    b05 = np.array([0.3], dtype=np.float32)
    ndre = calculate_ndre(b08, b05)
    assert np.isclose(ndre[0], 0.25, atol=1e-5)


def test_ndre_zero_denominator():
    b08 = np.array([0.0], dtype=np.float32)
    b05 = np.array([0.0], dtype=np.float32)
    ndre = calculate_ndre(b08, b05)
    assert np.isnan(ndre[0])


def test_ndre_band_alignment_and_resampling():
    # B08 is 4x4 (10m), B05 is 2x2 (20m)
    b08 = np.full((4, 4), 0.5, dtype=np.float32)
    b05_20m = np.full((2, 2), 0.3, dtype=np.float32)
    ndre = calculate_ndre(b08, b05_20m)
    assert ndre.shape == (4, 4)
    assert np.allclose(ndre, 0.25, atol=1e-3)


def test_ndre_nodata_and_statistics():
    grid = np.array([[0.1, 0.2], [0.3, np.nan]], dtype=np.float32)
    stats = calculate_ndre_statistics(grid)
    assert stats["total_pixels"] == 4
    assert stats["valid_pixels"] == 3
    assert np.isclose(stats["min"], 0.1)
    assert np.isclose(stats["max"], 0.3)
    assert np.isclose(stats["mean"], 0.2)
    assert "red_edge_distribution" in stats


def test_ndre_colormap():
    grid = np.array([[0.2, 0.4], [0.1, np.nan]], dtype=np.float32)
    cmap = generate_ndre_colormap(grid, is_demo=False)
    assert isinstance(cmap, str)
    assert cmap.startswith("data:image/png;base64,")


# =====================================================================
# 4. EVI TESTS: 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
# =====================================================================

def test_evi_normal_calculation():
    # NIR (B08) = 0.5, RED (B04) = 0.1, BLUE (B02) = 0.05
    # Numerator = 2.5 * (0.5 - 0.1) = 2.5 * 0.4 = 1.0
    # Denominator = 0.5 + 6*0.1 - 7.5*0.05 + 1.0
    #             = 0.5 + 0.6 - 0.375 + 1.0 = 1.725
    # EVI = 1.0 / 1.725 ~= 0.57971
    b02 = np.array([0.05], dtype=np.float32)
    b04 = np.array([0.1], dtype=np.float32)
    b08 = np.array([0.5], dtype=np.float32)
    evi = calculate_evi(b02, b04, b08)
    expected = 1.0 / 1.725
    assert np.isclose(evi[0], expected, atol=1e-4)


def test_evi_denominator_handling():
    # If denominator <= 0, should become NaN safely
    b02 = np.array([1.0], dtype=np.float32)  # High blue -> -7.5 pulls denom negative
    b04 = np.array([0.0], dtype=np.float32)
    b08 = np.array([0.0], dtype=np.float32)
    # Denom = 0 + 0 - 7.5 + 1 = -6.5 <= 0
    evi = calculate_evi(b02, b04, b08)
    assert np.isnan(evi[0])


def test_evi_nodata_and_valid_pixels():
    b02 = np.array([0.05, np.nan], dtype=np.float32)
    b04 = np.array([0.1, 0.1], dtype=np.float32)
    b08 = np.array([0.5, 0.5], dtype=np.float32)
    evi = calculate_evi(b02, b04, b08)
    assert not np.isnan(evi[0])
    assert np.isnan(evi[1])


def test_evi_statistics():
    grid = np.array([[0.2, 0.4], [0.6, np.nan]], dtype=np.float32)
    stats = calculate_evi_statistics(grid)
    assert stats["total_pixels"] == 4
    assert stats["valid_pixels"] == 3
    assert np.isclose(stats["valid_pixel_percentage"], 75.0)
    assert np.isclose(stats["min"], 0.2)
    assert np.isclose(stats["max"], 0.6)
    assert np.isclose(stats["mean"], 0.4)
    assert "canopy_signal_distribution" in stats


def test_evi_colormap():
    grid = np.array([[0.1, 0.5], [0.3, np.nan]], dtype=np.float32)
    cmap = generate_evi_colormap(grid, is_demo=True)
    assert isinstance(cmap, str)
    assert cmap.startswith("data:image/png;base64,")


# =====================================================================
# 5. DEMO SPECTRAL BANDS & COMMON CLOUD MASK
# =====================================================================

def test_demo_spectral_bands_generation():
    bands = generate_all_demo_spectral_bands(width=64, height=64)
    assert set(bands.keys()) == {"B02", "B04", "B05", "B08", "B11", "SCL", "dataMask"}
    # Check 10m vs 20m native resolution simulation
    assert bands["B02"].shape == (64, 64)
    assert bands["B04"].shape == (64, 64)
    assert bands["B08"].shape == (64, 64)
    assert bands["B05"].shape == (32, 32)  # 20m native simulated
    assert bands["B11"].shape == (32, 32)  # 20m native simulated

    # Verify cloud mask filters cloudy pixels across indices
    valid_mask = create_valid_pixel_mask(
        bands["B04"], bands["B08"], scl=bands["SCL"], data_mask=bands["dataMask"]
    )
    assert valid_mask.shape == (64, 64)
    assert np.any(valid_mask)
    assert np.any(~valid_mask)  # contains clouds/water/nodata


# =====================================================================
# 6. API ENDPOINTS INTEGRATION (DEMO MODE & VALIDATION)
# =====================================================================

def test_api_ndmi_endpoint():
    payload = {
        "geometry": SAMPLE_AOI,
        "start_date": "2026-06-01",
        "end_date": "2026-06-15",
        "max_cloud_coverage": 25.0,
    }
    response = client.post("/api/v1/analysis/ndmi", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["index_name"] == "NDMI"
    assert "moisture" in data["indicator_label"].lower()
    assert "statistics" in data
    assert "ndmi_map" in data
    assert data["ndmi_map"].startswith("data:image/png;base64,")
    assert data["is_demo"] is True


def test_api_ndre_endpoint():
    payload = {
        "geometry": SAMPLE_AOI,
        "start_date": "2026-06-01",
        "end_date": "2026-06-15",
        "max_cloud_coverage": 25.0,
    }
    response = client.post("/api/v1/analysis/ndre", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["index_name"] == "NDRE"
    assert "red-edge" in data["indicator_label"].lower()
    assert "statistics" in data
    assert "ndre_map" in data
    assert data["ndre_map"].startswith("data:image/png;base64,")
    assert data["is_demo"] is True


def test_api_evi_endpoint():
    payload = {
        "geometry": SAMPLE_AOI,
        "start_date": "2026-06-01",
        "end_date": "2026-06-15",
        "max_cloud_coverage": 25.0,
    }
    response = client.post("/api/v1/analysis/evi", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["index_name"] == "EVI"
    assert "enhanced" in data["indicator_label"].lower()
    assert "statistics" in data
    assert "evi_map" in data
    assert data["evi_map"].startswith("data:image/png;base64,")
    assert data["is_demo"] is True


def test_api_combined_indices_endpoint():
    payload = {
        "geometry": SAMPLE_AOI,
        "start_date": "2026-06-01",
        "end_date": "2026-06-15",
        "max_cloud_coverage": 25.0,
    }
    response = client.post("/api/v1/analysis/indices", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "scene" in data
    assert "ndvi" in data
    assert "ndmi" in data
    assert "ndre" in data
    assert "evi" in data

    # Verify each index has description, bands_used, statistics, and map
    for key in ["ndvi", "ndmi", "ndre", "evi"]:
        idx = data[key]
        assert "statistics" in idx
        assert "map" in idx
        assert "bands_used" in idx
        assert idx["map"].startswith("data:image/png;base64,")
        assert "mean" in idx["statistics"]
        assert "median" in idx["statistics"]
        assert "min" in idx["statistics"]
        assert "max" in idx["statistics"]

    assert data["scene"]["is_demo"] is True
