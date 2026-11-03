import io
import json
import pytest
import numpy as np
import tifffile
from fastapi.testclient import TestClient

from main import app
from services.fusion_service import (
    MultimodalFusionService,
    multimodal_fusion_service,
    MULTIMODAL_FEATURE_NAMES,
    MASK_VALID,
    MASK_PARTIAL,
    MASK_INVALID,
)
from services.raster_alignment import (
    CanonicalGrid,
    create_canonical_aoi_grid,
    resample_raster_to_canonical_grid,
    align_all_multimodal_features,
    calculate_utm_epsg,
    wgs84_to_utm_approx,
)
from services.historical_service import historical_monitoring_service

client = TestClient(app)

@pytest.fixture
def sample_geometry():
    """Standard field AOI polygon around Andhra Pradesh agricultural test site."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [80.6400, 16.5000],
                [80.6550, 16.5000],
                [80.6550, 16.5120],
                [80.6400, 16.5120],
                [80.6400, 16.5000],
            ]
        ],
    }

# ---------------------------------------------------------------------------
# 1. Canonical Grid and CRS Alignment Tests
# ---------------------------------------------------------------------------

def test_canonical_aoi_grid_creation(sample_geometry):
    grid = create_canonical_aoi_grid(sample_geometry, target_resolution_m=10.0)
    assert isinstance(grid, CanonicalGrid)
    assert grid.epsg == 32644  # Andhra Pradesh lon ~80.6° falls in UTM Zone 44N
    assert "EPSG:32644" in grid.crs
    assert grid.width > 0
    assert grid.height > 0
    assert len(grid.transform) == 6
    assert grid.resolution_m > 0
    assert grid.bounds[0] < grid.bounds[2]  # min_x < max_x
    assert grid.bounds[1] < grid.bounds[3]  # min_y < max_y

def test_wgs84_to_utm_projection():
    # Test equator and prime meridian
    x, y = wgs84_to_utm_approx(3.0, 0.0, 31)
    assert round(x, -2) == 500000.0  # Central meridian false easting
    assert round(y, -2) == 0.0

# ---------------------------------------------------------------------------
# 2. Raster Resampling & Spatial Alignment Tests
# ---------------------------------------------------------------------------

def test_raster_resampling_preserves_nodata():
    # 32x32 array with NaN block
    src = np.ones((32, 32), dtype=np.float32) * 0.45
    src[10:15, 10:15] = np.nan

    target_shape = (64, 64)
    resampled = resample_raster_to_canonical_grid(src, target_shape=target_shape, method="bilinear")

    assert resampled.shape == target_shape
    assert np.isnan(resampled[22, 22])  # Resampled center of NaN block must remain NaN
    assert not np.isnan(resampled[0, 0])
    assert abs(float(resampled[0, 0]) - 0.45) < 0.05

def test_align_all_multimodal_features(sample_geometry):
    grid = create_canonical_aoi_grid(sample_geometry, target_resolution_m=10.0)
    
    optical = {
        "B04": np.ones((30, 40), dtype=np.float32) * 0.1,
        "B08": np.ones((30, 40), dtype=np.float32) * 0.6,
        "SCL": np.ones((15, 20), dtype=np.float32) * 4,  # 20m band
    }
    sar = {
        "VV_LINEAR": np.ones((25, 35), dtype=np.float32) * 0.08,
        "VH_LINEAR": np.ones((25, 35), dtype=np.float32) * 0.015,
    }

    aligned_opt, aligned_sar = align_all_multimodal_features(optical, sar, grid)

    for name, arr in aligned_opt.items():
        assert arr.shape == (grid.height, grid.width), f"Optical {name} shape mismatch"

    for name, arr in aligned_sar.items():
        assert arr.shape == (grid.height, grid.width), f"SAR {name} shape mismatch"

# ---------------------------------------------------------------------------
# 3. Temporal Matching Engine Tests
# ---------------------------------------------------------------------------

def test_temporal_matching_closest_selection():
    service = MultimodalFusionService()
    s2_date = "2026-10-15"
    s1_catalog = [
        {"acquisition_date": "2026-10-10T00:30:00Z", "valid_percentage": 99.0, "scene_id": "S1_OCT10"},
        {"acquisition_date": "2026-10-14T00:30:00Z", "valid_percentage": 98.0, "scene_id": "S1_OCT14"}, # 1 day offset
        {"acquisition_date": "2026-10-17T00:30:00Z", "valid_percentage": 99.5, "scene_id": "S1_OCT17"}, # 2 day offset
    ]

    match = service.match_temporal_observations(s2_date, s1_catalog, temporal_window_days=3)

    assert match["matched"] is True
    assert match["sar_available"] is True
    assert match["sentinel1_date"] == "2026-10-14"
    assert match["temporal_gap_days"] == 1
    assert match["temporal_match_quality"] == "EXCELLENT"
    assert match["matched_scene"]["scene_id"] == "S1_OCT14"

def test_temporal_matching_outside_window():
    service = MultimodalFusionService()
    s2_date = "2026-10-15"
    s1_catalog = [
        {"acquisition_date": "2026-10-05T00:30:00Z", "valid_percentage": 99.0}, # 10 days
        {"acquisition_date": "2026-10-25T00:30:00Z", "valid_percentage": 99.0}, # 10 days
    ]

    # Test default 3 days window
    match = service.match_temporal_observations(s2_date, s1_catalog, temporal_window_days=3)

    assert match["matched"] is False
    assert match["sar_available"] is False
    assert match["sentinel1_date"] is None
    assert match["temporal_gap_days"] is None
    assert match["temporal_match_quality"] == "UNAVAILABLE"
    assert "unavailable for this time window" in match["message"]

def test_temporal_matching_empty_catalog():
    service = MultimodalFusionService()
    match = service.match_temporal_observations("2026-10-15", [], temporal_window_days=3)
    assert match["matched"] is False
    assert match["sar_available"] is False
    assert match["temporal_match_quality"] == "UNAVAILABLE"

# ---------------------------------------------------------------------------
# 4. Unified Quality Mask Tests
# ---------------------------------------------------------------------------

def test_unified_quality_mask_logic():
    service = MultimodalFusionService()
    shape = (10, 10)

    optical_mask = np.ones(shape, dtype=bool)
    optical_mask[0:3, 0:3] = False  # Cloud block in optical

    sar_mask = np.ones(shape, dtype=bool)
    sar_mask[7:10, 7:10] = False  # NoData block in SAR

    # Case A: Both optical and SAR available
    q_mask = service.build_unified_quality_mask(optical_mask, sar_mask, sar_available=True)
    assert q_mask[5, 5] == MASK_VALID    # Both valid
    assert q_mask[1, 1] == MASK_PARTIAL  # Optical invalid, SAR valid -> partial
    assert q_mask[8, 8] == MASK_PARTIAL  # Optical valid, SAR invalid -> partial

    optical_mask[0, 0] = False
    sar_mask[0, 0] = False
    q_mask2 = service.build_unified_quality_mask(optical_mask, sar_mask, sar_available=True)
    assert q_mask2[0, 0] == MASK_INVALID  # Both invalid

    # Case B: SAR completely unavailable
    q_mask_nosar = service.build_unified_quality_mask(optical_mask, None, sar_available=False)
    assert q_mask_nosar[5, 5] == MASK_PARTIAL  # Optical only is partial
    assert q_mask_nosar[1, 1] == MASK_INVALID  # Cloud is invalid

# ---------------------------------------------------------------------------
# 5. Feature Quality Statistics & Normalization Tests
# ---------------------------------------------------------------------------

def test_feature_quality_statistics_calculation():
    service = MultimodalFusionService()
    arr = np.array([[0.1, 0.2], [0.3, np.nan]], dtype=np.float32)
    stats = service.compute_feature_statistics(arr, valid_mask=None, feature_name="NDVI")

    assert stats["available"] is True
    assert stats["valid_pixel_count"] == 3
    assert stats["valid_pixel_percentage"] == 75.0
    assert stats["nodata_percentage"] == 25.0
    assert stats["min"] == 0.1
    assert stats["max"] == 0.3
    assert stats["mean"] == 0.2
    assert stats["median"] == 0.2

def test_normalization_methods():
    service = MultimodalFusionService()
    arr = np.array([[10.0, 20.0], [30.0, np.nan]], dtype=np.float32)
    features = {"VV": arr}

    # Raw
    norm_raw = service.normalize_feature_stack(features, method="raw")
    assert np.isnan(norm_raw["VV"][1, 1])
    assert norm_raw["VV"][0, 0] == 10.0

    # Min-Max
    norm_minmax = service.normalize_feature_stack(features, method="min-max")
    assert np.isnan(norm_minmax["VV"][1, 1])
    assert norm_minmax["VV"][0, 0] == 0.0  # Min is 10 -> 0
    assert norm_minmax["VV"][1, 0] == 1.0  # Max is 30 -> 1
    assert norm_minmax["VV"][0, 1] == 0.5  # Mid is 20 -> 0.5

    # Z-Score
    norm_z = service.normalize_feature_stack(features, method="z-score")
    assert np.isnan(norm_z["VV"][1, 1])
    assert abs(float(np.nanmean(norm_z["VV"]))) < 1e-4  # Zero-mean

# ---------------------------------------------------------------------------
# 6. End-to-End Multimodal Fusion Execution Test (Deterministic Demo)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_multimodal_fusion_demo(sample_geometry):
    service = MultimodalFusionService()
    result = await service.execute_multimodal_fusion(
        geometry=sample_geometry,
        start_date="2026-10-01",
        end_date="2026-10-25",
        temporal_window_days=3,
        normalization="raw",
    )

    assert result["status"] == "success"
    assert result["is_demo"] is True
    assert result["fusion_readiness"] in ("READY_FOR_AI", "PARTIAL_OPTICAL_ONLY")
    assert result["features_available_count"] >= 10
    assert result["total_features_count"] == 13

    # Check canonical grid
    grid = result["canonical_grid"]
    assert grid["epsg"] == 32644
    assert grid["width"] > 0
    assert grid["height"] > 0

    # Check temporal matching
    tm = result["temporal_matching"]
    assert tm["matched"] is True
    assert tm["sar_available"] is True
    assert tm["temporal_gap_days"] <= 3

    # Check feature statistics for 13 features
    stats = result["feature_statistics"]
    for feat in ("B02", "B04", "B08", "NDVI", "NDMI", "NDRE", "EVI", "VV", "VH", "VV_VH_RATIO"):
        assert feat in stats
        assert stats[feat]["valid_pixel_percentage"] > 0

    # Check previews
    previews = result["previews"]
    assert "quality_mask" in previews
    assert "ndvi" in previews
    assert "vv" in previews
    assert "vv_vh_ratio" in previews

# ---------------------------------------------------------------------------
# 7. Dataset Export (.npz, .json, GeoTIFF) Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dataset_export_formats(sample_geometry):
    service = MultimodalFusionService()
    record = await service.execute_multimodal_fusion(
        geometry=sample_geometry,
        start_date="2026-10-01",
        end_date="2026-10-25",
    )

    # 1. JSON Export
    json_export = service.export_dataset(record, export_format="json")
    assert isinstance(json_export, dict)
    assert "crs" in json_export
    assert json_export["labels_placeholder"] is None

    # 2. NPZ Export
    npz_bytes = service.export_dataset(record, export_format="npz")
    assert isinstance(npz_bytes, bytes)
    assert len(npz_bytes) > 1000

    bio = io.BytesIO(npz_bytes)
    npz_data = np.load(bio)
    assert "quality_mask" in npz_data
    assert "NDVI" in npz_data
    assert "VV" in npz_data
    assert "labels_placeholder" in npz_data
    assert npz_data["NDVI"].shape == (record["canonical_grid"]["height"], record["canonical_grid"]["width"])

    # 3. GeoTIFF Export
    tiff_bytes = service.export_dataset(record, export_format="geotiff")
    assert isinstance(tiff_bytes, bytes)
    assert len(tiff_bytes) > 1000

    tiff_bio = io.BytesIO(tiff_bytes)
    tiff_arr = tifffile.imread(tiff_bio)
    # Shape is (14, H, W) where 13 features + 1 quality mask
    assert tiff_arr.ndim == 3
    assert tiff_arr.shape[0] == 14
    assert tiff_arr.shape[1] == record["canonical_grid"]["height"]
    assert tiff_arr.shape[2] == record["canonical_grid"]["width"]

# ---------------------------------------------------------------------------
# 8. API Integration Tests (POST /api/v1/analysis/fusion and /export)
# ---------------------------------------------------------------------------

def test_api_fusion_endpoint(sample_geometry):
    response = client.post(
        "/api/v1/analysis/fusion",
        json={
            "geometry": sample_geometry,
            "start_date": "2026-10-01",
            "end_date": "2026-10-25",
            "temporal_window_days": 3,
            "normalization": "min-max",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["fusion_readiness"] in ("READY_FOR_AI", "PARTIAL_OPTICAL_ONLY")
    assert "feature_statistics" in data
    assert "previews" in data
    assert data["normalization_applied"] == "min-max"

def test_api_fusion_endpoint_lat_lon():
    response = client.post(
        "/api/v1/analysis/fusion",
        json={
            "latitude": 16.5062,
            "longitude": 80.6480,
            "radius_km": 2.0,
            "start_date": "2026-10-01",
            "end_date": "2026-10-25",
            "temporal_window_days": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "canonical_grid" in data
    assert "temporal_matching" in data

def test_api_fusion_export_endpoint(sample_geometry):
    # Test JSON export
    resp_json = client.post(
        "/api/v1/analysis/fusion/export",
        json={
            "geometry": sample_geometry,
            "start_date": "2026-10-01",
            "end_date": "2026-10-25",
            "export_format": "json",
        },
    )
    assert resp_json.status_code == 200
    assert "crs" in resp_json.json()

    # Test NPZ export binary
    resp_npz = client.post(
        "/api/v1/analysis/fusion/export",
        json={
            "geometry": sample_geometry,
            "start_date": "2026-10-01",
            "end_date": "2026-10-25",
            "export_format": "npz",
        },
    )
    assert resp_npz.status_code == 200
    assert resp_npz.headers["content-type"] == "application/octet-stream"
    assert len(resp_npz.content) > 1000

# ---------------------------------------------------------------------------
# 9. Historical Multimodal Fusion Timeline Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_historical_multimodal_timeline_integration(sample_geometry):
    result = await historical_monitoring_service.get_historical_time_series(
        geometry=sample_geometry,
        start_date="2026-06-01",
        end_date="2026-10-01",
    )
    assert result["status"] == "success"
    observations = result["observations"]
    assert len(observations) > 0

    # Verify each observation has multimodal optical + SAR properties
    for obs in observations:
        assert "ndvi" in obs
        assert "ndmi" in obs
        assert "ndre" in obs
        assert "evi" in obs
        assert "sar" in obs
        assert "quality" in obs
        assert "satellite_availability" in obs
        
        sar_info = obs["sar"]
        assert "available" in sar_info
        if sar_info["available"]:
            assert sar_info["vv"]["mean"] is not None
            assert sar_info["vh"]["mean"] is not None
            assert sar_info["vv_vh_ratio"]["mean"] is not None
        else:
            # If SAR unavailable, values must NOT be fabricated
            assert sar_info["vv"] is None
            assert sar_info["vh"] is None
            assert obs["quality"] == "Partial"
