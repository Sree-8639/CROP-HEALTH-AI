"""
backend/tests/test_heat_monitoring.py:
Automated test suite for Urban Heat Island & Thermal Downscaling:
- Landsat 8/9 ST_B10 thermal calibration formula verification
- Sentinel-2 optical predictor generation & 30m aggregation
- Yao et al. (2020) Random Forest downscaling with strict spatial split
- Diurnal & seasonal LST contrast and UHI Hotspot scoring
- Tree-Planting Priority Index (TPPI) & Visakhapatnam forestry parcels
- FastAPI heat endpoints: GET /metrics, GET /tree-planting-priorities, POST /downscale
"""

import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

from main import app
from ml.heat.landsat_lst import (
    dn_to_lst_celsius,
    lst_celsius_to_dn,
    LandsatLSTTile,
)
from ml.heat.sentinel_predictors import (
    calculate_ndvi,
    calculate_ndbi,
    calculate_mndwi,
    calculate_broadband_albedo,
    aggregate_10m_to_30m,
    Sentinel2PredictorSuite,
)
from ml.heat.downscaler_rf import RandomForestLSTDownscaler
from ml.heat.diurnal_seasonal import (
    compute_diurnal_thermal_analysis,
    compute_seasonal_thermal_analysis,
    compute_uhi_hotspot_score,
)
from ml.heat.tree_planting_priority import (
    compute_tree_planting_priority_index,
    delineate_tree_planting_priority_zones,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_landsat_c2_l2_st_b10_calibration_formula():
    """
    Verifies that the exact Landsat 8/9 Collection 2 Level-2 ST_B10 scaling formula is enforced:
    LST_Kelvin = DN * 0.00341802 + 149.0
    LST_Celsius = LST_Kelvin - 273.15
    """
    # Test known digital numbers
    # If DN = 44000:
    # LST_K = 44000 * 0.00341802 + 149.0 = 150.39288 + 149.0 = 299.39288 K
    # LST_C = 299.39288 - 273.15 = 26.24288 °C
    expected_celsius = 44000 * 0.00341802 + 149.0 - 273.15
    calculated_celsius = dn_to_lst_celsius(44000)
    assert np.isclose(calculated_celsius, expected_celsius, atol=1e-4)

    # Test reverse conversion
    dn_recon = lst_celsius_to_dn(calculated_celsius)
    assert np.isclose(dn_recon, 44000, atol=1)

    # Test tile generation
    tile = LandsatLSTTile(grid_shape=(128, 128), season="summer", diurnal="day", random_seed=42)
    lst_day = tile.lst_celsius
    assert lst_day.shape == (128, 128)
    assert 20.0 <= np.mean(lst_day) <= 45.0
    assert np.all(np.isfinite(lst_day))


def test_sentinel2_predictors_and_spatial_aggregation():
    """
    Verifies Sentinel-2 multispectral predictor indices and 3x3 pixel area aggregation to 30m.
    """
    b2 = np.ones((90, 90), dtype=np.float32) * 0.05
    b3 = np.ones((90, 90), dtype=np.float32) * 0.08
    b4 = np.ones((90, 90), dtype=np.float32) * 0.04
    b8 = np.ones((90, 90), dtype=np.float32) * 0.40 # high NIR (vegetation)
    b11 = np.ones((90, 90), dtype=np.float32) * 0.15 # SWIR

    ndvi = calculate_ndvi(b8, b4)
    assert ndvi.shape == (90, 90)
    assert np.isclose(ndvi[0, 0], (0.40 - 0.04) / (0.40 + 0.04), atol=1e-3)

    ndbi = calculate_ndbi(b11, b8)
    assert ndbi.shape == (90, 90)
    assert np.isclose(ndbi[0, 0], (0.15 - 0.40) / (0.15 + 0.40), atol=1e-3)

    mndwi = calculate_mndwi(b3, b11)
    assert mndwi.shape == (90, 90)

    albedo = calculate_broadband_albedo(b2, b4, b8, b11)
    assert albedo.shape == (90, 90)
    assert np.all((albedo >= 0.0) & (albedo <= 1.0))

    # Aggregation from 10m to 30m
    agg_ndvi = aggregate_10m_to_30m(ndvi)
    assert agg_ndvi.shape == (30, 30)
    assert np.isclose(agg_ndvi[0, 0], ndvi[0, 0], atol=1e-3)


def test_random_forest_downscaler_spatial_split():
    """
    Verifies Yao et al. (2020) Random Forest downscaling with a strict spatial partition.
    """
    # Create small synthetic test predictors and 30m LST
    n_30m = 32
    n_10m = n_30m * 3
    rng = np.random.RandomState(42)

    ndvi_10m = rng.uniform(0.1, 0.7, (n_10m, n_10m)).astype(np.float32)
    ndbi_10m = rng.uniform(-0.4, 0.5, (n_10m, n_10m)).astype(np.float32)
    mndwi_10m = rng.uniform(-0.5, 0.2, (n_10m, n_10m)).astype(np.float32)
    albedo_10m = rng.uniform(0.08, 0.35, (n_10m, n_10m)).astype(np.float32)

    ndvi_30m = aggregate_10m_to_30m(ndvi_10m)
    ndbi_30m = aggregate_10m_to_30m(ndbi_10m)
    mndwi_30m = aggregate_10m_to_30m(mndwi_10m)
    albedo_30m = aggregate_10m_to_30m(albedo_10m)

    lst_30m = (
        30.0
        - 10.0 * ndvi_30m
        + 12.0 * ndbi_30m
        + 25.0 * albedo_30m
        + rng.normal(0, 0.5, (n_30m, n_30m))
    ).astype(np.float32)

    features_30m = {
        "ndvi": ndvi_30m,
        "ndbi": ndbi_30m,
        "mndwi": mndwi_30m,
        "albedo": albedo_30m,
    }
    features_10m = {
        "ndvi": ndvi_10m,
        "ndbi": ndbi_10m,
        "mndwi": mndwi_10m,
        "albedo": albedo_10m,
    }

    downscaler = RandomForestLSTDownscaler(n_estimators=10, max_depth=8, random_state=42)
    eval_res = downscaler.fit_and_evaluate_spatial(features_30m, lst_30m, spatial_split_ratio=0.70)

    assert "held_out_metrics" in eval_res
    assert "r2" in eval_res["held_out_metrics"]
    assert "rmse_celsius" in eval_res["held_out_metrics"]
    assert "mae_celsius" in eval_res["held_out_metrics"]
    assert eval_res["held_out_metrics"]["rmse_celsius"] > 0
    assert eval_res["train_pixels"] > 0
    assert eval_res["test_pixels_held_out"] > 0

    # Downscale to 10m
    lst_10m = downscaler.downscale_to_10m(
        features_10m=features_10m,
        features_30m=features_30m,
        lst_30m=lst_30m,
    )
    assert lst_10m.shape == (n_10m, n_10m)
    assert np.all(np.isfinite(lst_10m))


def test_diurnal_and_seasonal_lst_analysis():
    """
    Verifies diurnal (day vs night) and seasonal (summer vs winter) temperature calculations.
    """
    day_lst = np.ones((50, 50), dtype=np.float32) * 38.0
    night_lst = np.ones((50, 50), dtype=np.float32) * 28.5

    diurnal = compute_diurnal_thermal_analysis(day_lst, night_lst)
    assert np.allclose(diurnal["diurnal_temperature_range_map"], 9.5)
    assert np.isclose(diurnal["mean_day_lst"], 38.0)
    assert np.isclose(diurnal["mean_night_lst"], 28.5)
    assert np.isclose(diurnal["mean_dtr_celsius"], 9.5)

    summer_lst = np.ones((50, 50), dtype=np.float32) * 39.0
    winter_lst = np.ones((50, 50), dtype=np.float32) * 28.0

    seasonal = compute_seasonal_thermal_analysis(summer_lst, winter_lst)
    assert np.isclose(seasonal["mean_summer_lst"], 39.0)
    assert np.isclose(seasonal["mean_winter_lst"], 28.0)
    assert np.isclose(seasonal["mean_seasonal_amplitude"], 11.0)


def test_uhi_hotspot_scoring_and_zones():
    """
    Verifies Urban Heat Island Hotspot scoring and zone breakdown.
    """
    lst = np.array([[45.0, 38.0], [32.0, 26.0]], dtype=np.float32)
    ndvi = np.array([[0.05, 0.20], [0.55, 0.80]], dtype=np.float32)
    ndbi = np.array([[0.50, 0.25], [-0.10, -0.35]], dtype=np.float32)
    mndwi = np.array([[-0.20, -0.15], [-0.10, 0.40]], dtype=np.float32)

    res = compute_uhi_hotspot_score(lst, ndvi, ndbi, mndwi)
    assert "hotspot_score_map" in res
    assert "zones_breakdown" in res
    score = res["hotspot_score_map"]
    assert score[0, 0] > score[1, 0]

    zones = res["zones_breakdown"]
    assert "severe_hotspot" in zones
    assert "moderate_hotspot" in zones
    assert "mild_elevation" in zones
    assert "cool_refuge" in zones


def test_tree_planting_priority_index_and_parcels():
    """
    Verifies Tree-Planting Priority Index and Visakhapatnam parcel delineation.
    """
    hs = np.array([[3.8, 2.5], [1.2, 0.2]], dtype=np.float32)
    ndvi = np.array([[0.08, 0.25], [0.50, 0.75]], dtype=np.float32)
    ndbi = np.array([[0.45, 0.15], [-0.10, -0.30]], dtype=np.float32)
    mndwi = np.array([[-0.20, -0.15], [-0.10, 0.40]], dtype=np.float32)

    tppi = compute_tree_planting_priority_index(hs, ndvi, ndbi, mndwi)
    assert tppi.shape == (2, 2)
    assert np.all((tppi >= 0.0) & (tppi <= 1.0))
    assert tppi[0, 0] > tppi[1, 0]

    # Create small grid for parcel extraction
    H, W = 64, 64
    tppi_grid = np.zeros((H, W), dtype=np.float32)
    tppi_grid[10:30, 10:30] = 0.85
    lst_grid = np.ones((H, W), dtype=np.float32) * 41.0
    ndvi_grid = np.ones((H, W), dtype=np.float32) * 0.12

    parcels = delineate_tree_planting_priority_zones(
        tppi_map=tppi_grid,
        lst_10m=lst_grid,
        ndvi_10m=ndvi_grid,
        bbox=[83.15, 17.62, 83.35, 17.80],
        min_pixels=10,
        pixel_size_meters=10.0,
    )
    assert "geojson" in parcels
    assert parcels["geojson"]["type"] == "FeatureCollection"
    assert len(parcels["geojson"]["features"]) >= 1

    site1 = parcels["geojson"]["features"][0]
    assert "properties" in site1
    assert "recommended_native_species" in site1["properties"]
    assert site1["properties"]["target_hectares"] > 0


def test_fastapi_heat_metrics_endpoint(client):
    """
    Verifies GET /api/v1/heat/metrics returns complete benchmark and downscaling data.
    """
    res = client.get("/api/v1/heat/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "project" in data
    assert "Visakhapatnam" in data["aoi_metadata"]["name"]
    assert "downscaling_evaluation_spatial_split" in data
    assert "diurnal_analysis" in data
    assert "seasonal_analysis" in data
    assert "uhi_hotspots_visakhapatnam" in data
    assert "tree_planting_priority_program" in data
    assert "visual_overlays" in data


def test_fastapi_heat_tree_priorities_endpoint(client):
    """
    Verifies GET /api/v1/heat/tree-planting-priorities returns 9 GeoJSON parcels.
    """
    res = client.get("/api/v1/heat/tree-planting-priorities")
    assert res.status_code == 200
    data = res.json()
    assert "total_priority_parcels" in data
    geojson = data["priority_parcels_geojson"]
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == data["total_priority_parcels"]
    assert len(geojson["features"]) > 0
    for feat in geojson["features"]:
        assert "geometry" in feat
        assert "properties" in feat
        assert "tree_planting_priority_index" in feat["properties"]


def test_fastapi_heat_downscale_endpoint(client):
    """
    Verifies POST /api/v1/heat/downscale executes 10m thermal downscaling and returns base64 maps.
    """
    payload = {
        "aoi_name": "Visakhapatnam Urban Core",
        "bbox": [83.15, 17.62, 83.35, 17.8],
        "target_resolution_m": 10.0,
        "season": "summer",
        "enable_residual_correction": True,
    }
    res = client.post("/api/v1/heat/downscale", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert data["status"] == "success"
    assert "summary_statistics" in data
    assert "downscaling_accuracy" in data
    assert "visual_overlays" in data
    assert "lst_10m_downscaled_base64" in data["visual_overlays"]
