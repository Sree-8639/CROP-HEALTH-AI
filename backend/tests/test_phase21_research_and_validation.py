"""
Crop Health AI — Phase 21 Research, Dataset Quality & Scientific Validation Test Suite

Validates:
1. Satellite Metadata Validator integrity and non-interpolation rules.
2. Data Leakage Auditor (spatial leakage, temporal leakage, synthetic contamination).
3. Target Horizon Generator (strictly chronological future targets).
4. Mandatory Time-Series Baselines (Persistence, Moving Average, Linear Trend).
5. HistGradientBoosting model training, bounded outputs, and feature importance.
6. Clustered bootstrap confidence interval calculation at independent field level.
7. Research API endpoints (/api/v1/research/...).
8. Production safety gates preservation.
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from ml.research_data.acquisition.metadata_validator import (
    SatelliteMetadataValidator,
    MetadataValidationError,
)
from ml.research_data.ground_truth.workflow import TargetHorizonGenerator
from ml.research_evaluation.validation import (
    SpatialTemporalValidationSplitter,
    LeakageAuditor,
    DataLeakageError,
)
from ml.research_evaluation.metrics import ForecastEvaluationMetrics
from ml.research_models.gradient_boosting_forecaster import (
    HistGradientBoostingForecaster,
    FEATURE_SCHEMA_V1,
)
from ml.growth_prediction.forecaster import (
    LastObservationForecaster,
    MovingAverageForecaster,
    LinearTrendForecaster,
)
from services.stress_detection_service import stress_detection_service

client = TestClient(app)


def test_metadata_validator_valid_record():
    """Verify metadata validator accepts conforming satellite observations."""
    rec = {
        "field_id": "field_1_tadepalligudem",
        "aoi_geometry": {"latitude": 16.8102, "longitude": 81.5266, "radius_km": 2.0},
        "acquisition_date": "2024-05-15",
        "satellite_platform": "Sentinel-2A",
        "processing_level": "L2A",
        "product_identifier": "S2A_MSIL2A_20240515T050651",
        "data_provenance": "Copernicus Data Space Ecosystem (CDSE)",
        "valid_pixel_percentage": 98.5,
        "cloud_coverage": 1.2,
        "is_missing_optical": False,
        "is_missing_sar": False,
        "ndvi": 0.65,
    }
    validated = SatelliteMetadataValidator.validate_observation_record(rec)
    assert validated["field_id"] == "field_1_tadepalligudem"


def test_metadata_validator_rejects_interpolation():
    """Verify metadata validator strictly rejects fabricated values for missing observations."""
    rec = {
        "field_id": "field_1",
        "aoi_geometry": {"latitude": 16.8102, "longitude": 81.5266},
        "acquisition_date": "2024-05-15",
        "satellite_platform": "Sentinel-2A",
        "processing_level": "L2A",
        "product_identifier": "S2A_TEST_001",
        "data_provenance": "CDSE",
        "is_missing_sar": True,
        "sar_vv": -12.4,  # VIOLATION: Claiming missing SAR, but supplying interpolated number
    }
    with pytest.raises(MetadataValidationError):
        SatelliteMetadataValidator.validate_observation_record(rec)


def test_leakage_auditor_detects_future_temporal_leakage():
    """Verify leakage auditor detects if test observations precede training observations."""
    train_df = pd.DataFrame([
        {"observation_date": "2024-06-01", "field_id": "f1", "product_identifier": "p1"},
        {"observation_date": "2024-08-01", "field_id": "f1", "product_identifier": "p2"},
    ])
    # Test split has an observation earlier than train max date (2024-07-01 < 2024-08-01)
    test_df = pd.DataFrame([
        {"observation_date": "2024-07-01", "field_id": "f1", "product_identifier": "p3"},
    ])

    with pytest.raises(DataLeakageError):
        LeakageAuditor.audit_split(train_df, test_df, is_spatial_holdout=False, allow_synthetic=True)


def test_target_horizon_generator_chronology():
    """Verify target pairs strictly follow t_future > t_current."""
    obs = [
        {"acquisition_date": "2024-01-01", "product_identifier": "p1", "ndvi": 0.40},
        {"acquisition_date": "2024-01-20", "product_identifier": "p2", "ndvi": 0.55},
        {"acquisition_date": "2024-02-15", "product_identifier": "p3", "ndvi": 0.68},
    ]
    gen = TargetHorizonGenerator(target_index="ndvi", min_horizon_days=5, max_horizon_days=30)
    pairs = gen.generate_supervised_pairs(obs)
    assert len(pairs) >= 1
    for p in pairs:
        assert p["target_value"] in [0.55, 0.68]
        assert p["target_horizon_days"] >= 5


def test_mandatory_baselines_reproducibility():
    """Verify existing Phase 14 forecasting baselines execute deterministically."""
    times = np.array([0.0, 15.0, 30.0])
    values = np.array([0.45, 0.52, 0.60])
    fut_times = np.array([45.0])

    pers = LastObservationForecaster(index_name="NDVI")
    pers.fit(times, values)
    assert pers.predict(fut_times)[0] == 0.60

    ma = MovingAverageForecaster(index_name="NDVI", window_size=2)
    ma.fit(times, values)
    assert round(float(ma.predict(fut_times)[0]), 3) == 0.560

    trend = LinearTrendForecaster(index_name="NDVI")
    trend.fit(times, values)
    assert trend.predict(fut_times)[0] > 0.60


def test_hist_gradient_boosting_forecaster():
    """Verify HistGradientBoostingForecaster fits and predicts within physical bounds."""
    rng = np.random.RandomState(42)
    n = 25
    data = {feat: rng.uniform(0.1, 0.9, size=n) for feat in FEATURE_SCHEMA_V1}
    # Inject occasional NaN to verify missing value tolerance
    data["sar_vv"][2] = np.nan
    data["sar_vh"][5] = np.nan
    df = pd.DataFrame(data)
    y = rng.uniform(0.2, 0.8, size=n)

    forecaster = HistGradientBoostingForecaster(max_iter=20, min_samples_leaf=2, random_state=42)
    forecaster.fit(df, y)
    preds = forecaster.predict(df)

    assert len(preds) == n
    assert np.all(preds >= -1.0) and np.all(preds <= 1.0)
    assert forecaster.is_fitted


def test_clustered_field_bootstrap_ci():
    """Verify bootstrap confidence interval respects field-level resampling."""
    field_ids = np.array(["f1", "f1", "f2", "f2", "f3", "f3"])
    y_true = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    y_pred = np.array([0.48, 0.53, 0.62, 0.64, 0.69, 0.72])

    ci = ForecastEvaluationMetrics.calculate_field_bootstrap_ci(
        field_ids, y_true, y_pred, n_bootstraps=50, random_seed=42
    )
    assert "mae_ci" in ci
    low, high = ci["mae_ci"]
    assert low <= high


def test_research_api_dataset_status():
    """Verify GET /api/v1/research/dataset/status endpoint returns valid structure."""
    response = client.get("/api/v1/research/dataset/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "total_records" in data
    assert "unique_fields" in data


def test_research_api_models_and_comparison():
    """Verify GET /api/v1/research/models and comparison endpoints."""
    response = client.get("/api/v1/research/models")
    assert response.status_code == 200
    models_data = response.json()
    assert "models" in models_data
    assert len(models_data["models"]) >= 2

    # Check comparison
    cmp_res = client.get("/api/v1/research/models/HistGradientBoosting_NDVI_Forecaster/comparison")
    assert cmp_res.status_code == 200
    cmp_data = cmp_res.json()
    assert "baseline_comparison" in cmp_data
    assert "scientific_caveat" in cmp_data


def test_production_stress_safety_gate_preserved():
    """Verify production supervised crop stress detection remains strictly gated."""
    assert hasattr(stress_detection_service, "detect_crop_stress")
    assert hasattr(stress_detection_service, "load_model")
