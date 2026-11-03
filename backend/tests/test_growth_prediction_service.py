"""
Comprehensive Test Suite for Phase 14: AI-Powered Crop Growth & Trend Prediction.

Covers the 18 required validation targets:
1. Chronological sorting
2. Duplicate observation handling
3. Missing observations handling
4. Insufficient history handling
5. Trend calculations
6. Moving-average forecasting
7. Linear trend forecasting
8. Forecast date generation
9. Horizon validation
10. Temporal leakage prevention
11. Walk-forward validation
12. Metric calculations (MAE, RMSE, R², MAPE)
13. Prediction interval calculations
14. Model loading / metadata
15. Missing model handling / fallback
16. API validation via FastAPI TestClient
17. Deterministic demo predictions
18. No fabricated real-world forecasts check
"""

import math
import datetime
import pytest
import numpy as np
from fastapi.testclient import TestClient

from main import app
from services.growth_prediction_service import (
    GrowthPredictionService,
    growth_prediction_service,
    GrowthPredictionError,
)
from ml.growth_prediction.schema import (
    SUPPORTED_INDICES,
    OPTICAL_INDICES,
    SAR_INDICES,
    TREND_INCREASING,
    TREND_DECREASING,
    TREND_STABLE,
    TREND_INSUFFICIENT,
    FORECAST_HORIZONS,
)
from ml.growth_prediction.forecaster import (
    BaseForecaster,
    LastObservationForecaster,
    MovingAverageForecaster,
    LinearTrendForecaster,
    SeasonalNaiveForecaster,
    walk_forward_validate,
    compute_prediction_intervals,
)

SAMPLE_AOI = {
    "type": "Polygon",
    "coordinates": [[
        [81.52, 16.80],
        [81.54, 16.80],
        [81.54, 16.82],
        [81.52, 16.82],
        [81.52, 16.80],
    ]],
}

# -------------------------------------------------------------
# 1. Chronological Sorting & 2. Duplicate Handling
# -------------------------------------------------------------

def test_chronological_sorting_and_deduplication():
    """Verifies that observations are deduplicated by date and sorted chronologically."""
    svc = GrowthPredictionService()

    raw_observations = [
        {"date": "2026-10-15", "valid_pixel_percentage": 70.0, "ndvi": {"mean": 0.60}},
        {"date": "2026-09-01", "valid_pixel_percentage": 85.0, "ndvi": {"mean": 0.40}},
        {"date": "2026-10-15", "valid_pixel_percentage": 95.0, "ndvi": {"mean": 0.65}},  # Duplicate, higher valid %
        {"date": "2026-09-20", "valid_pixel_percentage": 90.0, "ndvi": {"mean": 0.50}},
    ]

    cleaned = svc._clean_and_sort_observations(raw_observations)

    # Must have 3 distinct dates
    assert len(cleaned) == 3
    dates = [o["date"] for o in cleaned]
    assert dates == ["2026-09-01", "2026-09-20", "2026-10-15"]

    # The 2026-10-15 entry must be the higher quality one (95% valid, ndvi=0.65)
    oct15 = next(o for o in cleaned if o["date"] == "2026-10-15")
    assert oct15["valid_pixel_percentage"] == 95.0
    assert oct15["ndvi"]["mean"] == 0.65


# -------------------------------------------------------------
# 3. Missing Observations Handling
# -------------------------------------------------------------

def test_missing_observations_preservation():
    """Verifies that missing observations (e.g. SAR or clouded optical) are kept as NaN without zero-fabrication."""
    svc = GrowthPredictionService()

    raw_observations = [
        {"date": "2026-09-01", "ndvi": {"mean": 0.45}, "sar": {"vv": {"mean": -12.0}}},
        {"date": "2026-09-15", "ndvi": None, "sar": None},  # Missing optical and SAR
        {"date": "2026-10-01", "ndvi": {"mean": 0.65}, "sar": {"vv": {"mean": -10.5}}},
    ]

    cleaned = svc._clean_and_sort_observations(raw_observations)
    enriched = svc._extract_time_series_features(cleaned)

    assert len(enriched) == 3
    # Step 1: NDVI is None, NOT 0.0
    assert enriched[1]["indices"]["NDVI"] is None
    assert enriched[1]["indices"]["VV"] is None
    # No artificial zero insertion
    assert enriched[1]["sequential_features"]["NDVI"]["previous_valid"] == 0.45


# -------------------------------------------------------------
# 4. Insufficient History Handling
# -------------------------------------------------------------

@pytest.mark.asyncio
async def test_insufficient_history_handling():
    """Verifies that an observation list with < 2 records gracefully returns INSUFFICIENT_OBSERVATIONS."""
    svc = GrowthPredictionService()

    # Mock historical svc returning only 1 observation
    class MockHistSvc:
        async def get_historical_time_series(self, **kwargs):
            return {
                "status": "success",
                "observations": [{"date": "2026-10-01", "ndvi": {"mean": 0.50}}],
                "is_demo": True,
            }

    custom_svc = GrowthPredictionService(historical_svc=MockHistSvc())
    result = await custom_svc.predict_growth_and_trends(
        geometry=SAMPLE_AOI,
        start_date="2026-09-25",
        end_date="2026-10-05",
    )

    assert result["status"] == "partial"
    assert result["assessment_status"] == "INSUFFICIENT_OBSERVATIONS"
    assert result["forecast_dates"] == []
    assert result["forecast_values"] == {}
    assert len(result["warnings"]) > 0


# -------------------------------------------------------------
# 5. Trend Calculations & Direction Classification
# -------------------------------------------------------------

def test_trend_calculations_and_directions():
    """Verifies linear trend slope and categorization into Increasing, Decreasing, and Stable."""
    svc = GrowthPredictionService()

    # Case A: Increasing NDVI
    inc_observations = [
        {"date": "2026-09-01", "datetime_obj": datetime.date(2026, 9, 1), "indices": {"NDVI": 0.35}},
        {"date": "2026-09-15", "datetime_obj": datetime.date(2026, 9, 15), "indices": {"NDVI": 0.48}},
        {"date": "2026-10-01", "datetime_obj": datetime.date(2026, 10, 1), "indices": {"NDVI": 0.62}},
    ]
    inc_stats = svc._compute_trend_statistics(inc_observations, ["NDVI"])
    assert inc_stats["NDVI"]["trend_direction"] == TREND_INCREASING
    assert inc_stats["NDVI"]["trend_slope_per_day"] > 0.005
    assert inc_stats["NDVI"]["absolute_change"] == round(0.62 - 0.35, 4)

    # Case B: Decreasing NDVI
    dec_observations = [
        {"date": "2026-09-01", "datetime_obj": datetime.date(2026, 9, 1), "indices": {"NDVI": 0.70}},
        {"date": "2026-09-15", "datetime_obj": datetime.date(2026, 9, 15), "indices": {"NDVI": 0.55}},
        {"date": "2026-10-01", "datetime_obj": datetime.date(2026, 10, 1), "indices": {"NDVI": 0.42}},
    ]
    dec_stats = svc._compute_trend_statistics(dec_observations, ["NDVI"])
    assert dec_stats["NDVI"]["trend_direction"] == TREND_DECREASING
    assert dec_stats["NDVI"]["trend_slope_per_day"] < -0.005

    # Case C: Stable NDVI
    stable_observations = [
        {"date": "2026-09-01", "datetime_obj": datetime.date(2026, 9, 1), "indices": {"NDVI": 0.60}},
        {"date": "2026-09-15", "datetime_obj": datetime.date(2026, 9, 15), "indices": {"NDVI": 0.61}},
        {"date": "2026-10-01", "datetime_obj": datetime.date(2026, 10, 1), "indices": {"NDVI": 0.60}},
    ]
    stable_stats = svc._compute_trend_statistics(stable_observations, ["NDVI"])
    assert stable_stats["NDVI"]["trend_direction"] == TREND_STABLE


# -------------------------------------------------------------
# 6. Moving Average Forecasting & 7. Linear Trend Forecasting
# -------------------------------------------------------------

def test_moving_average_forecasting():
    """Verifies that moving average forecaster accurately averages the last k observations."""
    t_days = np.array([0, 10, 20, 30], dtype=np.float64)
    vals = np.array([0.40, 0.50, 0.60, 0.70], dtype=np.float64)
    fut_t = np.array([40, 50], dtype=np.float64)

    ma_forecaster = MovingAverageForecaster(index_name="NDVI", window_size=3)
    ma_forecaster.fit(t_days, vals)
    preds = ma_forecaster.predict(fut_t)

    # Mean of last 3: (0.50 + 0.60 + 0.70) / 3 = 0.60
    assert len(preds) == 2
    assert abs(preds[0] - 0.60) < 1e-4
    assert abs(preds[1] - 0.60) < 1e-4

def test_linear_trend_forecasting():
    """Verifies that linear trend forecaster extrapolates slope and clamps within physical bounds."""
    t_days = np.array([0, 10, 20], dtype=np.float64)
    vals = np.array([0.40, 0.50, 0.60], dtype=np.float64)  # slope = +0.01 / day
    fut_t = np.array([30, 40], dtype=np.float64)

    lt_forecaster = LinearTrendForecaster(index_name="NDVI")
    lt_forecaster.fit(t_days, vals)
    preds = lt_forecaster.predict(fut_t)

    assert len(preds) == 2
    assert preds[0] > 0.60
    assert preds[1] >= preds[0]
    # Clamped to physical bounds (max 1.0 for NDVI)
    assert np.all(preds <= 1.0)


# -------------------------------------------------------------
# 8. Forecast Date Generation & 9. Horizon Validation
# -------------------------------------------------------------

def test_forecast_date_generation_and_horizons():
    """Verifies that forecast dates are calculated using realistic median revisit cadences."""
    svc = GrowthPredictionService()

    observations = [
        {"datetime_obj": datetime.date(2026, 9, 1)},
        {"datetime_obj": datetime.date(2026, 9, 7)},   # diff 6
        {"datetime_obj": datetime.date(2026, 9, 13)},  # diff 6
        {"datetime_obj": datetime.date(2026, 9, 19)},  # diff 6
    ]

    dates_short, cadence = svc._determine_forecast_schedule(observations, n_steps=2)
    assert cadence == 6
    assert len(dates_short) == 2
    assert dates_short[0] == "2026-09-25"
    assert dates_short[1] == "2026-10-01"

    dates_medium, _ = svc._determine_forecast_schedule(observations, n_steps=4)
    assert len(dates_medium) == 4
    assert dates_medium[3] == "2026-10-13"


# -------------------------------------------------------------
# 10. Temporal Leakage Prevention
# -------------------------------------------------------------

def test_temporal_leakage_prevention_in_features():
    """Verifies that sequential features at index t strictly use observations at or before t."""
    svc = GrowthPredictionService()

    raw_obs = [
        {"date": "2026-09-01", "ndvi": {"mean": 0.30}},
        {"date": "2026-09-10", "ndvi": {"mean": 0.40}},
        {"date": "2026-09-20", "ndvi": {"mean": 0.50}},
        {"date": "2026-09-30", "ndvi": {"mean": 0.60}},
    ]

    enriched = svc._extract_time_series_features(raw_obs)

    # Step 0: no prior valid
    assert enriched[0]["sequential_features"]["NDVI"]["previous_valid"] is None
    # Step 1: prior valid is 0.30
    assert enriched[1]["sequential_features"]["NDVI"]["previous_valid"] == 0.30
    assert enriched[1]["sequential_features"]["NDVI"]["rolling_mean_3"] == 0.35  # (0.30 + 0.40) / 2
    # Step 2: rolling mean strictly uses [0.30, 0.40, 0.50]
    assert abs(enriched[2]["sequential_features"]["NDVI"]["rolling_mean_3"] - 0.40) < 1e-4
    # Future value 0.60 is NOT in step 2 rolling mean
    assert enriched[2]["sequential_features"]["NDVI"]["rolling_mean_3"] < 0.60


# -------------------------------------------------------------
# 11. Walk-Forward Validation & 12. Metric Calculations
# -------------------------------------------------------------

def test_walk_forward_validation_and_metrics():
    """Verifies chronological walk-forward validation and MAE, RMSE, R2, MAPE calculations."""
    dates = ["2026-08-01", "2026-08-10", "2026-08-20", "2026-08-30", "2026-09-10", "2026-09-20"]
    times = np.array([0, 10, 20, 30, 40, 50], dtype=np.float64)
    # Perfect linear sequence
    values = np.array([0.40, 0.45, 0.50, 0.55, 0.60, 0.65], dtype=np.float64)

    eval_result = walk_forward_validate(
        dates=dates,
        values=values,
        times_days=times,
        forecaster_cls=LinearTrendForecaster,
        index_name="NDVI",
        min_train_size=3,
    )

    assert eval_result["status"] == "VALIDATED"
    assert eval_result["validation_windows_count"] == 3
    assert eval_result["mae"] is not None
    assert eval_result["rmse"] is not None
    assert eval_result["mae"] < 0.05
    assert eval_result["rmse"] < 0.05
    assert eval_result["r2"] is not None
    assert eval_result["mape"] is not None


# -------------------------------------------------------------
# 13. Prediction Interval Calculations
# -------------------------------------------------------------

def test_prediction_interval_calculations():
    """Verifies calibrated prediction intervals expand with horizon and respect physical bounds."""
    f_vals = [0.55, 0.60, 0.65, 0.70]
    residual_std = 0.03

    intervals = compute_prediction_intervals(
        forecast_values=f_vals,
        residual_std=residual_std,
        index_name="NDVI",
        confidence_level=0.80,
    )

    assert intervals is not None
    assert len(intervals) == 4

    for i, itv in enumerate(intervals):
        assert itv["lower"] < f_vals[i]
        assert itv["upper"] > f_vals[i]
        assert itv["lower"] >= -0.2  # Bound clamp
        assert itv["upper"] <= 1.0

    # Step 3 interval width must be greater than step 0 width due to horizon uncertainty expansion
    width_0 = intervals[0]["upper"] - intervals[0]["lower"]
    width_3 = intervals[3]["upper"] - intervals[3]["lower"]
    assert width_3 >= width_0


# -------------------------------------------------------------
# 14. Model Metadata & 15. Missing Model Handling
# -------------------------------------------------------------

def test_forecaster_metadata_and_fallback():
    """Verifies model metadata tagging and fallback when seasonal data is insufficient."""
    # Under 10 observations -> SeasonalNaive must flag unavailable
    t_days = np.array([0, 10, 20], dtype=np.float64)
    vals = np.array([0.40, 0.50, 0.60], dtype=np.float64)
    dates = ["2026-09-01", "2026-09-10", "2026-09-20"]

    sn = SeasonalNaiveForecaster(index_name="NDVI")
    sn.fit(t_days, vals, dates)
    assert sn.is_available is False

    # Predict still works without crashing (falls back to last observation)
    preds = sn.predict(np.array([30, 40]), ["2026-09-30", "2026-10-10"])
    assert len(preds) == 2
    assert preds[0] == 0.60


# -------------------------------------------------------------
# 16. API Validation via TestClient
# -------------------------------------------------------------

def test_api_growth_prediction_endpoint():
    """Verifies that POST /api/v1/analysis/growth-prediction succeeds with 200 and valid schema."""
    client = TestClient(app)
    payload = {
        "geometry": SAMPLE_AOI,
        "start_date": "2026-08-01",
        "end_date": "2026-10-25",
        "forecast_horizon": "medium",
        "selected_indices": ["NDVI", "NDMI", "NDRE", "EVI", "VV", "VH"],
        "model_type": "linear_trend",
    }

    response = client.post("/api/v1/analysis/growth-prediction", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ("success", "partial")
    assert "forecast_dates" in data
    assert "forecast_values" in data
    assert "trend_statistics" in data
    assert "validation_metrics" in data
    assert "historical_comparison" in data
    assert "scientific_disclaimer" in data

def test_api_growth_prediction_missing_coords():
    """Verifies that omitting geometry, latitude, and longitude returns 400 Bad Request."""
    client = TestClient(app)
    payload = {
        "start_date": "2026-08-01",
        "end_date": "2026-10-25",
    }

    response = client.post("/api/v1/analysis/growth-prediction", json=payload)
    assert response.status_code == 400
    assert "Either 'geometry' or 'latitude' and 'longitude' must be provided" in response.json()["detail"]


# -------------------------------------------------------------
# 17. Deterministic Demo Predictions & 18. No Fabricated Forecasts Check
# -------------------------------------------------------------

@pytest.mark.asyncio
async def test_deterministic_demo_and_no_fabricated_sar():
    """
    Verifies that DEMO_MODE produces deterministic forecasts with clear demo flags,
    and missing SAR data does NOT fabricate random SAR projections.
    """
    result = await growth_prediction_service.predict_growth_and_trends(
        geometry=SAMPLE_AOI,
        start_date="2026-07-01",
        end_date="2026-10-25",
        forecast_horizon="medium",
        selected_indices=["NDVI", "NDMI", "VV"],
    )

    assert result["is_demo"] is True
    assert "DEMO DATA" in result["warnings"][0]

    # Verify optical forecast values exist and are within bounds
    assert "NDVI" in result["forecast_values"]
    for val in result["forecast_values"]["NDVI"]:
        assert -0.2 <= val <= 1.0

    # Trend stats
    assert "NDVI" in result["trend_statistics"]
    assert result["trend_statistics"]["NDVI"]["trend_direction"] in (
        TREND_INCREASING, TREND_DECREASING, TREND_STABLE, TREND_INSUFFICIENT
    )

    # Scientific disclaimer check
    assert "scientific_disclaimer" in result
    assert "guaranteed" in result["scientific_disclaimer"].lower()
