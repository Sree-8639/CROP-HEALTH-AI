"""
Comprehensive Test Suite for Phase 17: Advanced Farmer Intelligence Dashboard.

Covers:
1. Dashboard overview generation
2. Field selection
3. Missing field handling (404)
4. Historical observation integration
5. Vegetation metric integration & deltas (NDVI, NDMI, NDRE, EVI)
6. Stress model availability & production ground truth advisory
7. Growth forecast integration & prediction intervals
8. Field risk statistics & spatial layers integration
9. Recommendation evidence & non-prescriptive rule validation
10. Insufficient evidence handling
11. Model provenance & versioning
12. Demo vs production separation
13. Field comparison validation & temporal mismatch warnings
14. API endpoint response validation
15. No fabricated metrics or diagnostic claims
16. Regression compatibility with Phases 1–16
"""

import os
import pytest
from fastapi.testclient import TestClient

from main import app
from services.farmer_dashboard_service import (
    farmer_dashboard_service,
    FarmerDashboardService,
    FieldNotFoundError,
)
from services.agricultural_recommendation_service import (
    agricultural_recommendation_service,
    AgriculturalRecommendationService,
)
from ml.farmer_dashboard.schema import (
    DashboardOverviewResponse,
    FarmerDashboardResponse,
    FieldComparisonResponse,
    RegisterFieldRequest,
)

client = TestClient(app)


@pytest.fixture
def dashboard_service():
    return FarmerDashboardService()


@pytest.fixture
def recommendation_service():
    return AgriculturalRecommendationService()


# -----------------------------------------------------------------------------
# 1. Dashboard Overview Generation
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dashboard_overview_generation(dashboard_service):
    overview = await dashboard_service.get_overview()
    assert isinstance(overview, DashboardOverviewResponse)
    assert overview.status == "success"
    assert overview.total_registered_fields >= 3
    assert len(overview.fields) >= 3
    assert overview.active_field_id in [f.field_id for f in overview.fields]

    first_f = overview.fields[0]
    assert first_f.field_id == "field_1"
    assert "Tadepalligudem" in first_f.field_name
    assert first_f.area_hectares > 0.0
    assert first_f.overall_risk_level in ["Low", "Moderate", "Elevated", "Insufficient Evidence"]
    assert first_f.mean_ndvi is not None


# -----------------------------------------------------------------------------
# 2. Field Selection
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_field_selection(dashboard_service):
    dash = await dashboard_service.get_field_dashboard("field_1")
    assert isinstance(dash, FarmerDashboardResponse)
    assert dash.status == "success"
    assert dash.field_overview.field_id == "field_1"
    assert dash.field_overview.crop_type == "Paddy / Rice"
    assert "West Godavari" in dash.field_overview.location_name
    assert dash.field_overview.total_area_hectares > 0.0


# -----------------------------------------------------------------------------
# 3. Missing Field Handling (404)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_missing_field_handling(dashboard_service):
    with pytest.raises(FieldNotFoundError):
        await dashboard_service.get_field_dashboard("non_existent_field_999")

    # Also test via HTTP API
    resp = client.get("/api/v1/farmer-dashboard/field/non_existent_field_999")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# -----------------------------------------------------------------------------
# 4. Historical Observation Integration
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_historical_observation_integration(dashboard_service):
    dash = await dashboard_service.get_field_dashboard("field_1")
    hist = dash.historical_monitoring
    assert hist.observation_count >= 2
    assert len(hist.timeline) >= 2
    assert hist.median_cloud_coverage >= 0.0

    first_obs = hist.timeline[0]
    assert first_obs.date != ""
    assert first_obs.cloud_coverage >= 0.0
    assert first_obs.valid_pixel_pct > 0.0
    assert first_obs.ndvi is not None


# -----------------------------------------------------------------------------
# 5. Vegetation Metric Integration & Deltas
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_vegetation_metric_integration_and_deltas(dashboard_service):
    dash = await dashboard_service.get_field_dashboard("field_1")
    v_metrics = dash.vegetation_metrics

    for idx_key in ["NDVI", "NDMI", "NDRE", "EVI"]:
        assert idx_key in v_metrics
        metric = v_metrics[idx_key]
        assert metric.name == idx_key
        assert metric.latest_value is not None
        assert metric.quality_status in ["Good", "Moderate", "Limited"]
        assert metric.trend_direction in ["Increasing", "Stable", "Decreasing", "Insufficient Data"]
        # If previous value exists, delta should match difference
        if metric.previous_value is not None and metric.delta_absolute is not None:
            diff = round(metric.latest_value - metric.previous_value, 4)
            assert abs(metric.delta_absolute - diff) < 0.001


# -----------------------------------------------------------------------------
# 6. Stress Model Availability & Ground Truth Advisory
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stress_model_availability(dashboard_service, monkeypatch):
    dash = await dashboard_service.get_field_dashboard("field_1")
    stress = dash.stress_intelligence
    assert stress.is_model_available is True
    assert stress.ground_truth_advisory != ""
    assert "Satellite anomalies identify physical canopy changes" in stress.ground_truth_advisory or "Supervised" in stress.ground_truth_advisory    # Test production mode with unavailable model
    from services.copernicus_auth import copernicus_auth
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setattr(copernicus_auth, "is_configured", lambda: False)
    monkeypatch.setattr(dashboard_service.stress_service, "load_model", lambda prefer_production=True: (None, None))
    # Clear cache
    dashboard_service._dashboard_cache.clear()

    dash_prod = await dashboard_service.get_field_dashboard("field_1", prefer_production=True, force_refresh=True)
    stress_prod = dash_prod.stress_intelligence
    assert stress_prod.is_model_available is False
    assert "STRESS MODEL UNAVAILABLE — VERIFIED AGRONOMIC GROUND TRUTH REQUIRED" in stress_prod.status_message
    assert stress_prod.stress_class is None
    assert stress_prod.stress_probability is None


# -----------------------------------------------------------------------------
# 7. Growth Forecast Integration
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_growth_forecast_integration(dashboard_service):
    dash = await dashboard_service.get_field_dashboard("field_1")
    forecast = dash.growth_forecast
    assert forecast.target_index == "NDVI"
    assert len(forecast.forecast_dates) >= 1
    assert len(forecast.projected_values) == len(forecast.forecast_dates)
    assert len(forecast.prediction_intervals) > 0

    pi = forecast.prediction_intervals[0]
    assert pi.lower <= pi.upper
    assert "yield promises" in forecast.disclaimer.lower() or "not constitute guaranteed agricultural outcomes" in forecast.disclaimer.lower() or "harvest yield" in forecast.disclaimer.lower()


# -----------------------------------------------------------------------------
# 8. Field Risk Statistics Integration
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_field_risk_statistics_integration(dashboard_service):
    dash = await dashboard_service.get_field_dashboard("field_1")
    risk = dash.field_risk
    assert risk.total_area_hectares > 0.0
    assert risk.valid_canopy_percentage > 0.0
    assert len(risk.risk_distribution) >= 3

    for r_key in ["low_risk", "moderate_risk", "elevated_risk"]:
        if r_key in risk.risk_distribution:
            cat = risk.risk_distribution[r_key]
            assert cat.area_hectares >= 0.0
            assert 0.0 <= cat.percentage_of_field <= 100.0


# -----------------------------------------------------------------------------
# 9. Recommendation Evidence & Non-Prescriptive Rules
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_recommendations_evidence_and_non_prescriptive(dashboard_service):
    dash = await dashboard_service.get_field_dashboard("field_1")
    recs = dash.recommendations
    assert len(recs) > 0

    forbidden_words = ["spray", "pesticide", "fertilizer", "kg/ha", "apply chemical", "irrigate 50mm"]
    for rec in recs:
        assert rec.evidence_source != ""
        assert rec.evidence_date != ""
        assert rec.reason != ""
        assert isinstance(rec.field_verification_required, bool)
        assert rec.priority in ["high", "medium", "low", "info"]

        # Ensure no prescriptive chemical dosages
        action_lower = rec.action.lower()
        for forbidden in forbidden_words:
            assert forbidden not in action_lower, f"Forbidden prescriptive term '{forbidden}' found in recommendation action: {rec.action}"


# -----------------------------------------------------------------------------
# 10. Insufficient Evidence Handling
# -----------------------------------------------------------------------------
def test_insufficient_evidence_recommendation(recommendation_service):
    recs = recommendation_service.generate_recommendations(
        field_id="field_cloudy",
        observation_date="2026-11-01",
        data_reliability={
            "valid_pixel_percentage": 25.0,
            "cloud_coverage_percentage": 75.0,
        },
        historical_monitoring={
            "observation_count": 1,
        },
    )
    assert len(recs) > 0
    insufficient_rec = [r for r in recs if "Insufficient Observational Evidence" in r.title]
    assert len(insufficient_rec) == 1
    assert insufficient_rec[0].priority == "high"
    assert insufficient_rec[0].field_verification_required is True


# -----------------------------------------------------------------------------
# 11. Model Provenance
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_model_provenance(dashboard_service):
    dash = await dashboard_service.get_field_dashboard("field_1")
    assert dash.data_reliability.optical_sensor_status != ""
    assert dash.data_reliability.sar_sensor_status != ""
    assert dash.stress_intelligence.model_version != ""
    assert dash.growth_forecast.baseline_algorithm != ""


# -----------------------------------------------------------------------------
# 12. Demo vs Production Separation
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_demo_production_separation(dashboard_service, monkeypatch):
    # In default demo mode
    dash_demo = await dashboard_service.get_field_dashboard("field_1")
    assert dash_demo.field_overview.is_demo is True
    assert dash_demo.data_reliability.is_demo is True

    # When demo mode is disabled and model is missing
    from services.copernicus_auth import copernicus_auth
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setattr(copernicus_auth, "is_configured", lambda: False)
    monkeypatch.setattr(dashboard_service.stress_service, "load_model", lambda prefer_production=True: (None, None))
    dashboard_service._dashboard_cache.clear()

    dash_prod = await dashboard_service.get_field_dashboard("field_1", prefer_production=True, force_refresh=True)
    assert dash_prod.stress_intelligence.is_demo is False
    assert dash_prod.stress_intelligence.is_model_available is False


# -----------------------------------------------------------------------------
# 13. Field Comparison Validation & Temporal Warnings
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_field_comparison_validation(dashboard_service):
    # Need at least 2 fields
    with pytest.raises(ValueError):
        await dashboard_service.compare_fields(["field_1"])

    comp = await dashboard_service.compare_fields(["field_1", "field_2"])
    assert isinstance(comp, FieldComparisonResponse)
    assert comp.status == "success"
    assert len(comp.compared_fields) == 2
    assert comp.compared_fields[0].field_id == "field_1"
    assert comp.compared_fields[1].field_id == "field_2"
    assert comp.compared_fields[0].ndvi_mean is not None


@pytest.mark.asyncio
async def test_field_comparison_temporal_mismatch(dashboard_service):
    # Manipulate dates of field_2 to simulate temporal discrepancy > 10 days
    dashboard_service._fields["field_2"]["start_date"] = "2026-05-01"
    dashboard_service._fields["field_2"]["end_date"] = "2026-08-01"
    dashboard_service._dashboard_cache.clear()

    comp = await dashboard_service.compare_fields(["field_1", "field_2"])
    assert comp.temporal_alignment_compatible is False
    assert comp.temporal_discrepancy_days > 10
    assert len(comp.comparison_warnings) > 0
    assert "Temporal discrepancy between fields is" in comp.comparison_warnings[0]



# -----------------------------------------------------------------------------
# 14. API Endpoints Validation
# -----------------------------------------------------------------------------
def test_api_endpoints():
    # 1. Overview
    r1 = client.get("/api/v1/farmer-dashboard/overview")
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["status"] == "success"
    assert len(d1["fields"]) >= 3

    # 2. Single Field Dashboard
    r2 = client.get("/api/v1/farmer-dashboard/field/field_1")
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["field_overview"]["field_id"] == "field_1"
    assert "NDVI" in d2["vegetation_metrics"]
    assert "crop_condition" in d2

    # 3. Recommendations
    r3 = client.get("/api/v1/farmer-dashboard/field/field_1/recommendations")
    assert r3.status_code == 200
    d3 = r3.json()
    assert isinstance(d3, list)
    assert len(d3) > 0

    # 4. Compare fields (Valid)
    r4 = client.get("/api/v1/farmer-dashboard/compare?fields=field_1,field_2")
    assert r4.status_code == 200
    d4 = r4.json()
    assert len(d4["compared_fields"]) == 2

    # 5. Compare fields (Invalid < 2)
    r5 = client.get("/api/v1/farmer-dashboard/compare?fields=field_1")
    assert r5.status_code == 400

    # 6. Register new field
    reg_payload = {
        "field_name": "Test Cooperative Cotton Plot",
        "crop_type": "Cotton",
        "latitude": 16.35,
        "longitude": 80.52,
        "radius_km": 1.2,
    }
    r6 = client.post("/api/v1/farmer-dashboard/field", json=reg_payload)
    assert r6.status_code == 200
    d6 = r6.json()
    assert d6["field_name"] == "Test Cooperative Cotton Plot"
    assert d6["total_area_hectares"] > 0.0


# -----------------------------------------------------------------------------
# 15. No Fabricated Metrics or Diagnostic Claims
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_fabricated_metrics_or_diagnostic_claims(dashboard_service):
    dash = await dashboard_service.get_field_dashboard("field_1")
    # Disclaimers must explicitly declare non-causation
    all_disclaimers = " ".join(dash.scientific_disclaimers).lower()
    assert "do not prove causal presence" in all_disclaimers or "physical surface reflectance anomalies" in all_disclaimers
    assert "yield promises" in all_disclaimers or "guaranteed agricultural outcomes" in all_disclaimers or "harvest yield" in all_disclaimers


# -----------------------------------------------------------------------------
# 16. Regression Compatibility with Phases 1–16
# -----------------------------------------------------------------------------
def test_regression_endpoints_exist():
    # Verify core prior phase endpoints are intact
    # Phase 8: History
    r_hist = client.post("/api/v1/analysis/history", json={})
    assert r_hist.status_code in [400, 422]  # Endpoint exists and validates payload


    # Phase 14: Growth Prediction
    r_gp = client.post("/api/v1/analysis/growth-prediction", json={})
    assert r_gp.status_code in [400, 422]  # Endpoint exists and validates payload

    # Phase 16: Field Risk Map
    r_frm = client.post("/api/v1/analysis/field-risk-map", json={})
    assert r_frm.status_code in [400, 422]  # Endpoint exists and validates payload

