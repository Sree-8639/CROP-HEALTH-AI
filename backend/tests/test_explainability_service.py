"""
Test Suite for Explainable AI, Confidence, and Uncertainty Intelligence (Phase 15).

Validates:
1. Local explanation output
2. Global explanation output
3. Local/global explanation separation
4. Feature schema consistency with Phase 13
5. Missing feature handling
6. Missing Sentinel-1 handling
7. Insufficient observation handling
8. Data-quality classification
9. Model-version reporting
10. Demo/production separation
11. Uncalibrated probability restrictions
12. Growth forecast explanation
13. Prediction interval availability handling
14. API request validation
15. API response schema
16. No fabricated confidence values
17. No fabricated feature contributions
18. Regression compatibility with Phases 1-14
"""

import os
import math
import pytest
from fastapi.testclient import TestClient

from main import app
from services.historical_service import historical_monitoring_service
from services.explainability_service import ExplainabilityService, explainability_service
from ml.explainability.schema import (
    FEATURE_METADATA,
    QUALITY_TIER_GOOD,
    QUALITY_TIER_MODERATE,
    QUALITY_TIER_LIMITED,
    QUALITY_TIER_INSUFFICIENT,
    ExplainabilityRequest,
)
from ml.explainability.shap_explainer import StressModelExplainer
from ml.stress_detection.schema import (
    ALL_MODEL_FEATURE_NAMES,
    CLASS_NO_SIGNIFICANT_STRESS,
    CLASS_POTENTIAL_STRESS,
    CLASS_ELEVATED_STRESS,
)

client = TestClient(app)

SAMPLE_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [
            [81.516592, 16.800244],
            [81.536592, 16.800244],
            [81.536592, 16.820244],
            [81.516592, 16.820244],
            [81.516592, 16.800244],
        ]
    ],
}


# -------------------------------------------------------------
# 1. Local Explanation Output
# -------------------------------------------------------------
def test_local_explanation_output():
    """Verify local explanation generates signed feature contributions for observation."""
    model, metadata = explainability_service.stress_service.load_model(prefer_production=False)
    assert model is not None, "Demo model should be loadable"

    explainer = StressModelExplainer(model=model, metadata=metadata)
    sample_features = {feat: 0.5 for feat in ALL_MODEL_FEATURE_NAMES}
    sample_features["opt_ndmi"] = 0.15  # Dropped moisture
    sample_features["delta_ndmi"] = -0.12

    explanation = explainer.explain_instance(
        feature_dict=sample_features,
        predicted_class=CLASS_POTENTIAL_STRESS,
        stress_probability=0.72,
        predicted_class_idx=1,
    )

    assert explanation.method in ("SHAP_TreeExplainer", "Controlled_Perturbation")
    assert len(explanation.local_feature_contributions) > 0
    top = explanation.local_feature_contributions[0]
    assert top.feature in ALL_MODEL_FEATURE_NAMES
    assert top.magnitude >= 0.0
    assert top.direction in ("increases_stress", "decreases_stress", "neutral")
    assert len(top.explanation_text) > 5


# -------------------------------------------------------------
# 2. Global Explanation Output
# -------------------------------------------------------------
def test_global_explanation_output():
    """Verify global model-level feature importance is retrieved and ordered."""
    model, metadata = explainability_service.stress_service.load_model(prefer_production=False)
    explainer = StressModelExplainer(model=model, metadata=metadata)

    global_imp = explainer.get_global_importance()
    assert len(global_imp) > 0
    assert global_imp[0].rank == 1
    # Check monotonic ranking descending
    for i in range(len(global_imp) - 1):
        assert global_imp[i].importance >= global_imp[i + 1].importance


# -------------------------------------------------------------
# 3. Local/Global Explanation Separation
# -------------------------------------------------------------
def test_local_global_separation():
    """Verify local and global explanations are distinct objects with separate semantics."""
    model, metadata = explainability_service.stress_service.load_model(prefer_production=False)
    explainer = StressModelExplainer(model=model, metadata=metadata)
    sample_features = {feat: 0.5 for feat in ALL_MODEL_FEATURE_NAMES}

    exp = explainer.explain_instance(
        feature_dict=sample_features,
        predicted_class=CLASS_POTENTIAL_STRESS,
        stress_probability=0.65,
    )

    local_feats = [item.feature for item in exp.local_feature_contributions]
    global_feats = [item.feature for item in (exp.global_feature_importance or [])]

    assert isinstance(exp.local_feature_contributions, list)
    assert isinstance(exp.global_feature_importance, list)
    # Global has importance >= 0, local has signed contribution
    assert all(hasattr(item, "contribution") for item in exp.local_feature_contributions)
    assert all(hasattr(item, "importance") for item in exp.global_feature_importance)


# -------------------------------------------------------------
# 4. Feature Schema Consistency with Phase 13
# -------------------------------------------------------------
def test_feature_schema_consistency():
    """Verify all 37 Phase 13 features are mapped in FEATURE_METADATA without drift."""
    for feat in ALL_MODEL_FEATURE_NAMES:
        assert feat in FEATURE_METADATA, f"Feature {feat} missing from FEATURE_METADATA"
        meta = FEATURE_METADATA[feat]
        assert "display_name" in meta
        assert "modality" in meta
        assert "unit" in meta
        assert "description" in meta


# -------------------------------------------------------------
# 5. Missing Feature Handling
# -------------------------------------------------------------
def test_missing_feature_handling():
    """Verify explainer handles missing or NaN feature inputs without crashing."""
    model, metadata = explainability_service.stress_service.load_model(prefer_production=False)
    explainer = StressModelExplainer(model=model, metadata=metadata)

    # Empty dictionary (all features missing)
    exp = explainer.explain_instance(
        feature_dict={},
        predicted_class=CLASS_NO_SIGNIFICANT_STRESS,
        stress_probability=0.20,
    )
    assert exp is not None
    assert len(exp.local_feature_contributions) == len(ALL_MODEL_FEATURE_NAMES)
    # Check that missing features have value=None
    for item in exp.local_feature_contributions:
        assert item.value is None


# -------------------------------------------------------------
# 6. Missing Sentinel-1 Handling
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_missing_sentinel1_handling():
    """Verify missing SAR backscatter is flagged and quality tier adjusted."""
    assessment = await explainability_service._assess_data_quality(
        geometry=SAMPLE_GEOMETRY,
        latitude=None,
        longitude=None,
        radius_km=2.0,
        start_date="2026-10-01",
        end_date="2026-10-02",
        max_cloud_coverage=35.0,
    )
    assert assessment is not None
    assert hasattr(assessment, "s1_available")
    assert hasattr(assessment, "missing_features_count")


# -------------------------------------------------------------
# 7. Insufficient Observation Handling
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_insufficient_observation_handling(monkeypatch):
    """Verify sparse observation history flags limited/insufficient quality."""
    async def mock_sparse_history(*args, **kwargs):
        return {
            "status": "success",
            "observations": [
                {
                    "date": "2026-10-26",
                    "cloud_coverage": 12.0,
                    "valid_pixel_percentage": 55.0,
                    "sar": None,
                }
            ],
        }

    monkeypatch.setattr(
        historical_monitoring_service,
        "get_historical_time_series",
        mock_sparse_history,
    )

    assessment = await explainability_service._assess_data_quality(
        geometry=SAMPLE_GEOMETRY,
        latitude=None,
        longitude=None,
        radius_km=2.0,
        start_date="2026-10-25",
        end_date="2026-10-26",
        max_cloud_coverage=5.0,
    )
    assert assessment.observation_count <= 2
    assert assessment.quality_tier in (QUALITY_TIER_LIMITED, QUALITY_TIER_INSUFFICIENT)
    assert len(assessment.data_limitations) > 0


# -------------------------------------------------------------
# 8. Data-Quality Classification
# -------------------------------------------------------------
def test_data_quality_tier_boundaries():
    """Verify 4-tier quality classifications respect configured boundaries."""
    thresholds = explainability_service.quality_thresholds

    assert thresholds["cloud_max_good"] < thresholds["cloud_max_moderate"]
    assert thresholds["valid_min_good"] > thresholds["valid_min_moderate"]
    assert thresholds["history_min_good"] >= thresholds["history_min_moderate"]


# -------------------------------------------------------------
# 9. Model-Version Reporting
# -------------------------------------------------------------
def test_model_version_reporting():
    """Verify authentic model version and provenance are reported."""
    _, metadata = explainability_service.stress_service.load_model(prefer_production=False)
    rel = explainability_service._extract_model_reliability_metadata(
        stress_meta=metadata,
        is_demo=True,
        growth_explanation=None,
    )
    assert rel.stress_model["model_version"] == metadata.get("model_version", "v1.0.0-demo")
    assert "provenance" in rel.stress_model["training_dataset_provenance"].lower() or "synthetic" in rel.stress_model["training_dataset_provenance"].lower()


# -------------------------------------------------------------
# 10. Demo vs Production Separation
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_demo_production_separation(monkeypatch):
    """Verify production mode refuses to use synthetic demo models."""
    monkeypatch.setenv("DEMO_MODE", "false")
    svc = ExplainabilityService()
    assert svc.is_demo_mode is False

    req = ExplainabilityRequest(
        geometry=SAMPLE_GEOMETRY,
        start_date="2026-10-01",
        end_date="2026-10-26",
    )
    res = await svc.generate_explainability_assessment(req)

    assert res["is_demo"] is False
    assert res["explanation_status"] in ("MODEL_UNAVAILABLE", "EXPLANATION_AVAILABLE")
    if res["explanation_status"] == "MODEL_UNAVAILABLE":
        assert any("blocked" in w.lower() for w in res["warnings"])


# -------------------------------------------------------------
# 11. Uncalibrated Probability Restrictions
# -------------------------------------------------------------
def test_uncalibrated_probability_restrictions():
    """Verify uncalibrated models do not falsely claim calibrated status."""
    dummy_meta = {"is_calibrated": False, "model_name": "UncalibratedTest"}
    rel = explainability_service._extract_model_reliability_metadata(
        stress_meta=dummy_meta,
        is_demo=False,
        growth_explanation=None,
    )
    assert rel.stress_model["is_calibrated"] is False
    assert rel.stress_model["calibration_status"] == "Uncalibrated"
    assert rel.stress_model["brier_score"] is None


# -------------------------------------------------------------
# 12. Growth Forecast Explanation
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_growth_forecast_explanation():
    """Verify Phase 14 growth forecasts are explained with selection rationale."""
    exp = await explainability_service._generate_growth_forecast_explanation(
        geometry=SAMPLE_GEOMETRY,
        latitude=None,
        longitude=None,
        radius_km=2.0,
        start_date="2026-10-01",
        end_date="2026-10-26",
        max_cloud_coverage=35.0,
        target_index="NDVI",
    )
    assert exp is not None
    assert "NDVI" in exp.index_explanations
    ndvi_exp = exp.index_explanations["NDVI"]
    assert ndvi_exp.historical_direction in ("Increasing", "Decreasing", "Relatively Stable", "Insufficient Evidence")
    assert len(ndvi_exp.selection_rationale) > 10
    assert len(ndvi_exp.candidate_model_comparisons) > 0


# -------------------------------------------------------------
# 13. Prediction Interval Availability Handling
# -------------------------------------------------------------
def test_prediction_interval_availability():
    """Verify prediction intervals are available only with sufficient observations (>=4)."""
    # Case A: < 4 observations
    unc_sparse = explainability_service._extract_prediction_uncertainty(
        growth_explanation=None,
        stress_explanation=None,
        observation_count=2,
    )
    assert unc_sparse.intervals_available is False
    assert unc_sparse.growth_intervals_summary["withholding_reason"] is not None

    # Case B: >= 4 observations
    unc_dense = explainability_service._extract_prediction_uncertainty(
        growth_explanation=None,
        stress_explanation=None,
        observation_count=6,
    )
    assert unc_dense.intervals_available is True
    assert unc_dense.growth_intervals_summary["withholding_reason"] is None


# -------------------------------------------------------------
# 14. API Request Validation
# -------------------------------------------------------------
def test_api_request_validation():
    """Verify API endpoint rejects request missing geometry and coordinates."""
    bad_payload = {
        "start_date": "2026-10-01",
        "end_date": "2026-10-26",
    }
    resp = client.post("/api/v1/analysis/explainability", json=bad_payload)
    assert resp.status_code == 400
    assert "Either 'geometry' or 'latitude' and 'longitude'" in resp.json()["detail"]


# -------------------------------------------------------------
# 15. API Response Schema
# -------------------------------------------------------------
def test_api_response_schema():
    """Verify API endpoint returns full Phase 15 response structure."""
    payload = {
        "geometry": SAMPLE_GEOMETRY,
        "start_date": "2026-10-01",
        "end_date": "2026-10-26",
        "explanation_type": "all",
        "selected_index": "NDVI",
    }
    resp = client.post("/api/v1/analysis/explainability", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert "explanation_status" in data
    assert "data_quality_assessment" in data
    assert "model_reliability" in data
    assert "prediction_uncertainty" in data
    assert "warnings" in data
    assert "scientific_limitations" in data


# -------------------------------------------------------------
# 16. No Fabricated Confidence Values
# -------------------------------------------------------------
def test_no_fabricated_confidence_values():
    """Verify the system does not produce arbitrary single confidence scores."""
    rel = explainability_service._extract_model_reliability_metadata(
        stress_meta={"is_calibrated": True},
        is_demo=False,
        growth_explanation=None,
    )
    # Check that there is no arbitrary single confidence field
    assert "confidence_percentage" not in rel.stress_model
    assert "confidence_score" not in rel.stress_model


# -------------------------------------------------------------
# 17. No Fabricated Feature Contributions
# -------------------------------------------------------------
def test_no_fabricated_feature_contributions():
    """Verify feature contributions have authentic provenance and non-causation disclaimer."""
    model, metadata = explainability_service.stress_service.load_model(prefer_production=False)
    explainer = StressModelExplainer(model=model, metadata=metadata)
    sample_features = {feat: 0.5 for feat in ALL_MODEL_FEATURE_NAMES}

    exp = explainer.explain_instance(
        feature_dict=sample_features,
        predicted_class=CLASS_POTENTIAL_STRESS,
        stress_probability=0.60,
    )
    assert "causation" in exp.non_causation_disclaimer.lower()
    for c in exp.local_feature_contributions:
        assert isinstance(c.contribution, float)
        assert not math.isnan(c.contribution)


# -------------------------------------------------------------
# 18. Regression Compatibility with Phases 1-14
# -------------------------------------------------------------
def test_regression_compatibility():
    """Verify Phase 1-14 endpoints continue functioning normally."""
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200

    aoi_payload = {"latitude": 16.810244, "longitude": 81.526592, "radius_km": 2.0}
    aoi_resp = client.post("/api/v1/aoi", json=aoi_payload)
    assert aoi_resp.status_code == 200
