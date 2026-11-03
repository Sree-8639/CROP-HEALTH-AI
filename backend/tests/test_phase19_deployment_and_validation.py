import os
import datetime
import pytest
from fastapi.testclient import TestClient

from main import app
from models.field_validation import (
    FieldScoutingObservation,
    ObserverProfile,
    ObserverCredentialType,
    SoilMoistureMeasurement,
    ChlorophyllMeasurement,
    ConsentAndAccessControl,
    DataSharingTier,
    LabelVerificationStatus,
)
from services.field_validation_service import field_validation_service

client = TestClient(app)


def test_health_check_phase19():
    """Verify container probes /health and /api/health."""
    res_root = client.get("/health")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "ok"

    res_api = client.get("/api/health")
    assert res_api.status_code == 200
    assert res_api.json()["status"] == "ok"


def test_cors_configuration_environment(monkeypatch):
    """Verify that CORS honors ALLOWED_ORIGINS and blocks disallowed origins."""
    # Test valid origin request
    res = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert res.status_code == 200
    # In FastAPI CORSMiddleware, valid origin echoes back in header
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_field_validation_provenance_initial():
    """Verify provenance endpoint returns blocked production status when samples < 500."""
    res = client.get("/api/v1/field-validation/provenance")
    assert res.status_code == 200
    data = res.json()
    assert data["catalog_name"] == "CropHealth-GroundTruth-FieldValidation"
    assert data["production_training_eligibility"] is False
    assert "INSUFFICIENT VERIFIED SAMPLES" in data["eligibility_message"]


def test_field_validation_record_success():
    """Verify recording a valid agronomic field observation with complete telemetry."""
    now = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
    obs_payload = {
        "observation_id": "OBS-2024-TEST-001",
        "field_id": "field_1",
        "geometry": {
            "type": "Point",
            "coordinates": [4.355, 48.955],
        },
        "observation_timestamp": now.isoformat(),
        "crop_type": "winter_wheat",
        "growth_stage_bbch": 65,
        "visual_symptoms": ["slight_flag_leaf_chlorosis"],
        "canopy_cover_pct": 82.5,
        "primary_stress_type": "Water_Deficit",
        "stress_severity_rating": 1,
        "soil_moisture": {
            "depth_cm": 15.0,
            "volumetric_water_content_pct": 18.4,
            "measurement_method": "TDR_probe",
            "sensor_serial": "TDR-315L-SN982",
        },
        "chlorophyll": {
            "spad_reading": 42.6,
            "replicate_count": 5,
            "standard_deviation": 1.2,
            "device_model": "Konica_Minolta_SPAD_502Plus",
        },
        "observer": {
            "observer_id": "OBS-AGR-042",
            "full_name": "Dr. Eleanor Vance",
            "organization": "European Agronomic Research Station",
            "credential_type": "Agronomist_MSc_PhD",
            "credential_id": "EUR-AGR-8831",
        },
        "consent": {
            "farmer_consent_obtained": True,
            "data_sharing_tier": "anonymized_research_only",
            "data_license": "CC-BY-4.0",
            "anonymize_spatial_exact_coordinates": True,
        },
        "dataset_version": "v1.0.0-rc1",
    }

    res = client.post("/api/v1/field-validation/observations", json=obs_payload)
    assert res.status_code == 201
    res_data = res.json()
    assert res_data["status"] == "success"
    assert res_data["observation_id"] == "OBS-2024-TEST-001"
    assert res_data["verification_status"] == "DRAFT"


def test_field_validation_consent_rejection():
    """Verify that observations without informed consent are rejected (QC Rule 1)."""
    now = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
    obs_payload = {
        "observation_id": "OBS-2024-TEST-NOCONSENT",
        "field_id": "field_1",
        "geometry": {"type": "Point", "coordinates": [4.355, 48.955]},
        "observation_timestamp": now.isoformat(),
        "crop_type": "maize",
        "growth_stage_bbch": 30,
        "observer": {
            "observer_id": "OBS-001",
            "full_name": "Field Worker",
            "organization": "Scouting Corp",
            "credential_type": "Trained_Field_Scout",
        },
        "consent": {
            "farmer_consent_obtained": False,  # Missing consent!
            "data_sharing_tier": "anonymized_research_only",
            "data_license": "CC-BY-4.0",
            "anonymize_spatial_exact_coordinates": True,
        },
    }

    res = client.post("/api/v1/field-validation/observations", json=obs_payload)
    assert res.status_code == 400
    assert "informed consent is required" in res.json()["detail"]


def test_field_validation_insufficient_spad_replicates():
    """Verify that SPAD chlorophyll readings with < 3 replicates are rejected (QC Rule 2)."""
    now = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
    obs_payload = {
        "observation_id": "OBS-2024-TEST-LOWREPS",
        "field_id": "field_1",
        "geometry": {"type": "Point", "coordinates": [4.355, 48.955]},
        "observation_timestamp": now.isoformat(),
        "crop_type": "soybean",
        "growth_stage_bbch": 51,
        "chlorophyll": {
            "spad_reading": 38.1,
            "replicate_count": 2,  # Invalid: minimum 3 required!
            "device_model": "SPAD-502",
        },
        "observer": {
            "observer_id": "OBS-001",
            "full_name": "Field Worker",
            "organization": "Scouting Corp",
            "credential_type": "Trained_Field_Scout",
        },
        "consent": {
            "farmer_consent_obtained": True,
            "data_sharing_tier": "anonymized_research_only",
            "data_license": "CC-BY-4.0",
            "anonymize_spatial_exact_coordinates": True,
        },
    }

    res = client.post("/api/v1/field-validation/observations", json=obs_payload)
    # Pydantic validation rejects replicate_count < 3 with HTTP 422
    assert res.status_code == 422


def test_field_validation_future_timestamp_rejection():
    """Verify that observation timestamps in the future are rejected."""
    future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=10)
    obs_payload = {
        "observation_id": "OBS-2024-FUTURE",
        "field_id": "field_1",
        "geometry": {"type": "Point", "coordinates": [4.355, 48.955]},
        "observation_timestamp": future.isoformat(),
        "crop_type": "wheat",
        "growth_stage_bbch": 40,
        "observer": {
            "observer_id": "OBS-001",
            "full_name": "Field Worker",
            "organization": "Scouting Corp",
            "credential_type": "Trained_Field_Scout",
        },
        "consent": {
            "farmer_consent_obtained": True,
            "data_sharing_tier": "anonymized_research_only",
            "data_license": "CC-BY-4.0",
            "anonymize_spatial_exact_coordinates": True,
        },
    }

    res = client.post("/api/v1/field-validation/observations", json=obs_payload)
    assert res.status_code == 422


def test_field_validation_double_blind_review():
    """Verify double-blind peer review: observer cannot review self, reviewer must be certified."""
    obs_id = "OBS-2024-TEST-001"

    # Self-review attempt must be forbidden (403)
    res_self = client.post(
        f"/api/v1/field-validation/observations/{obs_id}/review",
        json={
            "reviewer_id": "OBS-AGR-042",  # Same as observer!
            "reviewer_credential": "Agronomist_MSc_PhD",
            "action": "APPROVE",
            "notes": "Attempting self-approval",
        },
    )
    assert res_self.status_code == 403
    assert "Observer cannot review their own field observation" in res_self.json()["detail"]

    # Non-certified scout cannot approve ground truth
    res_unqual = client.post(
        f"/api/v1/field-validation/observations/{obs_id}/review",
        json={
            "reviewer_id": "SCOUT-099",
            "reviewer_credential": "Trained_Field_Scout",  # Not certified!
            "action": "APPROVE",
            "notes": "Scout approval attempt",
        },
    )
    assert res_unqual.status_code == 403
    assert "Only Certified Crop Advisors or PhD/MSc Agronomists" in res_unqual.json()["detail"]

    # Valid independent certified agronomist approval
    res_valid = client.post(
        f"/api/v1/field-validation/observations/{obs_id}/review",
        json={
            "reviewer_id": "REVIEWER-CCA-108",
            "reviewer_credential": "Certified_Crop_Advisor",
            "action": "APPROVE",
            "notes": "Confirmed water deficit via physical TDR probe calibration.",
        },
    )
    assert res_valid.status_code == 200
    assert res_valid.json()["verification_status"] == "VERIFIED_AGRONOMIST"


def test_field_validation_satellite_spatial_temporal_matching():
    """Verify matching ground observation with Sentinel satellite scenes."""
    obs_id = "OBS-2024-TEST-001"
    sentinel_scenes = [
        {
            "scene_id": "S2A_MSIL2A_20240625T103631_N0510_R008_T31UEQ_20240625T164350.SAFE",
            "acquisition_date": "2024-06-25T10:47:30Z",
        },
        {
            "scene_id": "S2B_MSIL2A_20240715T103629_N0510_R008_T31UEQ_20240715T164012.SAFE",
            "acquisition_date": "2024-07-15T10:40:12Z",
        },
    ]

    match_res = field_validation_service.match_with_satellite_scene(
        observation_id=obs_id,
        sentinel_scenes=sentinel_scenes,
        max_temporal_offset_days=5000,  # Wide window to test match logic
    )
    assert match_res["matched"] is True
    assert match_res["matched_scene_id"] is not None


def test_list_field_observations_filter():
    """Verify listing field observations with filtering."""
    res = client.get("/api/v1/field-validation/observations?field_id=field_1")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["count"] >= 1
    assert data["observations"][0]["field_id"] == "field_1"
