from __future__ import annotations

import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator


class LabelVerificationStatus(str, Enum):
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    PENDING_SECOND_REVIEW = "PENDING_SECOND_REVIEW"
    VERIFIED_AGRONOMIST = "VERIFIED_AGRONOMIST"
    REJECTED = "REJECTED"


class ObserverCredentialType(str, Enum):
    CERTIFIED_CROP_ADVISOR = "Certified_Crop_Advisor"
    AGRONOMIST_MSC_PHD = "Agronomist_MSc_PhD"
    UNIVERSITY_RESEARCHER = "University_Researcher"
    EXTENSION_AGENT = "Extension_Agent"
    TRAINED_FIELD_SCOUT = "Trained_Field_Scout"


class DataSharingTier(str, Enum):
    ANONYMIZED_RESEARCH_ONLY = "anonymized_research_only"
    PLATFORM_BENCHMARKING = "platform_benchmarking"
    FULL_OPEN_ACCESS = "full_open_access"


class ObserverProfile(BaseModel):
    observer_id: str = Field(..., description="Unique anonymized observer ID")
    full_name: str = Field(..., description="Full legal name of the observer")
    organization: str = Field(..., description="Institutional affiliation, agronomy firm, or university")
    credential_type: ObserverCredentialType
    credential_id: Optional[str] = Field(None, description="Certification/license ID (e.g., CCA #12345)")


class SoilMoistureMeasurement(BaseModel):
    depth_cm: float = Field(..., ge=0.0, le=200.0, description="Measurement depth in centimeters")
    volumetric_water_content_pct: float = Field(..., ge=0.0, le=100.0, description="Volumetric soil water content (0-100%)")
    measurement_method: str = Field("TDR_probe", description="Method: TDR_probe, gravimetric_oven, or capacitance")
    sensor_serial: Optional[str] = None


class ChlorophyllMeasurement(BaseModel):
    spad_reading: float = Field(..., ge=0.0, le=99.9, description="Relative chlorophyll index (SPAD-502 reading)")
    replicate_count: int = Field(..., ge=3, description="Minimum 3 distinct sample replicates required")
    standard_deviation: Optional[float] = Field(None, ge=0.0)
    device_model: str = Field("Konica_Minolta_SPAD_502Plus")


class LaboratoryAnalysis(BaseModel):
    lab_name: str
    report_id: str
    sample_date: datetime.date
    leaf_nitrogen_pct: Optional[float] = Field(None, ge=0.0, le=10.0, description="Dry-matter leaf nitrogen percentage")
    pathogen_identified: Optional[str] = None
    pathogen_load: Optional[str] = None
    analysis_protocol: str = Field(..., description="Analytical method (e.g., Kjeldahl, PCR, ELISA)")


class ConsentAndAccessControl(BaseModel):
    farmer_consent_obtained: bool = Field(..., description="Explicit informed consent from landowner/operator")
    data_sharing_tier: DataSharingTier = DataSharingTier.ANONYMIZED_RESEARCH_ONLY
    data_license: str = Field("CC-BY-4.0", description="Open access or research license")
    anonymize_spatial_exact_coordinates: bool = True


class FieldScoutingObservation(BaseModel):
    observation_id: str = Field(..., description="Unique UUID for observation record")
    field_id: str = Field(..., description="Canonical field identifier")
    geometry: Dict[str, Any] = Field(..., description="GeoJSON Point or Polygon of sampling location")
    observation_timestamp: datetime.datetime = Field(..., description="ISO 8601 UTC timestamp of in-situ sampling")
    crop_type: str = Field(..., description="Standardized crop commodity (e.g., wheat, maize, soybean)")
    growth_stage_bbch: int = Field(..., ge=0, le=99, description="Standardized BBCH phenological growth stage (0-99)")
    visual_symptoms: List[str] = Field(default_factory=list, description="Observed symptom flags (e.g., chlorosis, stunting)")
    canopy_cover_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    primary_stress_type: Optional[str] = Field(None, description="Water_Deficit, Nitrogen_Deficiency, Disease, Weed_Pressure, None")
    stress_severity_rating: Optional[int] = Field(None, ge=0, le=4, description="0=None, 1=Mild, 2=Moderate, 3=Severe, 4=Lethal")
    
    # In-situ instrumental telemetry
    soil_moisture: Optional[SoilMoistureMeasurement] = None
    chlorophyll: Optional[ChlorophyllMeasurement] = None
    lab_analysis: Optional[LaboratoryAnalysis] = None
    
    # Metadata & Quality Assurance
    observer: ObserverProfile
    consent: ConsentAndAccessControl
    verification_status: LabelVerificationStatus = LabelVerificationStatus.DRAFT
    second_reviewer_id: Optional[str] = None
    reviewer_notes: Optional[str] = None
    dataset_version: str = Field("v1.0.0-rc1", description="Dataset release version")
    matched_sentinel_scene_id: Optional[str] = None
    satellite_temporal_offset_days: Optional[int] = None

    @field_validator("observation_timestamp")
    @classmethod
    def validate_timestamp_not_future(cls, v: datetime.datetime) -> datetime.datetime:
        # Enforce timestamp within realistic bounds (must not be in future)
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        if v.tzinfo is None:
            v_cmp = v.replace(tzinfo=datetime.timezone.utc)
        else:
            v_cmp = v
        if v_cmp > now_utc + datetime.timedelta(hours=1):
            raise ValueError("Observation timestamp cannot be in the future.")
        return v
