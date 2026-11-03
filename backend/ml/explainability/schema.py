"""
Schema specifications for Explainable AI, Model Reliability, and Uncertainty Intelligence (Phase 15).
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# -------------------------------------------------------------
# 37-Feature Metadata Dictionary (Display Names, Units, Roles)
# -------------------------------------------------------------

FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    # 1. Optical Spectral Indices & Bands
    "opt_ndvi": {
        "display_name": "Normalized Difference Vegetation Index (NDVI)",
        "modality": "optical",
        "unit": "index",
        "description": "Canopy greenness, chlorophyll density, and vegetative vigor.",
    },
    "opt_ndmi": {
        "display_name": "Normalized Difference Moisture Index (NDMI)",
        "modality": "optical",
        "unit": "index",
        "description": "Liquid water content within leaf spongy mesophyll tissue.",
    },
    "opt_ndre": {
        "display_name": "Normalized Difference Red Edge (NDRE)",
        "modality": "optical",
        "unit": "index",
        "description": "Red-edge sensitivity for dense canopies and nitrogen status.",
    },
    "opt_evi": {
        "display_name": "Enhanced Vegetation Index (EVI)",
        "modality": "optical",
        "unit": "index",
        "description": "Atmosphere-corrected canopy structural biomass index.",
    },
    "opt_b02": {
        "display_name": "Blue Band Reflectance (B02 - 490nm)",
        "modality": "optical",
        "unit": "reflectance",
        "description": "Atmospheric scattering and soil background reflectance.",
    },
    "opt_b03": {
        "display_name": "Green Band Reflectance (B03 - 560nm)",
        "modality": "optical",
        "unit": "reflectance",
        "description": "Green spectral peak reflectance from healthy foliage.",
    },
    "opt_b04": {
        "display_name": "Red Band Reflectance (B04 - 665nm)",
        "modality": "optical",
        "unit": "reflectance",
        "description": "Chlorophyll-a photosynthetic absorption valley.",
    },
    "opt_b05": {
        "display_name": "Red Edge Band 1 Reflectance (B05 - 705nm)",
        "modality": "optical",
        "unit": "reflectance",
        "description": "Early vegetative stress and leaf structure boundary.",
    },
    "opt_b08": {
        "display_name": "NIR Broad Band Reflectance (B08 - 842nm)",
        "modality": "optical",
        "unit": "reflectance",
        "description": "Cellular mesophyll scattering and canopy volume.",
    },
    "opt_b11": {
        "display_name": "SWIR Band Reflectance (B11 - 1610nm)",
        "modality": "optical",
        "unit": "reflectance",
        "description": "Foliage moisture absorption and surface dryness.",
    },

    # 2. Sentinel-1 SAR Channels
    "sar_vv": {
        "display_name": "SAR VV Backscatter",
        "modality": "sar",
        "unit": "dB",
        "description": "Co-polarized microwave backscatter from vertical stalks & surface moisture.",
    },
    "sar_vh": {
        "display_name": "SAR VH Backscatter",
        "modality": "sar",
        "unit": "dB",
        "description": "Cross-polarized radar return driven by volumetric canopy scattering.",
    },
    "sar_vv_vh_ratio": {
        "display_name": "SAR VV/VH Polarimetric Ratio",
        "modality": "sar",
        "unit": "ratio",
        "description": "Structural ratio separating soil geometry from dense vegetative biomass.",
    },

    # 3. Temporal Lagged Values
    "prev_ndvi": {
        "display_name": "Prior NDVI Observation",
        "modality": "temporal",
        "unit": "index",
        "description": "Previous baseline canopy greenness value.",
    },
    "prev_ndmi": {
        "display_name": "Prior NDMI Observation",
        "modality": "temporal",
        "unit": "index",
        "description": "Previous baseline canopy moisture value.",
    },
    "prev_ndre": {
        "display_name": "Prior NDRE Observation",
        "modality": "temporal",
        "unit": "index",
        "description": "Previous baseline red-edge chlorophyll value.",
    },
    "prev_evi": {
        "display_name": "Prior EVI Observation",
        "modality": "temporal",
        "unit": "index",
        "description": "Previous baseline enhanced vigor value.",
    },
    "prev_vv": {
        "display_name": "Prior SAR VV Backscatter",
        "modality": "temporal",
        "unit": "dB",
        "description": "Previous baseline vertical radar return.",
    },
    "prev_vh": {
        "display_name": "Prior SAR VH Backscatter",
        "modality": "temporal",
        "unit": "dB",
        "description": "Previous baseline volume scattering return.",
    },

    # 4. Temporal Deltas
    "delta_ndvi": {
        "display_name": "NDVI Net Change (Current - Prior)",
        "modality": "temporal",
        "unit": "Δ index",
        "description": "Net shift in vegetative canopy vigor between acquisitions.",
    },
    "delta_ndmi": {
        "display_name": "NDMI Net Change (Current - Prior)",
        "modality": "temporal",
        "unit": "Δ index",
        "description": "Net hydration change indicating moisture accretion or deficit.",
    },
    "delta_ndre": {
        "display_name": "NDRE Net Change (Current - Prior)",
        "modality": "temporal",
        "unit": "Δ index",
        "description": "Chlorophyll concentration change over observation interval.",
    },
    "delta_evi": {
        "display_name": "EVI Net Change (Current - Prior)",
        "modality": "temporal",
        "unit": "Δ index",
        "description": "Structural canopy density change between satellite passes.",
    },
    "delta_vv": {
        "display_name": "SAR VV Net Change (Current - Prior)",
        "modality": "temporal",
        "unit": "Δ dB",
        "description": "Radar backscatter delta from vertical stalk and surface shifts.",
    },
    "delta_vh": {
        "display_name": "SAR VH Net Change (Current - Prior)",
        "modality": "temporal",
        "unit": "Δ dB",
        "description": "Radar delta in volumetric biomass scattering.",
    },

    # 5. Temporal Rates of Change Per Day
    "rate_ndvi_per_day": {
        "display_name": "NDVI Rate of Change Per Day",
        "modality": "temporal",
        "unit": "/day",
        "description": "Normalized daily speed of canopy vigor growth or loss.",
    },
    "rate_ndmi_per_day": {
        "display_name": "NDMI Rate of Change Per Day",
        "modality": "temporal",
        "unit": "/day",
        "description": "Daily rate of moisture desiccation or replenishment.",
    },
    "rate_ndre_per_day": {
        "display_name": "NDRE Rate of Change Per Day",
        "modality": "temporal",
        "unit": "/day",
        "description": "Daily rate of red-edge shift in cellular chlorophyll.",
    },
    "rate_evi_per_day": {
        "display_name": "EVI Rate of Change Per Day",
        "modality": "temporal",
        "unit": "/day",
        "description": "Daily rate of structural biomass expansion or collapse.",
    },
    "temporal_interval_days": {
        "display_name": "Observation Interval Gap",
        "modality": "temporal",
        "unit": "days",
        "description": "Elapsed calendar days between current and prior acquisition.",
    },
    "usable_history_count": {
        "display_name": "Usable Historical Acquisitions",
        "modality": "temporal",
        "unit": "count",
        "description": "Total prior valid satellite acquisitions in timeline.",
    },

    # 6. Quality & Missing Data Indicators
    "cloud_coverage": {
        "display_name": "Scene Cloud Coverage",
        "modality": "quality",
        "unit": "%",
        "description": "Optical sensor cloud obscuration across satellite footprint.",
    },
    "valid_pixel_pct": {
        "display_name": "Valid Pixel Coverage",
        "modality": "quality",
        "unit": "%",
        "description": "Percentage of unmasked, unobscured canopy pixels in AOI.",
    },
    "nodata_pct": {
        "display_name": "NoData / Masked Percentage",
        "modality": "quality",
        "unit": "%",
        "description": "Percentage of pixels excluded due to cloud shadow, water, or sensor border.",
    },
    "s1_s2_temporal_gap_days": {
        "display_name": "S1-S2 Temporal Matching Gap",
        "modality": "quality",
        "unit": "days",
        "description": "Absolute time delta between Sentinel-2 optical and Sentinel-1 SAR passes.",
    },
    "missing_sar_flag": {
        "display_name": "SAR Data Absence Flag",
        "modality": "quality",
        "unit": "flag",
        "description": "Flag (1.0 = missing, 0.0 = present) indicating absence of concurrent radar data.",
    },
    "insufficient_history_flag": {
        "display_name": "Insufficient History Flag",
        "modality": "quality",
        "unit": "flag",
        "description": "Flag (1.0 = single observation, 0.0 = multi-temporal timeline exists).",
    },
}

# -------------------------------------------------------------
# Data Quality Thresholds & Evaluation
# -------------------------------------------------------------

QUALITY_TIER_GOOD = "Good"
QUALITY_TIER_MODERATE = "Moderate"
QUALITY_TIER_LIMITED = "Limited"
QUALITY_TIER_INSUFFICIENT = "Insufficient"

DEFAULT_QUALITY_THRESHOLDS = {
    "cloud_max_good": 20.0,
    "cloud_max_moderate": 40.0,
    "cloud_max_limited": 70.0,
    "valid_min_good": 85.0,
    "valid_min_moderate": 65.0,
    "valid_min_limited": 25.0,
    "history_min_good": 4,
    "history_min_moderate": 2,
    "max_s1_gap_days_good": 5.0,
}

# -------------------------------------------------------------
# Pydantic Schemas for Phase 15 Explainability & Uncertainty
# -------------------------------------------------------------

class LocalFeatureContribution(BaseModel):
    feature: str
    display_name: str
    modality: str
    value: Optional[float]
    unit: str
    contribution: float
    magnitude: float
    direction: str  # "increases_stress", "decreases_stress", "neutral"
    explanation_text: str

class GlobalFeatureImportance(BaseModel):
    feature: str
    display_name: str
    modality: str
    importance: float
    rank: int

class StressExplainability(BaseModel):
    method: str  # "SHAP_TreeExplainer", "Controlled_Perturbation", "Model_Unavailable"
    predicted_class: str
    stress_probability: Optional[float]
    local_feature_contributions: List[LocalFeatureContribution]
    global_feature_importance: Optional[List[GlobalFeatureImportance]] = None
    explanation_limitations: List[str]
    non_causation_disclaimer: str

class SingleIndexGrowthExplanation(BaseModel):
    index_name: str
    historical_direction: str
    historical_slope_per_day: Optional[float]
    selected_model_name: str
    selection_rationale: str
    recent_influencing_observations: List[Dict[str, Any]]
    candidate_model_comparisons: List[Dict[str, Any]]
    trend_explanation: str

class GrowthForecastExplainability(BaseModel):
    index_explanations: Dict[str, SingleIndexGrowthExplanation]
    overall_growth_summary: str

class DataQualityAssessment(BaseModel):
    quality_tier: str  # "Good", "Moderate", "Limited", "Insufficient"
    cloud_coverage: float
    valid_pixel_pct: float
    nodata_pct: float
    observation_count: int
    temporal_gap_days: Optional[float] = None
    s1_available: bool
    s2_available: bool
    missing_features_count: int
    score_factors: Dict[str, Any]
    data_limitations: List[str]

class ModelReliabilityMetadata(BaseModel):
    stress_model: Dict[str, Any]
    growth_model: Dict[str, Any]

class PredictionUncertaintyInfo(BaseModel):
    intervals_available: bool
    growth_intervals_summary: Dict[str, Any]
    stress_probability_distribution: Optional[Dict[str, float]] = None
    probability_margin: Optional[float] = None
    uncertainty_representation_note: str

class ExplainabilityRequest(BaseModel):
    geometry: Optional[Dict[str, Any]] = Field(default=None, description="GeoJSON polygon")
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    radius_km: Optional[float] = Field(default=2.0, gt=0.0, le=50.0)
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    explanation_type: Optional[str] = Field(
        default="all",
        description="Explanation target: 'all', 'stress_local', 'stress_global', 'growth_trend', 'data_quality', 'uncertainty'",
    )
    selected_index: Optional[str] = Field(default="NDVI", description="Target index for growth explanation")
    model_version: Optional[str] = Field(default=None)
    max_cloud_coverage: Optional[float] = Field(default=35.0, ge=0.0, le=100.0)
    aoi_area_km2: Optional[float] = Field(default=None)

class ExplainabilityResponse(BaseModel):
    status: str
    is_demo: bool
    explanation_status: str  # "EXPLANATION_AVAILABLE", "MODEL_UNAVAILABLE", "INSUFFICIENT_DATA"
    explanation_type: str
    stress_explainability: Optional[StressExplainability] = None
    growth_forecast_explainability: Optional[GrowthForecastExplainability] = None
    data_quality_assessment: DataQualityAssessment
    model_reliability: ModelReliabilityMetadata
    prediction_uncertainty: PredictionUncertaintyInfo
    warnings: List[str]
    scientific_limitations: List[str]
