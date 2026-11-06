"""
Data schemas for Phase 17: Advanced Farmer Intelligence Dashboard.

Defines Pydantic models for:
- Field overview & catalog
- Crop condition & multi-spectral indicators
- Early stress intelligence & feature attributions
- Growth trend forecasting & prediction intervals
- Field-level spatial risk summary
- Historical monitoring timeline
- Evidence-linked agricultural recommendations
- Multi-dimensional data reliability indicators
- Multi-field comparison
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# 1. Field Overview & Catalog
# -----------------------------------------------------------------------------

class FieldOverview(BaseModel):
    field_id: str
    field_name: str
    crop_type: Optional[str] = "Paddy / Rice"
    location_name: str
    latitude: float
    longitude: float
    radius_km: float
    total_area_hectares: float
    total_area_sq_km: float
    geometry: Optional[Dict[str, Any]] = None
    latest_observation_date: str
    sensors_available: List[str] = Field(default_factory=lambda: ["Sentinel-2 MSI"])
    data_quality_status: str # "Good", "Moderate", "Limited", "Insufficient"
    last_analysis_status: str # "SUCCESS", "ANALYSIS_AVAILABLE", "PENDING"
    is_demo: bool = True

class FieldOverviewListItem(BaseModel):
    field_id: str
    field_name: str
    crop_type: str
    location_name: str
    area_hectares: float
    latest_observation_date: str
    overall_risk_level: str # "Low", "Moderate", "Elevated", "Insufficient Evidence"
    data_quality_tier: str
    mean_ndvi: Optional[float] = None
    is_demo: bool = True

class DashboardOverviewResponse(BaseModel):
    status: str
    total_registered_fields: int
    fields: List[FieldOverviewListItem]
    active_field_id: str
    is_demo_mode: bool
    platform_version: str = "v1.7.0"

# -----------------------------------------------------------------------------
# 2. Crop Condition & Vegetation Metrics
# -----------------------------------------------------------------------------

class VegetationIndicatorMetric(BaseModel):
    name: str # e.g. "NDVI"
    label: str # e.g. "Vegetation Vigor"
    latest_value: Optional[float]
    previous_value: Optional[float]
    delta_absolute: Optional[float]
    delta_percentage: Optional[float]
    trend_direction: str # "Increasing", "Stable", "Decreasing", "Insufficient Data"
    observation_date: str
    previous_observation_date: Optional[str]
    quality_status: str
    physical_indicator: str
    status_color: str # "#10b981", "#f59e0b", "#ef4444"

class CropConditionSummary(BaseModel):
    dominant_condition: str
    condition_score: float # 0.0 to 100.0
    healthy_canopy_pct: float
    moderate_canopy_pct: float
    stressed_canopy_pct: float
    observation_date: str
    evaluation_method: str = "Multispectral Radiometric Assessment"

# -----------------------------------------------------------------------------
# 3. Vigour Anomaly Intelligence
# -----------------------------------------------------------------------------

class FeatureContribution(BaseModel):
    feature_name: str
    feature_label: str
    measured_value: Optional[float]
    contribution_weight: float
    direction: str # "increases_stress", "decreases_stress", "neutral"

class StressIntelligenceSummary(BaseModel):
    is_model_available: bool
    status_message: str
    stress_class: Optional[str] = None
    stress_probability: Optional[float] = None
    confidence_level: Optional[str] = None
    top_contributing_features: List[FeatureContribution] = Field(default_factory=list)
    missing_sensors: List[str] = Field(default_factory=list)
    model_name: str
    model_version: str
    is_demo: bool = True
    ground_truth_advisory: str

# -----------------------------------------------------------------------------
# 4. Growth Trend & Forecast Summary
# -----------------------------------------------------------------------------

class PredictionInterval(BaseModel):
    date: str
    lower: float
    upper: float
    confidence_level: float = 0.80

class GrowthForecastSummary(BaseModel):
    target_index: str = "NDVI"
    baseline_algorithm: str
    forecast_horizon: str # "short", "medium", "long"
    historical_slope_per_day: Optional[float]
    forecast_dates: List[str]
    projected_values: List[float]
    prediction_intervals: List[PredictionInterval] = Field(default_factory=list)
    walk_forward_metrics: Dict[str, Any] = Field(default_factory=dict)
    forecast_limitations: List[str] = Field(default_factory=list)
    disclaimer: str

# -----------------------------------------------------------------------------
# 5. Field-Level Risk Summary
# -----------------------------------------------------------------------------

class FieldRiskCategoryItem(BaseModel):
    category_name: str
    area_hectares: float
    percentage_of_field: float
    pixel_count: int
    color_hex: str
    description: str

class FieldRiskSummary(BaseModel):
    analysis_id: Optional[str]
    total_area_hectares: float
    valid_canopy_hectares: float
    valid_canopy_percentage: float
    nodata_percentage: float
    stress_signal_coverage_pct: float
    risk_distribution: Dict[str, FieldRiskCategoryItem]
    canonical_crs: str
    spatial_resolution_m: float
    available_layer_keys: List[str]
    has_geotiff_export: bool = True

# -----------------------------------------------------------------------------
# 6. Historical Monitoring Timeline
# -----------------------------------------------------------------------------

class HistoricalObservationSummary(BaseModel):
    date: str
    cloud_coverage: float
    valid_pixel_pct: float
    ndvi: Optional[float]
    ndmi: Optional[float]
    ndre: Optional[float]
    evi: Optional[float]
    quality_verdict: str

class HistoricalMonitoringSummary(BaseModel):
    observation_count: int
    time_span_days: int
    date_range: List[str]
    timeline: List[HistoricalObservationSummary]
    overall_trend_direction: str
    median_cloud_coverage: float

# -----------------------------------------------------------------------------
# 7. Evidence-Linked Agricultural Recommendations
# -----------------------------------------------------------------------------

class AgriculturalRecommendation(BaseModel):
    id: str
    category: str # "Field Scouting", "Moisture Verification", "Satellite Recheck", "Nutrient Assessment", "Temporal Review"
    title: str
    action: str
    reason: str
    evidence_source: str
    evidence_date: str
    data_quality_limitations: List[str] = Field(default_factory=list)
    field_verification_required: bool = True
    priority: str = "medium" # "high", "medium", "low", "info"
    created_at: str

# -----------------------------------------------------------------------------
# 8. Multi-Dimensional Data Reliability
# -----------------------------------------------------------------------------

class DataReliabilitySummary(BaseModel):
    satellite_data_quality: str # "Good", "Moderate", "Limited", "Insufficient"
    valid_pixel_percentage: float
    cloud_coverage_percentage: float
    observation_count: int
    optical_sensor_status: str # "Sentinel-2 L2A Active"
    sar_sensor_status: str # "Sentinel-1 GRD Co-Registered" or "Missing"
    stress_model_status: str # "CALIBRATED_ACTIVE" or "GROUND_TRUTH_REQUIRED"
    stress_model_brier_score: Optional[float]
    forecast_model_validated: bool
    forecast_model_mae: Optional[float]
    uncertainty_intervals_available: bool
    is_demo: bool = True
    non_causation_disclaimer: str

# -----------------------------------------------------------------------------
# 9. Unified Farmer Dashboard Response
# -----------------------------------------------------------------------------

class FarmerDashboardResponse(BaseModel):
    status: str
    field_overview: FieldOverview
    crop_condition: CropConditionSummary
    vegetation_metrics: Dict[str, VegetationIndicatorMetric]
    stress_intelligence: StressIntelligenceSummary
    growth_forecast: GrowthForecastSummary
    field_risk: FieldRiskSummary
    historical_monitoring: HistoricalMonitoringSummary
    recommendations: List[AgriculturalRecommendation]
    data_reliability: DataReliabilitySummary
    warnings: List[str] = Field(default_factory=list)
    scientific_disclaimers: List[str] = Field(default_factory=list)

# -----------------------------------------------------------------------------
# 10. Multi-Field Comparison
# -----------------------------------------------------------------------------

class FieldComparisonItem(BaseModel):
    field_id: str
    field_name: str
    crop_type: str
    area_hectares: float
    observation_date: str
    data_quality_tier: str
    ndvi_mean: Optional[float]
    ndmi_mean: Optional[float]
    ndre_mean: Optional[float]
    evi_mean: Optional[float]
    growth_trend_slope: Optional[float]
    elevated_risk_pct: float
    stress_probability: Optional[float]
    stress_class: Optional[str]

class FieldComparisonResponse(BaseModel):
    status: str
    compared_fields: List[FieldComparisonItem]
    temporal_alignment_compatible: bool
    temporal_discrepancy_days: int
    comparison_warnings: List[str] = Field(default_factory=list)
    is_demo: bool = True

# -----------------------------------------------------------------------------
# 11. Request Schemas
# -----------------------------------------------------------------------------

class RegisterFieldRequest(BaseModel):
    field_name: str
    crop_type: Optional[str] = "Paddy / Rice"
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    radius_km: Optional[float] = Field(2.0, gt=0.0, le=50.0)
    geometry: Optional[Dict[str, Any]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    max_cloud_coverage: Optional[float] = Field(35.0, ge=0.0, le=100.0)
