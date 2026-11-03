"""
Schema, class constants, and data models for AI-Powered Crop Growth & Trend Prediction (Phase 14).
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# -------------------------------------------------------------
# Supported Target Indices & Modalities
# -------------------------------------------------------------

OPTICAL_INDICES = ["NDVI", "NDMI", "NDRE", "EVI"]
SAR_INDICES = ["VV", "VH"]
SUPPORTED_INDICES = OPTICAL_INDICES + SAR_INDICES

# Physical clamp boundaries to prevent impossible extrapolation
PHYSICAL_BOUNDS: Dict[str, tuple] = {
    "NDVI": (-0.2, 1.0),
    "NDMI": (-1.0, 1.0),
    "NDRE": (-0.2, 1.0),
    "EVI": (-0.2, 1.5),
    "VV": (-30.0, 0.0),    # C-band radar backscatter in dB
    "VH": (-35.0, -5.0),   # C-band radar cross-pol in dB
}

INDEX_DESCRIPTIONS: Dict[str, str] = {
    "NDVI": "Normalized Difference Vegetation Index (Canopy vigor & green biomass)",
    "NDMI": "Normalized Difference Moisture Index (Canopy cellular water content)",
    "NDRE": "Normalized Difference Red Edge (Early chlorophyll & nitrogen dynamics)",
    "EVI": "Enhanced Vegetation Index (High-biomass canopy structural sensitivity)",
    "VV": "Sentinel-1 C-band vertical co-polarization backscatter in dB (Surface roughness & soil/stem interaction)",
    "VH": "Sentinel-1 C-band cross-polarization backscatter in dB (Volumetric canopy volume scattering)",
}

# -------------------------------------------------------------
# Trend Directions & Categorization
# -------------------------------------------------------------

TREND_INCREASING = "Increasing"
TREND_DECREASING = "Decreasing"
TREND_STABLE = "Relatively Stable"
TREND_INSUFFICIENT = "Insufficient Evidence"

TREND_DIRECTIONS = [
    TREND_INCREASING,
    TREND_DECREASING,
    TREND_STABLE,
    TREND_INSUFFICIENT,
]

# Configurable sensitivity thresholds for trend classification
DEFAULT_TREND_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "optical": {
        "min_delta": 0.03,            # Total absolute index change threshold
        "min_slope_per_day": 0.0008,  # Rate of change per day threshold (~0.024 / month)
        "min_observations": 3,        # Minimum observation count needed for reliable trend
    },
    "sar": {
        "min_delta": 1.0,             # Decibel difference threshold
        "min_slope_per_day": 0.025,   # dB per day
        "min_observations": 3,
    },
}

# -------------------------------------------------------------
# Forecast Horizons
# -------------------------------------------------------------

FORECAST_HORIZONS: Dict[str, Dict[str, Any]] = {
    "short": {
        "steps": 2,
        "nominal_days": 14,
        "label": "Short-Term (~14 Days)",
        "description": "Projection over next 2 satellite revisit cycles (~10-14 days).",
    },
    "medium": {
        "steps": 4,
        "nominal_days": 28,
        "label": "Medium-Term (~30 Days)",
        "description": "Projection over next 4 satellite revisit cycles (~24-30 days).",
    },
    "long": {
        "steps": 6,
        "nominal_days": 42,
        "label": "Longer-Term (~45-60 Days)",
        "description": "Projection over next 6 satellite revisit cycles (~36-45 days).",
    },
}

# -------------------------------------------------------------
# Pydantic Request & Response Models
# -------------------------------------------------------------

class GrowthPredictionRequest(BaseModel):
    geometry: Optional[Dict[str, Any]] = Field(default=None, description="GeoJSON Polygon geometry")
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0, description="Center latitude")
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0, description="Center longitude")
    radius_km: Optional[float] = Field(default=2.0, gt=0.0, le=50.0, description="Radius in kilometers")
    start_date: str = Field(..., description="Start date of historical window (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date of historical window (YYYY-MM-DD)")
    forecast_horizon: Optional[str] = Field(
        default="medium",
        description="Forecast horizon preset: 'short' (2 steps), 'medium' (4 steps), or 'long' (6 steps)",
    )
    selected_indices: Optional[List[str]] = Field(
        default=["NDVI", "NDMI", "NDRE", "EVI"],
        description="List of vegetation/SAR indices to forecast: 'NDVI', 'NDMI', 'NDRE', 'EVI', 'VV', 'VH'",
    )
    model_type: Optional[str] = Field(
        default="linear_trend",
        description="Forecasting model choice: 'linear_trend', 'moving_average', 'last_observation', 'seasonal_naive', or 'best_validated'",
    )
    max_cloud_coverage: Optional[float] = Field(default=35.0, ge=0.0, le=100.0)
    aoi_area_km2: Optional[float] = Field(default=None)

class PredictionInterval(BaseModel):
    lower: float
    upper: float
    confidence_level: float = 0.80

class TrendStatistics(BaseModel):
    index_name: str
    current_value: Optional[float]
    first_value: Optional[float]
    absolute_change: Optional[float]
    relative_change_pct: Optional[float]
    trend_slope_per_day: Optional[float]
    trend_direction: str
    observation_count: int
    time_span_days: int
    description: str

class ModelValidationMetric(BaseModel):
    model_name: str
    index_name: str
    mae: Optional[float]
    rmse: Optional[float]
    r2: Optional[float]
    mape: Optional[float]
    prediction_interval_coverage: Optional[float]
    validation_windows_count: int
    is_best_model: bool

class HistoricalComparisonPeriod(BaseModel):
    period_name: str
    date_range: List[str]
    observation_count: int
    mean_values: Dict[str, Optional[float]]
    net_changes: Dict[str, Optional[float]]
    trajectory_note: str

class GrowthPredictionResponse(BaseModel):
    status: str
    assessment_status: str
    is_demo: bool
    model_name: str
    model_version: str
    forecast_horizon: str
    forecast_dates: List[str]
    forecast_values: Dict[str, List[float]]
    prediction_intervals: Dict[str, List[Dict[str, float]]]
    trend_statistics: Dict[str, TrendStatistics]
    historical_observations: List[Dict[str, Any]]
    validation_metrics: Dict[str, Dict[str, Any]]
    data_quality: Dict[str, Any]
    observation_count: int
    historical_comparison: Dict[str, Any]
    warnings: List[str]
    limitations: List[str]
    scientific_disclaimer: str
