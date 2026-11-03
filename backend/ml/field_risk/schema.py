"""
Pydantic Schemas and Configuration Constants for Phase 16: Field-Level AI Risk Mapping.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

# -------------------------------------------------------------------------
# Risk Classification Categories
# -------------------------------------------------------------------------
RISK_LOW = "Low Potential Risk"
RISK_MODERATE = "Moderate Potential Risk"
RISK_ELEVATED = "Elevated Potential Risk"
RISK_INSUFFICIENT_EVIDENCE = "Insufficient Evidence"

RISK_COLORS = {
    RISK_LOW: (34, 197, 94, 210),                  # Emerald Green
    RISK_MODERATE: (245, 158, 11, 220),            # Amber
    RISK_ELEVATED: (225, 29, 72, 235),             # Crimson/Rose
    RISK_INSUFFICIENT_EVIDENCE: (100, 116, 139, 0),# Transparent / Slate
}

RISK_HEX_COLORS = {
    RISK_LOW: "#22c55e",
    RISK_MODERATE: "#f59e0b",
    RISK_ELEVATED: "#e11d48",
    RISK_INSUFFICIENT_EVIDENCE: "#64748b",
}

# -------------------------------------------------------------------------
# Request Models
# -------------------------------------------------------------------------
class FieldRiskMapRequest(BaseModel):
    geometry: Optional[Dict[str, Any]] = Field(
        None, description="GeoJSON Polygon or MultiPolygon representing the field boundary."
    )
    latitude: Optional[float] = Field(
        None, ge=-90.0, le=90.0, description="Center latitude if geometry is omitted."
    )
    longitude: Optional[float] = Field(
        None, ge=-180.0, le=180.0, description="Center longitude if geometry is omitted."
    )
    radius_km: Optional[float] = Field(
        2.0, gt=0.0, le=50.0, description="Radius in kilometers if geometry is omitted."
    )
    start_date: str = Field(
        ..., description="Start date in ISO format YYYY-MM-DD."
    )
    end_date: str = Field(
        ..., description="End date in ISO format YYYY-MM-DD."
    )
    selected_indices: Optional[List[str]] = Field(
        default=["NDVI", "NDMI", "NDRE", "EVI"],
        description="List of spectral indices to compute and align."
    )
    baseline_date: Optional[str] = Field(
        None, description="Optional comparison acquisition date for historical change mapping."
    )
    selected_layer: Optional[str] = Field(
        "risk_classification", description="Selected active spatial layer key."
    )
    model_version: Optional[str] = Field(
        None, description="Requested stress model version."
    )
    max_cloud_coverage: Optional[float] = Field(
        35.0, ge=0.0, le=100.0, description="Maximum allowable cloud coverage percentage."
    )
    aoi_area_km2: Optional[float] = Field(
        None, gt=0.0, description="Optional precalculated AOI area in square kilometers."
    )
    demo_mode: Optional[bool] = Field(
        None, description="Explicit override for demo mode simulation."
    )

class PixelInspectionRequest(BaseModel):
    analysis_id: Optional[str] = Field(
        None, description="Identifier of the cached field risk analysis."
    )
    latitude: Optional[float] = Field(
        None, ge=-90.0, le=90.0, description="WGS84 latitude coordinate to inspect."
    )
    longitude: Optional[float] = Field(
        None, ge=-180.0, le=180.0, description="WGS84 longitude coordinate to inspect."
    )
    pixel_row: Optional[int] = Field(
        None, ge=0, description="Optional raster row index."
    )
    pixel_col: Optional[int] = Field(
        None, ge=0, description="Optional raster column index."
    )

# -------------------------------------------------------------------------
# Output / Response Models
# -------------------------------------------------------------------------
class RiskCategoryStats(BaseModel):
    category_name: str
    area_hectares: float
    area_sq_km: float
    percentage_of_field: float
    pixel_count: int
    color_hex: str
    description: str

class FieldSummaryStatistics(BaseModel):
    total_area_hectares: float
    total_area_sq_km: float
    valid_area_hectares: float
    valid_area_sq_km: float
    valid_pixel_percentage: float
    nodata_percentage: float
    risk_distribution: Dict[str, RiskCategoryStats]
    vegetation_indices_mean: Dict[str, Optional[float]]
    vegetation_indices_median: Dict[str, Optional[float]]
    vegetation_indices_std: Dict[str, Optional[float]]
    stress_signal_coverage_pct: float
    observation_count: int
    observation_date: str
    model_version: str
    crs: str
    spatial_resolution_m: float
    is_area_approximate: bool
    data_quality_limitations: List[str]

class PixelInspectionResult(BaseModel):
    status: str
    is_valid_pixel: bool
    pixel_coords: Dict[str, int]
    geo_coords: Dict[str, float]
    risk_category: str
    risk_color_hex: str
    stress_class: Optional[str] = None
    stress_probability: Optional[float] = None
    vegetation_indices: Dict[str, Optional[float]]
    sar_backscatter: Dict[str, Optional[float]]
    historical_change_delta: Optional[float] = None
    growth_trend_slope: Optional[float] = None
    quality_status: str
    evidence_summary: str
    scientific_disclaimer: str

class SpatialLayerMetadata(BaseModel):
    layer_key: str
    layer_name: str
    layer_type: str # "risk", "stress", "vegetation", "change", "trend", "quality"
    description: str
    image_png_base64: Optional[str] = None
    bounds: List[float] # [min_lon, min_lat, max_lon, max_lat]
    legend: Dict[str, Any]

class FieldRiskMapResponse(BaseModel):
    status: str
    analysis_id: str
    is_demo: bool
    assessment_status: str
    aoi_metadata: Dict[str, Any]
    canonical_grid: Dict[str, Any]
    statistics: FieldSummaryStatistics
    layers: Dict[str, SpatialLayerMetadata]
    available_layer_keys: List[str]
    historical_comparison_dates: List[str]
    classification_methodology: Dict[str, Any]
    data_quality: Dict[str, Any]
    model_provenance: Dict[str, Any]
    warnings: List[str]
    scientific_limitations: List[str]
