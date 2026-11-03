from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field

# -------------------------------------------------------------
# Structured Input Evidence Models
# -------------------------------------------------------------

class LocationEvidence(BaseModel):
    latitude: float = Field(..., description="Center latitude of AOI")
    longitude: float = Field(..., description="Center longitude of AOI")
    name: Optional[str] = Field(default=None, description="Location / field name")
    radius_km: Optional[float] = Field(default=None, description="AOI radius in kilometers")
    area_sq_km: Optional[float] = Field(default=None, description="AOI area in square kilometers")

class ObservationEvidence(BaseModel):
    date: str = Field(..., description="Acquisition date (YYYY-MM-DD)")
    cloud_coverage: float = Field(..., description="Cloud coverage percentage")
    valid_pixel_percentage: float = Field(..., description="Percentage of valid data pixels")
    scene_id: Optional[str] = Field(default=None, description="Satellite scene product ID")
    data_source: Optional[str] = Field(default="Copernicus Sentinel-2", description="Satellite constellation")

class IndexStatsEvidence(BaseModel):
    mean: Optional[float] = Field(default=None, description="Mean index value")
    median: Optional[float] = Field(default=None, description="Median index value")
    min: Optional[float] = Field(default=None, description="Minimum index value")
    max: Optional[float] = Field(default=None, description="Maximum index value")

class IndicesEvidence(BaseModel):
    ndvi: Optional[IndexStatsEvidence] = Field(default=None, description="Normalized Difference Vegetation Index")
    ndmi: Optional[IndexStatsEvidence] = Field(default=None, description="Normalized Difference Moisture Index")
    ndre: Optional[IndexStatsEvidence] = Field(default=None, description="Normalized Difference Red Edge Index")
    evi: Optional[IndexStatsEvidence] = Field(default=None, description="Enhanced Vegetation Index")

class ConditionEvidence(BaseModel):
    score: Optional[float] = Field(default=None, description="Calculated composite condition score (0-100)")
    category: Optional[str] = Field(default=None, description="Overall condition category (e.g. Good, Moderate)")
    dominant_zone: Optional[str] = Field(default=None, description="Dominant spatial vigor zone")
    zones: Optional[Dict[str, float]] = Field(default=None, description="Percentage breakdown across condition zones")

class HistoricalEvidence(BaseModel):
    observation_count: Optional[int] = Field(default=None, description="Number of historical observations in window")
    start_date: Optional[str] = Field(default=None, description="Earliest historical observation date")
    latest_date: Optional[str] = Field(default=None, description="Latest historical observation date")
    trends: Optional[Dict[str, Any]] = Field(default=None, description="Temporal index trends (e.g. ndvi trend, delta)")
    summary: Optional[str] = Field(default=None, description="Pre-calculated trend interpretation statement")

class ComparisonEvidence(BaseModel):
    delta_ndvi: Optional[float] = Field(default=None, description="NDVI difference (Date B - Date A)")
    delta_ndmi: Optional[float] = Field(default=None, description="NDMI difference (Date B - Date A)")
    delta_ndre: Optional[float] = Field(default=None, description="NDRE difference (Date B - Date A)")
    delta_evi: Optional[float] = Field(default=None, description="EVI difference (Date B - Date A)")
    breakdown: Optional[Dict[str, float]] = Field(default=None, description="Spatial shift breakdown percentages")

class QualityEvidence(BaseModel):
    valid_percentage: float = Field(..., description="Valid pixel coverage percentage")
    quality_level: str = Field(..., description="Quality tier (Excellent, Good, Fair, Poor, Limited)")
    limitations: Optional[List[str]] = Field(default_factory=list, description="Specific observed data limitations")

class FieldEvidenceInput(BaseModel):
    """
    Validated structured input container provided to the AI interpretation layer.
    Only includes verifiable, pre-calculated analytical evidence from Sentinel-2.
    """
    location: Optional[LocationEvidence] = None
    observation: Optional[ObservationEvidence] = None
    indices: Optional[IndicesEvidence] = None
    condition: Optional[ConditionEvidence] = None
    historical: Optional[HistoricalEvidence] = None
    comparison: Optional[ComparisonEvidence] = None
    quality: Optional[QualityEvidence] = None
    is_demo: bool = False

# -------------------------------------------------------------
# Structured AI Output Models
# -------------------------------------------------------------

class ObservationItem(BaseModel):
    title: str = Field(..., description="Brief headline of the observation")
    description: str = Field(..., description="Detailed analytical description")
    evidence: List[str] = Field(default_factory=list, description="Specific numerical values backing this observation")

class PatternItem(BaseModel):
    pattern: str = Field(..., description="Observed spatial or spectral pattern")
    evidence: str = Field(..., description="Metrics indicating this pattern")
    severity: Literal["low", "moderate", "high"] = Field(..., description="Analytical significance or stress tier")
    explanation: str = Field(..., description="Scientifically grounded explanation of plausible factors")

class HistoricalInterpretation(BaseModel):
    summary: str = Field(..., description="Summary of multi-temporal trajectory")
    trend: Literal["increasing", "stable", "decreasing", "insufficient_data"] = Field(..., description="Classified direction")

class AreaToWatch(BaseModel):
    description: str = Field(..., description="Description of spatial zone or metric needing attention")
    reason: str = Field(..., description="Underlying spectral evidence")

class DataQualityAssessment(BaseModel):
    assessment: str = Field(..., description="Assessment of observational confidence")
    limitations: List[str] = Field(default_factory=list, description="Specific observational caveats")

class AIFieldReport(BaseModel):
    """
    Strictly validated structured response schema for AI Field Intelligence.
    Ensures explainability, evidence linkage, and compliance with scientific guardrails.
    """
    headline: str = Field(..., description="High-level field intelligence summary title")
    summary: str = Field(..., description="Comprehensive analytical synthesis of field condition")
    observations: List[ObservationItem] = Field(default_factory=list, description="Key evidence-based observations")
    possible_patterns: List[PatternItem] = Field(default_factory=list, description="Identified patterns and plausible factors")
    historical_interpretation: Optional[HistoricalInterpretation] = Field(default=None, description="Temporal trajectory analysis")
    areas_to_watch: List[AreaToWatch] = Field(default_factory=list, description="Zones or indicators warranting monitoring")
    data_quality: DataQualityAssessment = Field(..., description="Data quality confidence and limitations")
    recommended_next_steps: List[str] = Field(default_factory=list, description="Non-prescriptive field verification steps")
    confidence: Literal["low", "moderate", "high"] = Field(..., description="Overall assessment confidence level")
    is_demo: bool = Field(default=False, description="Flag indicating deterministic demo analysis")
    generated_at: Optional[str] = Field(default=None, description="ISO timestamp of report generation")
    provider_used: Optional[str] = Field(default=None, description="AI provider identifier")
