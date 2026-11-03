"""
Schema, class constants, and feature specifications for the AI Early Crop Stress Detection Pipeline.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# -------------------------------------------------------------
# Official Output Class Names (Scientific & Cautious)
# -------------------------------------------------------------

CLASS_NO_RELIABLE_ASSESSMENT = "No reliable assessment"
CLASS_NO_SIGNIFICANT_STRESS = "No significant stress signal detected"
CLASS_POTENTIAL_STRESS = "Potential stress signal detected"
CLASS_ELEVATED_STRESS = "Elevated potential stress signal"

STRESS_CLASSES: List[str] = [
    CLASS_NO_SIGNIFICANT_STRESS,
    CLASS_POTENTIAL_STRESS,
    CLASS_ELEVATED_STRESS,
]

ALL_STATUS_CLASSES: List[str] = [
    CLASS_NO_RELIABLE_ASSESSMENT,
    CLASS_NO_SIGNIFICANT_STRESS,
    CLASS_POTENTIAL_STRESS,
    CLASS_ELEVATED_STRESS,
]

# Model internal target integers for trained classifiers (3-class active model)
LABEL_MAPPING = {
    CLASS_NO_SIGNIFICANT_STRESS: 0,
    CLASS_POTENTIAL_STRESS: 1,
    CLASS_ELEVATED_STRESS: 2,
}

REVERSE_LABEL_MAPPING = {
    0: CLASS_NO_SIGNIFICANT_STRESS,
    1: CLASS_POTENTIAL_STRESS,
    2: CLASS_ELEVATED_STRESS,
}

# -------------------------------------------------------------
# Standard Engineered Feature Specification (29 Features)
# -------------------------------------------------------------

OPTICAL_FEATURES = [
    "opt_ndvi",
    "opt_ndmi",
    "opt_ndre",
    "opt_evi",
    "opt_b02",
    "opt_b03",
    "opt_b04",
    "opt_b05",
    "opt_b08",
    "opt_b11",
]

SAR_FEATURES = [
    "sar_vv",
    "sar_vh",
    "sar_vv_vh_ratio",
]

TEMPORAL_FEATURES = [
    "prev_ndvi",
    "prev_ndmi",
    "prev_ndre",
    "prev_evi",
    "prev_vv",
    "prev_vh",
    "delta_ndvi",
    "delta_ndmi",
    "delta_ndre",
    "delta_evi",
    "delta_vv",
    "delta_vh",
    "rate_ndvi_per_day",
    "rate_ndmi_per_day",
    "rate_ndre_per_day",
    "rate_evi_per_day",
    "temporal_interval_days",
    "usable_history_count",
]

QUALITY_FEATURES = [
    "cloud_coverage",
    "valid_pixel_pct",
    "nodata_pct",
    "s1_s2_temporal_gap_days",
    "missing_sar_flag",
    "insufficient_history_flag",
]

ALL_MODEL_FEATURE_NAMES: List[str] = (
    OPTICAL_FEATURES + SAR_FEATURES + TEMPORAL_FEATURES + QUALITY_FEATURES
)

# -------------------------------------------------------------
# Metadata & Validation Models
# -------------------------------------------------------------

class FeatureSchema(BaseModel):
    name: str
    modality: str  # optical, sar, temporal, quality
    description: str
    unit: str
    expected_min: Optional[float] = None
    expected_max: Optional[float] = None
    can_be_nan: bool = True

class ModelMetadata(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    created_at: str
    features: List[str]
    target_classes: List[str]
    is_demo_model: bool
    is_calibrated: bool
    training_dataset_version: Optional[str] = None
    dataset_provenance: Optional[str] = None
    split_method: Optional[str] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None
    scientific_disclaimer: str
