"""
ml/crop module:
Supervised and unsupervised crop stress evaluation:
- Descals et al. optical phenology window baseline
- Improved Multimodal S1+S2 fusion classifier (XGBoost / Random Forest) with TreeSHAP
- Spatial Block Cross-Validation (GroupKFold) with 95% Confidence Intervals
- Sensor modality ablations (S2-only vs S1-only vs S1+S2)
"""

from .dataset import load_or_create_labels_dataset, extract_multimodal_features
from .baseline_descals import DescalsPhenologyBaseline
from .improved_model import MultimodalCropClassifier
from .spatial_cv import evaluate_spatial_block_cv
from .ablations import run_sensor_ablations, run_ablations

__all__ = [
    "load_or_create_labels_dataset",
    "extract_multimodal_features",
    "DescalsPhenologyBaseline",
    "MultimodalCropClassifier",
    "evaluate_spatial_block_cv",
    "run_sensor_ablations",
    "run_ablations",
]
