"""
ml/crop/baseline_descals.py:
Baseline Model reproducing Descals et al. 2025, Remote Sensing 17(2):340.

Uses Sentinel-2 phenology / temporal NDVI window features:
- early-stage mean NDVI (early vegetative phase)
- mid-stage peak NDVI (reproductive/flowering phase)
- late-stage mean NDVI (maturation/senescence phase)
- seasonal NDVI amplitude (peak - early)
- greenup rate & senescence rate
- seasonal mean NDVI
"""

from typing import Dict, Any
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.crop.spatial_cv import evaluate_spatial_block_cv


class DescalsPhenologyBaseline:
    """
    Classifier reproducing Descals et al. phenology-driven crop stress classification
    using strictly optical Sentinel-2 temporal window features.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def create_estimator(self):
        """Returns fresh estimator instance configured as in Descals et al."""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=3,
            random_state=self.random_seed,
            class_weight="balanced",
        )

    def evaluate_spatial_cv(
        self,
        X_descals: pd.DataFrame,
        y: pd.Series,
        spatial_groups: pd.Series,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """Runs spatial block cross-validation on optical phenology features."""
        cv_results = evaluate_spatial_block_cv(
            model_factory=self.create_estimator,
            X=X_descals,
            y=y,
            spatial_groups=spatial_groups,
            n_splits=n_splits,
            random_seed=self.random_seed,
        )

        return {
            "model_name": "Descals et al. (S2 Phenology Window Baseline)",
            "citation": "Descals et al. 2025, Remote Sensing 17(2):340",
            "input_sensor": "Sentinel-2 MSI Level-2A (Optical only)",
            "features_used": list(X_descals.columns),
            "feature_count": len(X_descals.columns),
            "cross_validation_strategy": "Spatial Block GroupKFold (Non-overlapping geographic clusters)",
            "results": cv_results,
        }

