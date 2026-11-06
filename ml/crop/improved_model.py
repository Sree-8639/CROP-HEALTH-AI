"""
ml/crop/improved_model.py:
Improved Multimodal S1+S2 Fusion Model with XGBoost / RandomForest and SHAP attribution.

Features:
- Sentinel-2: NDVI, NDMI (moisture), NDRE (chlorophyll), EVI (canopy structure)
- Sentinel-1: VV dB, VH dB, VV-VH dB difference ratio, linear ratio, backscatter stats
- Multimodal interactions: SAR-optical canopy coupling
- SHAP: TreeExplainer attribution values quantify feature importance rankings
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap

from ml.crop.spatial_cv import evaluate_spatial_block_cv


class MultimodalCropClassifier:
    """
    Improved Multimodal S1+S2 Crop Stress Classifier with SHAP explainability.
    """

    def __init__(self, model_type: str = "xgboost", random_seed: int = 42):
        self.model_type = model_type.lower()
        self.random_seed = random_seed
        self.fitted_model = None

    def create_estimator(self):
        """Returns fresh estimator instance."""
        if self.model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=self.random_seed,
                eval_metric="mlogloss",
            )
        else:
            return RandomForestClassifier(
                n_estimators=120,
                max_depth=7,
                min_samples_leaf=2,
                random_state=self.random_seed,
                class_weight="balanced",
            )

    def evaluate_spatial_cv(
        self,
        X_fused: pd.DataFrame,
        y: pd.Series,
        spatial_groups: pd.Series,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """Runs spatial block cross-validation and computes SHAP feature attributions."""
        # 1. Spatial Block Cross-Validation
        cv_results = evaluate_spatial_block_cv(
            model_factory=self.create_estimator,
            X=X_fused,
            y=y,
            spatial_groups=spatial_groups,
            n_splits=n_splits,
            random_seed=self.random_seed,
        )

        # 2. Fit full model on all samples to derive global SHAP attribution
        full_model = self.create_estimator()
        full_model.fit(X_fused, y)
        self.fitted_model = full_model

        # 3. Compute SHAP explanations
        shap_analysis = self._compute_shap_attribution(full_model, X_fused)

        return {
            "model_name": f"Multimodal Fused S1+S2 ({self.model_type.upper()})",
            "model_architecture": self.model_type.upper(),
            "input_sensor": "Sentinel-2 MSI Level-2A + Sentinel-1 C-SAR GRD (Multimodal Fused)",
            "features_used": list(X_fused.columns),
            "feature_count": len(X_fused.columns),
            "cross_validation_strategy": "Spatial Block GroupKFold (Non-overlapping geographic clusters)",
            "results": cv_results,
            "shap_attribution": shap_analysis,
        }

    def _compute_shap_attribution(
        self, model, X: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculates TreeSHAP feature attributions and ranks importance."""
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # Handle multiclass vs binary shap values
            if isinstance(shap_values, list):
                # Multiclass: average absolute SHAP values across classes
                mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # Shape (samples, features, classes)
                mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)

            # Rank features by mean |SHAP| value
            feature_names = list(X.columns)
            ranked_pairs = sorted(
                zip(feature_names, mean_abs_shap.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )

            total_importance = sum(v for _, v in ranked_pairs) or 1.0

            rankings = []
            for rank, (feat, score) in enumerate(ranked_pairs, start=1):
                pct = round((score / total_importance) * 100.0, 2)
                sensor_tag = "SAR" if any(k in feat for k in ["vv", "vh", "sar"]) else "Optical"
                rankings.append({
                    "rank": rank,
                    "feature": feat,
                    "sensor": sensor_tag,
                    "mean_abs_shap": round(float(score), 5),
                    "relative_importance_pct": pct,
                })

            # Calculate optical vs SAR cumulative contribution
            sar_importance = sum(r["relative_importance_pct"] for r in rankings if r["sensor"] == "SAR")
            optical_importance = sum(r["relative_importance_pct"] for r in rankings if r["sensor"] == "Optical")

            return {
                "explainer_type": "TreeSHAP (Lundberg & Lee, 2017)",
                "sample_count": len(X),
                "sensor_attribution_balance": {
                    "optical_relative_pct": round(optical_importance, 2),
                    "sar_relative_pct": round(sar_importance, 2),
                },
                "feature_importance_rankings": rankings,
            }
        except Exception as exc:
            return {
                "explainer_type": "TreeSHAP (Fallback)",
                "error": str(exc),
                "feature_importance_rankings": [],
            }

