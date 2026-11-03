"""
Local and Global Model Explainability Engine (Phase 15).

Implements:
1. SHAP TreeExplainer for tree ensembles (RandomForest inside CalibratedClassifierCV)
2. Controlled feature perturbation fallback for arbitrary estimators
3. Global feature importance aggregation
4. Non-causation boundaries and agronomic disclaimers
"""

import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from ml.explainability.schema import (
    FEATURE_METADATA,
    LocalFeatureContribution,
    GlobalFeatureImportance,
    StressExplainability,
)
from ml.stress_detection.schema import (
    ALL_MODEL_FEATURE_NAMES,
    CLASS_NO_SIGNIFICANT_STRESS,
    CLASS_POTENTIAL_STRESS,
    CLASS_ELEVATED_STRESS,
)

NON_CAUSATION_DISCLAIMER = (
    "Feature contributions represent mathematical attribution within the machine-learning model "
    "based on satellite radiometric and radar reflectance signals. They do NOT establish biological or "
    "pathological causation (such as diagnosing specific bacterial/fungal diseases, pests, or fertilizer deficits) "
    "without in-field ground verification."
)


def _generate_feature_sentence(
    feat_name: str,
    feat_meta: Dict[str, str],
    val: Optional[float],
    direction: str,
    contrib: float,
) -> str:
    """Generates an intuitive, non-causal explanation sentence for a feature contribution."""
    disp = feat_meta.get("display_name", feat_name)
    unit = feat_meta.get("unit", "")
    val_str = f"{val:.3f} {unit}".strip() if val is not None else "unavailable"

    if direction == "increases_stress":
        if "ndmi" in feat_name:
            return f"{disp} of {val_str} indicates reduced canopy water content, contributing to elevated stress signals."
        elif "ndre" in feat_name:
            return f"{disp} of {val_str} reflects lower red-edge chlorophyll absorption, pointing toward vegetative strain."
        elif "ndvi" in feat_name:
            return f"{disp} of {val_str} shows depressed foliage greenness relative to typical vigorous canopy levels."
        elif "sar" in feat_name or "vh" in feat_name:
            return f"{disp} of {val_str} indicates reduced volume scattering or altered structural roughness."
        elif "delta" in feat_name or "rate" in feat_name:
            return f"{disp} of {val_str} reveals a downward trajectory compared to prior observations."
        else:
            return f"{disp} ({val_str}) mathematically elevated the model's stress assessment."
    elif direction == "decreases_stress":
        if "ndmi" in feat_name or "ndre" in feat_name or "ndvi" in feat_name or "evi" in feat_name:
            return f"{disp} of {val_str} remains in a healthy vigorous range, countering stress signals."
        elif "sar" in feat_name:
            return f"{disp} of {val_str} maintains robust structural backscatter consistency."
        else:
            return f"{disp} ({val_str}) mathematically supported non-stressed vegetative conditions."
    else:
        return f"{disp} ({val_str}) had minimal influence on the classification outcome."


class StressModelExplainer:
    """
    Computes local and global explanations for the crop stress detection model.
    """

    def __init__(self, model: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        self.model = model
        self.metadata = metadata or {}
        self.feature_names = self.metadata.get("features", ALL_MODEL_FEATURE_NAMES)

    def explain_instance(
        self,
        feature_dict: Dict[str, Any],
        predicted_class: str,
        stress_probability: Optional[float],
        predicted_class_idx: int = 1,
        prefer_method: str = "auto",
    ) -> StressExplainability:
        """
        Generates local observation-level feature attribution and global importances.
        """
        if self.model is None:
            return StressExplainability(
                method="Model_Unavailable",
                predicted_class=predicted_class,
                stress_probability=stress_probability,
                local_feature_contributions=[],
                global_feature_importance=self.get_global_importance(),
                explanation_limitations=[
                    "Production stress model artifact is not loaded.",
                    "No feature attributions can be computed without an active trained estimator.",
                ],
                non_causation_disclaimer=NON_CAUSATION_DISCLAIMER,
            )

        # Assemble feature array in exact model order
        x_raw = []
        for feat in self.feature_names:
            v = feature_dict.get(feat)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                x_raw.append(np.nan)
            else:
                x_raw.append(float(v))
        X_vec = np.array(x_raw).reshape(1, -1)

        # Attempt SHAP TreeExplainer first if available and applicable
        method_used = "Controlled_Perturbation"
        contributions: Optional[np.ndarray] = None

        if HAS_SHAP and prefer_method in ("auto", "shap"):
            try:
                contributions, method_used = self._compute_shap_attributions(
                    X_vec, target_class_idx=predicted_class_idx
                )
            except Exception:
                contributions = None

        # Fallback to Controlled Feature Perturbation if SHAP is unavailable or failed
        if contributions is None:
            contributions = self._compute_perturbation_attributions(
                X_vec, target_class_idx=predicted_class_idx
            )
            method_used = "Controlled_Perturbation"

        # Format local feature contributions
        local_contributions: List[LocalFeatureContribution] = []
        for i, feat_name in enumerate(self.feature_names):
            meta = FEATURE_METADATA.get(feat_name, {
                "display_name": feat_name,
                "modality": "unknown",
                "unit": "",
                "description": "",
            })
            raw_val = x_raw[i] if not math.isnan(x_raw[i]) else None
            c_val = float(contributions[i])
            mag = abs(c_val)

            # Determine direction
            if predicted_class == CLASS_NO_SIGNIFICANT_STRESS:
                # Class 0: positive attribution pushes toward No Stress (decreases stress)
                if c_val > 0.002:
                    direction = "decreases_stress"
                elif c_val < -0.002:
                    direction = "increases_stress"
                else:
                    direction = "neutral"
            else:
                # Class 1 or 2: positive attribution pushes toward stress (increases stress)
                if c_val > 0.002:
                    direction = "increases_stress"
                elif c_val < -0.002:
                    direction = "decreases_stress"
                else:
                    direction = "neutral"

            exp_sentence = _generate_feature_sentence(
                feat_name, meta, raw_val, direction, c_val
            )

            local_contributions.append(
                LocalFeatureContribution(
                    feature=feat_name,
                    display_name=meta.get("display_name", feat_name),
                    modality=meta.get("modality", "unknown"),
                    value=round(raw_val, 4) if raw_val is not None else None,
                    unit=meta.get("unit", ""),
                    contribution=round(c_val, 5),
                    magnitude=round(mag, 5),
                    direction=direction,
                    explanation_text=exp_sentence,
                )
            )

        # Sort by absolute magnitude descending
        local_contributions.sort(key=lambda item: item.magnitude, reverse=True)

        # Retrieve global importance
        global_importance = self.get_global_importance()

        limitations = [
            f"Local explanation generated using {method_used}.",
            "Attributions reflect relative model decision weighting, not physical biological processes.",
            "Features with missing or unobserved satellite channels were imputed using median historical baselines.",
        ]

        return StressExplainability(
            method=method_used,
            predicted_class=predicted_class,
            stress_probability=stress_probability,
            local_feature_contributions=local_contributions,
            global_feature_importance=global_importance,
            explanation_limitations=limitations,
            non_causation_disclaimer=NON_CAUSATION_DISCLAIMER,
        )

    def _compute_shap_attributions(
        self,
        X_vec: np.ndarray,
        target_class_idx: int,
    ) -> Tuple[np.ndarray, str]:
        """
        Extracts SHAP values using TreeExplainer from tree ensembles within CalibratedClassifierCV.
        """
        # Case A: CalibratedClassifierCV wrapper
        if hasattr(self.model, "calibrated_classifiers_"):
            all_fold_shap = []
            for cc in self.model.calibrated_classifiers_:
                est = cc.estimator
                if hasattr(est, "named_steps"):
                    imputer = est.named_steps.get("imputer")
                    classifier = est.named_steps.get("classifier")
                    X_trans = imputer.transform(X_vec) if imputer else X_vec
                    explainer = shap.TreeExplainer(classifier)
                    sv = explainer.shap_values(X_trans)
                else:
                    explainer = shap.TreeExplainer(est)
                    sv = explainer.shap_values(X_vec)

                # Format SHAP values: (1, n_features, n_classes) or list of (1, n_features)
                if isinstance(sv, list):
                    target_idx = min(target_class_idx, len(sv) - 1)
                    vals = sv[target_idx][0]
                elif isinstance(sv, np.ndarray) and len(sv.shape) == 3:
                    target_idx = min(target_class_idx, sv.shape[2] - 1)
                    vals = sv[0, :, target_idx]
                else:
                    vals = np.array(sv).flatten()
                all_fold_shap.append(vals)

            avg_shap = np.mean(all_fold_shap, axis=0)
            return avg_shap, "SHAP_TreeExplainer"

        # Case B: Direct Pipeline
        elif hasattr(self.model, "named_steps"):
            imputer = self.model.named_steps.get("imputer")
            classifier = self.model.named_steps.get("classifier")
            X_trans = imputer.transform(X_vec) if imputer else X_vec
            explainer = shap.TreeExplainer(classifier)
            sv = explainer.shap_values(X_trans)
            if isinstance(sv, list):
                target_idx = min(target_class_idx, len(sv) - 1)
                vals = sv[target_idx][0]
            elif isinstance(sv, np.ndarray) and len(sv.shape) == 3:
                target_idx = min(target_class_idx, sv.shape[2] - 1)
                vals = sv[0, :, target_idx]
            else:
                vals = np.array(sv).flatten()
            return vals, "SHAP_TreeExplainer"

        # Case C: Standalone Tree Classifier
        else:
            explainer = shap.TreeExplainer(self.model)
            sv = explainer.shap_values(X_vec)
            if isinstance(sv, list):
                target_idx = min(target_class_idx, len(sv) - 1)
                vals = sv[target_idx][0]
            elif isinstance(sv, np.ndarray) and len(sv.shape) == 3:
                target_idx = min(target_class_idx, sv.shape[2] - 1)
                vals = sv[0, :, target_idx]
            else:
                vals = np.array(sv).flatten()
            return vals, "SHAP_TreeExplainer"

    def _compute_perturbation_attributions(
        self,
        X_vec: np.ndarray,
        target_class_idx: int,
    ) -> np.ndarray:
        """
        Computes marginal sensitivity feature attribution via controlled perturbation against neutral medians.
        """
        n_features = X_vec.shape[1]
        diffs = np.zeros(n_features)

        try:
            base_proba = self.model.predict_proba(X_vec)[0]
            base_score = float(base_proba[min(target_class_idx, len(base_proba) - 1)])
        except Exception:
            return diffs

        # Define reasonable neutral reference values for imputation/perturbation
        neutral_vector = np.zeros(n_features)
        for i, feat in enumerate(self.feature_names):
            if "ndvi" in feat or "evi" in feat:
                neutral_vector[i] = 0.55
            elif "ndmi" in feat:
                neutral_vector[i] = 0.25
            elif "ndre" in feat:
                neutral_vector[i] = 0.35
            elif "sar_vv" in feat:
                neutral_vector[i] = -12.0
            elif "sar_vh" in feat:
                neutral_vector[i] = -18.0
            elif "sar_vv_vh_ratio" in feat:
                neutral_vector[i] = 0.65
            elif "cloud" in feat or "nodata" in feat:
                neutral_vector[i] = 0.0
            elif "valid" in feat:
                neutral_vector[i] = 100.0
            else:
                neutral_vector[i] = 0.0

        for i in range(n_features):
            X_pert = X_vec.copy()
            # Replace feature with neutral baseline
            X_pert[0, i] = neutral_vector[i]
            try:
                pert_proba = self.model.predict_proba(X_pert)[0]
                pert_score = float(pert_proba[min(target_class_idx, len(pert_proba) - 1)])
                # Contribution = base score minus score without this feature
                diffs[i] = base_score - pert_score
            except Exception:
                diffs[i] = 0.0

        return diffs

    def get_global_importance(self) -> List[GlobalFeatureImportance]:
        """
        Retrieves global feature importance from model metadata or inspects tree ensemble.
        """
        importances: List[GlobalFeatureImportance] = []

        # 1. Check if metadata already recorded global feature importances
        meta_imp = self.metadata.get("global_feature_importance")
        if isinstance(meta_imp, list) and len(meta_imp) > 0:
            for rank, item in enumerate(meta_imp, start=1):
                f_name = item.get("feature") or item.get("name")
                imp_val = float(item.get("importance", 0.0))
                meta = FEATURE_METADATA.get(f_name, {
                    "display_name": f_name,
                    "modality": "unknown",
                })
                importances.append(
                    GlobalFeatureImportance(
                        feature=f_name,
                        display_name=meta.get("display_name", f_name),
                        modality=meta.get("modality", "unknown"),
                        importance=round(imp_val, 4),
                        rank=rank,
                    )
                )
            return importances

        # 2. Extract feature_importances_ from tree ensemble if available
        raw_imp = None
        if hasattr(self.model, "calibrated_classifiers_"):
            fold_imps = []
            for cc in self.model.calibrated_classifiers_:
                est = cc.estimator
                clf = est.named_steps.get("classifier") if hasattr(est, "named_steps") else est
                if hasattr(clf, "feature_importances_"):
                    fold_imps.append(clf.feature_importances_)
            if fold_imps:
                raw_imp = np.mean(fold_imps, axis=0)
        elif hasattr(self.model, "named_steps"):
            clf = self.model.named_steps.get("classifier")
            if hasattr(clf, "feature_importances_"):
                raw_imp = clf.feature_importances_
        elif hasattr(self.model, "feature_importances_"):
            raw_imp = self.model.feature_importances_

        if raw_imp is not None and len(raw_imp) == len(self.feature_names):
            sorted_indices = np.argsort(raw_imp)[::-1]
            for rank, idx in enumerate(sorted_indices, start=1):
                f_name = self.feature_names[idx]
                imp_val = float(raw_imp[idx])
                meta = FEATURE_METADATA.get(f_name, {
                    "display_name": f_name,
                    "modality": "unknown",
                })
                importances.append(
                    GlobalFeatureImportance(
                        feature=f_name,
                        display_name=meta.get("display_name", f_name),
                        modality=meta.get("modality", "unknown"),
                        importance=round(imp_val, 4),
                        rank=rank,
                    )
                )

        return importances
