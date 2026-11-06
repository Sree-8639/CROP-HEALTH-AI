"""
ml/crop/ablations.py:
Systematic Sensor Input Ablation Study for Crop Stress Classification:
Evaluates:
1. S2-Only (Sentinel-2 Optical Indices: NDVI, NDMI, NDRE, EVI)
2. S1-Only (Sentinel-1 C-SAR Backscatter: VV, VH, VV-VH dB difference, linear ratio)
3. S1+S2 Multimodal (Fused Optical + SAR feature space)

Evaluated under identical Spatial Block Cross-Validation splits.
"""

from typing import Dict, Any, List
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.crop.spatial_cv import evaluate_spatial_block_cv


def run_sensor_ablations(
    fused_df: pd.DataFrame,
    labels: pd.Series,
    spatial_groups: pd.Series,
    random_seed: int = 42,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    Executes systematic sensor ablations:
    - S2 only vs S1 only vs S1+S2
    """
    # Feature subsets
    s2_cols = [c for c in fused_df.columns if not any(k in c for k in ["vv", "vh", "sar"])]
    s1_cols = [c for c in fused_df.columns if any(k in c for k in ["vv", "vh", "sar"]) and not "optical" in c]
    fused_cols = list(fused_df.columns)

    configurations = {
        "s2_only": {
            "name": "Sentinel-2 Optical Only",
            "sensor": "Sentinel-2 MSI Level-2A",
            "features": s2_cols,
            "feature_df": fused_df[s2_cols],
            "description": "Multispectral vegetation indices without cloud-penetrating radar.",
        },
        "s1_only": {
            "name": "Sentinel-1 SAR Only",
            "sensor": "Sentinel-1 C-SAR GRD",
            "features": s1_cols,
            "feature_df": fused_df[s1_cols],
            "description": "All-weather microwave backscatter without optical spectral detail.",
        },
        "s1_plus_s2": {
            "name": "Sentinel-1 + Sentinel-2 Fused",
            "sensor": "Sentinel-1 SAR + Sentinel-2 Optical (Multimodal Fusion)",
            "features": fused_cols,
            "feature_df": fused_df[fused_cols],
            "description": "Unified multimodal feature stack combining spectral response and canopy structure.",
        },
    }

    ablation_results = {}

    for config_key, cfg in configurations.items():
        def _make_estimator():
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=2,
                random_state=random_seed,
                class_weight="balanced",
            )

        cv_eval = evaluate_spatial_block_cv(
            model_factory=_make_estimator,
            X=cfg["feature_df"],
            y=labels,
            spatial_groups=spatial_groups,
            n_splits=n_splits,
            random_seed=random_seed,
        )

        ablation_results[config_key] = {
            "configuration_name": cfg["name"],
            "sensor": cfg["sensor"],
            "feature_count": len(cfg["features"]),
            "features_used": cfg["features"],
            "description": cfg["description"],
            "accuracy": cv_eval["metrics"]["accuracy"],
            "f1_macro": cv_eval["metrics"]["f1_macro"],
            "f1_weighted": cv_eval["metrics"]["f1_weighted"],
            "roc_auc": cv_eval["metrics"]["roc_auc"],
        }

    # Calculate multimodal performance gains over single-sensor baselines
    s1_acc = ablation_results["s1_only"]["accuracy"]["mean"]
    s2_acc = ablation_results["s2_only"]["accuracy"]["mean"]
    fused_acc = ablation_results["s1_plus_s2"]["accuracy"]["mean"]

    s1_f1 = ablation_results["s1_only"]["f1_macro"]["mean"]
    s2_f1 = ablation_results["s2_only"]["f1_macro"]["mean"]
    fused_f1 = ablation_results["s1_plus_s2"]["f1_macro"]["mean"]

    multimodal_gain = {
        "accuracy_gain_over_s2_pct": round((fused_acc - s2_acc) * 100.0, 2),
        "accuracy_gain_over_s1_pct": round((fused_acc - s1_acc) * 100.0, 2),
        "f1_macro_gain_over_s2_pct": round((fused_f1 - s2_f1) * 100.0, 2),
        "f1_macro_gain_over_s1_pct": round((fused_f1 - s1_f1) * 100.0, 2),
    }

    return {
        "ablation_study_name": "Sensor Modality Ablation (S2-only vs S1-only vs S1+S2)",
        "cross_validation": f"{n_splits}-Fold Spatial Block GroupKFold",
        "configurations": ablation_results,
        "multimodal_gain_summary": multimodal_gain,
    }

# Backwards compatibility alias
run_ablations = run_sensor_ablations
