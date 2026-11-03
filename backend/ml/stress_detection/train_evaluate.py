"""
Model training and evaluation pipeline for AI early crop stress detection.
Trains baseline Random Forest / Gradient Boosting models with class balancing,
imputation, probability calibration, and rigorous multi-class evaluation metrics.
"""

import os
import json
import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.preprocessing import label_binarize

from ml.stress_detection.schema import (
    ALL_MODEL_FEATURE_NAMES,
    STRESS_CLASSES,
    CLASS_NO_SIGNIFICANT_STRESS,
    CLASS_POTENTIAL_STRESS,
    CLASS_ELEVATED_STRESS,
    REVERSE_LABEL_MAPPING,
)
from ml.stress_detection.data_pipeline import (
    generate_synthetic_demo_dataset,
    spatial_temporal_split,
)

def evaluate_stress_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes rigorous multi-class evaluation metrics:
    - Precision, Recall, F1 (macro and per-class)
    - ROC-AUC (One-vs-Rest, when statistically viable)
    - PR-AUC (Average Precision, suited for class imbalance)
    - Confusion Matrix
    - Multi-class Brier score for probability calibration
    """
    target_names = target_names or STRESS_CLASSES
    n_classes = len(target_names)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Overall metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    per_class_metrics: Dict[str, Any] = {}
    for idx, name in enumerate(target_names):
        per_class_metrics[name] = {
            "precision": float(prec_per_class[idx]) if idx < len(prec_per_class) else 0.0,
            "recall": float(rec_per_class[idx]) if idx < len(rec_per_class) else 0.0,
            "f1": float(f1_per_class[idx]) if idx < len(f1_per_class) else 0.0,
            "support": int(np.count_nonzero(y_test == idx)),
        }

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes))).tolist()

    # ROC-AUC & PR-AUC
    roc_auc_val: Optional[float] = None
    pr_auc_val: Optional[float] = None
    brier_val: Optional[float] = None

    if y_prob is not None:
        try:
            y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
            if y_test_bin.shape[1] == 1:
                # Binary fallback
                y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
            
            roc_auc_val = float(roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro"))
            pr_auc_val = float(average_precision_score(y_test_bin, y_prob, average="macro"))

            # Multi-class Brier score: mean squared error of predicted probabilities vs one-hot
            brier_val = float(np.mean(np.sum((y_prob - y_test_bin) ** 2, axis=1)))
        except Exception:
            pass

    return {
        "sample_count": int(len(y_test)),
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "roc_auc_ovr": round(roc_auc_val, 4) if roc_auc_val is not None else None,
        "pr_auc_macro": round(pr_auc_val, 4) if pr_auc_val is not None else None,
        "brier_score_calibration": round(brier_val, 4) if brier_val is not None else None,
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
    }

def train_stress_model_pipeline(
    train_records: List[Dict[str, Any]],
    val_records: Optional[List[Dict[str, Any]]] = None,
    test_records: Optional[List[Dict[str, Any]]] = None,
    feature_names: Optional[List[str]] = None,
    is_demo_training: bool = False,
    model_version: str = "v1.0.0",
    dataset_provenance: str = "Unspecified",
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains a balanced Random Forest model pipeline with median imputation
    and probability calibration.
    """
    feature_names = feature_names or ALL_MODEL_FEATURE_NAMES

    # Prepare training array
    X_train = np.array([[r.get(f, np.nan) for f in feature_names] for r in train_records], dtype=np.float32)
    y_train = np.array([int(r["label"]) for r in train_records], dtype=np.int64)

    # Base pipeline: SimpleImputer (median) + RandomForestClassifier (balanced)
    base_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    imputer = SimpleImputer(strategy="median")
    pipeline = Pipeline([
        ("imputer", imputer),
        ("classifier", base_rf),
    ])

    pipeline.fit(X_train, y_train)

    # Calibration using Sigmoid / Platt scaling via 3-fold CV if sufficient samples
    calibrated_pipeline: Any = pipeline
    is_calibrated = False

    if len(train_records) >= 60:
        try:
            calibrated_clf = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=3)
            calibrated_clf.fit(X_train, y_train)
            calibrated_pipeline = calibrated_clf
            is_calibrated = True
        except Exception:
            calibrated_pipeline = pipeline

    # Evaluation on holdout test set if provided
    eval_metrics: Optional[Dict[str, Any]] = None
    if test_records:
        X_test = np.array([[r.get(f, np.nan) for f in feature_names] for r in test_records], dtype=np.float32)
        y_test = np.array([int(r["label"]) for r in test_records], dtype=np.int64)
        eval_metrics = evaluate_stress_model(calibrated_pipeline, X_test, y_test)

    # Feature importances from base Random Forest
    importances: List[Dict[str, float]] = []
    try:
        raw_rf = pipeline.named_steps["classifier"]
        feat_imp = raw_rf.feature_importances_
        sorted_idx = np.argsort(feat_imp)[::-1]
        for idx in sorted_idx:
            importances.append({
                "feature": feature_names[idx],
                "importance": round(float(feat_imp[idx]), 4),
            })
    except Exception:
        pass

    metadata: Dict[str, Any] = {
        "model_name": "MultimodalCropStressDetector",
        "model_version": model_version,
        "model_type": "RandomForest_Calibrated" if is_calibrated else "RandomForest_Balanced",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "is_demo_model": is_demo_training,
        "is_calibrated": is_calibrated,
        "feature_count": len(feature_names),
        "features": feature_names,
        "target_classes": STRESS_CLASSES,
        "dataset_provenance": dataset_provenance,
        "training_sample_count": len(train_records),
        "test_sample_count": len(test_records) if test_records else 0,
        "evaluation_metrics": eval_metrics,
        "global_feature_importance": importances,
        "scientific_disclaimer": (
            "Early stress detection outputs indicate physical radiometric and active radar backscatter "
            "anomalies. They do not constitute diagnostic proof of biological pathogens, soil nitrogen deficiency, "
            "or specific causal agents without local agronomist ground verification."
        ),
    }

    return calibrated_pipeline, metadata

def train_and_save_demo_baseline(
    output_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Trains and saves the verified demo baseline model artifact in `backend/ml/models/`.
    Explicitly tags the artifact as a demo demonstrator so synthetic metrics are never
    mixed with real scientific benchmarks.
    """
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "models")

    os.makedirs(output_dir, exist_ok=True)

    # Generate synthetic demo dataset
    demo_records = generate_synthetic_demo_dataset(n_samples=300, random_state=42)
    train_recs, val_recs, test_recs = spatial_temporal_split(demo_records, random_state=42)

    model_pipeline, metadata = train_stress_model_pipeline(
        train_records=train_recs,
        val_records=val_recs,
        test_records=test_recs,
        is_demo_training=True,
        model_version="v1.0.0-demo",
        dataset_provenance="Synthetic agricultural demonstrator (agronomic physics curves)",
    )

    model_path = os.path.join(output_dir, "demo_stress_model.joblib")
    meta_path = os.path.join(output_dir, "demo_stress_model_metadata.json")

    joblib.dump(model_pipeline, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_path, meta_path

if __name__ == "__main__":
    m_path, meta_path = train_and_save_demo_baseline()
    print(f"Successfully generated baseline demo model: {m_path}")
    print(f"Metadata saved: {meta_path}")
