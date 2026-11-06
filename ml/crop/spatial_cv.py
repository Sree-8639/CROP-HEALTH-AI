"""
ml/crop/spatial_cv.py:
Spatial Block Cross-Validation Engine.

Partitions fields into non-overlapping spatial blocks to strictly evaluate
out-of-region generalization and prevent spatial autocorrelation leakage.
Computes Accuracy, Macro F1, Weighted F1, and Multi-class ROC-AUC with 95% Confidence Intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_spatial_block_cv(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    spatial_groups: pd.Series,
    n_splits: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes Spatial Block Cross-Validation using GroupKFold.
    
    Args:
        model_factory: Callable returning a fresh estimator instance
        X: Feature matrix
        y: Target label vector
        spatial_groups: Group identifiers (e.g. spatial block ID or cluster)
        n_splits: Number of spatial folds
    """
    gkf = GroupKFold(n_splits=n_splits)

    fold_accuracies = []
    fold_f1_macros = []
    fold_f1_weighteds = []
    fold_roc_aucs = []
    fold_summaries = []
    all_y_true = []
    all_y_pred = []

    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=spatial_groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_block = spatial_groups.iloc[test_idx].iloc[0]

        # Fit fresh model instance
        clf = model_factory()
        clf.fit(X_train, y_train)

        # Predictions
        y_pred = clf.predict(X_test)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        acc = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_wt = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # ROC-AUC calculation
        roc_auc_val = None
        if hasattr(clf, "predict_proba"):
            try:
                y_prob = clf.predict_proba(X_test)
                if n_classes == 2:
                    roc_auc_val = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    roc_auc_val = roc_auc_score(
                        y_test,
                        y_prob,
                        multi_class="ovr",
                        average="macro",
                    )
            except Exception:
                roc_auc_val = None

        if roc_auc_val is not None:
            fold_roc_aucs.append(roc_auc_val)

        fold_accuracies.append(acc)
        fold_f1_macros.append(f1_mac)
        fold_f1_weighteds.append(f1_wt)

        fold_summaries.append({
            "fold": fold_idx + 1,
            "held_out_spatial_block": str(test_block),
            "test_sample_count": len(test_idx),
            "accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1_mac), 4),
            "f1_weighted": round(float(f1_wt), 4),
            "roc_auc": round(float(roc_auc_val), 4) if roc_auc_val is not None else None,
        })

    # Statistical aggregation & 95% Confidence Intervals
    def _calc_stats_and_ci(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "ci_95_low": 0.0, "ci_95_high": 0.0}
        arr = np.array(values)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        se = std_val / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        ci_half = 1.96 * se
        return {
            "mean": float(round(mean_val, 4)),
            "std": float(round(std_val, 4)),
            "ci_95_low": float(round(max(0.0, mean_val - ci_half), 4)),
            "ci_95_high": float(round(min(1.0, mean_val + ci_half), 4)),
        }

    overall_cm = confusion_matrix(all_y_true, all_y_pred).tolist() if all_y_true else []

    return {
        "spatial_folds_evaluated": n_splits,
        "metrics": {
            "accuracy": _calc_stats_and_ci(fold_accuracies),
            "f1_macro": _calc_stats_and_ci(fold_f1_macros),
            "f1_weighted": _calc_stats_and_ci(fold_f1_weighteds),
            "roc_auc": _calc_stats_and_ci(fold_roc_aucs) if fold_roc_aucs else None,
        },
        "confusion_matrix": overall_cm,
        "fold_details": fold_summaries,
    }

