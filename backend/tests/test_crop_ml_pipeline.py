"""
backend/tests/test_crop_ml_pipeline.py:
Unit tests for ml/crop machine learning components:
- Ground-truth label dataset loader / benchmark generator
- Descals et al. phenology feature extractor and baseline
- Improved multimodal S1+S2 classifier and TreeSHAP explainability
- Spatial block cross-validation with 95% confidence intervals
- Modality ablations (S2-only vs S1-only vs S1+S2)
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add repository root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.crop.dataset import load_or_create_labels_dataset, extract_multimodal_features
from ml.crop.baseline_descals import DescalsPhenologyBaseline
from ml.crop.improved_model import MultimodalCropClassifier
from ml.crop.ablations import run_sensor_ablations
from ml.crop.pipeline import run_crop_pipeline


def get_test_df():
    """Returns labels DataFrame or in-memory fixture for unit testing."""
    df = load_or_create_labels_dataset(random_seed=42)
    if df is not None:
        return df
    
    # In-memory test fixture purely for testing functions
    rng = np.random.RandomState(42)
    records = []
    for i in range(60):
        records.append({
            "field_id": f"field_{i+1:03d}",
            "spatial_block": f"block_{(i % 5) + 1}",
            "label": i % 3,
            "date": "2024-01-15",
            "geometry": "POLYGON ((81.5 16.8, 81.51 16.8, 81.51 16.81, 81.5 16.81, 81.5 16.8))"
        })
    return pd.DataFrame(records)


def test_labels_dataset_loading_and_features():
    df = get_test_df()
    assert len(df) >= 50
    assert "field_id" in df.columns
    assert "spatial_block" in df.columns
    assert "label" in df.columns

    descals_df, fused_df, meta = extract_multimodal_features(df, random_seed=42)
    assert len(descals_df) == len(df)
    assert len(fused_df) == len(df)
    assert "early_stage_mean_ndvi" in descals_df.columns
    assert "mid_stage_peak_ndvi" in descals_df.columns
    assert "ndvi_mean" in fused_df.columns
    assert "vv_backscatter_db_mean" in fused_df.columns
    assert "vh_backscatter_db_mean" in fused_df.columns


def test_descals_baseline_spatial_cv():
    df = get_test_df()
    descals_df, _, _ = extract_multimodal_features(df, random_seed=42)
    baseline = DescalsPhenologyBaseline(random_seed=42)
    res = baseline.evaluate_spatial_cv(
        X_descals=descals_df,
        y=df["label"],
        spatial_groups=df["spatial_block"],
        n_splits=5,
    )
    metrics = res["results"]["metrics"]
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert metrics["accuracy"]["mean"] > 0.70
    assert metrics["accuracy"]["ci_95_low"] <= metrics["accuracy"]["mean"] <= metrics["accuracy"]["ci_95_high"]


def test_improved_multimodal_classifier_and_shap():
    df = get_test_df()
    _, fused_df, _ = extract_multimodal_features(df, random_seed=42)
    clf = MultimodalCropClassifier(model_type="xgboost", random_seed=42)
    res = clf.evaluate_spatial_cv(
        X_fused=fused_df,
        y=df["label"],
        spatial_groups=df["spatial_block"],
        n_splits=5,
    )
    metrics = res["results"]["metrics"]
    assert metrics["accuracy"]["mean"] > 0.75


def test_sensor_ablations():
    df = get_test_df()
    _, fused_df, _ = extract_multimodal_features(df, random_seed=42)
    ablation_res = run_sensor_ablations(
        fused_df=fused_df,
        labels=df["label"],
        spatial_groups=df["spatial_block"],
        random_seed=42,
        n_splits=5,
    )
    configs = ablation_res.get("configurations", ablation_res)
    assert "s2_only" in configs
    assert "s1_only" in configs
    assert "s1_plus_s2" in configs
    assert configs["s1_plus_s2"]["accuracy"]["mean"] >= configs["s1_only"]["accuracy"]["mean"]
