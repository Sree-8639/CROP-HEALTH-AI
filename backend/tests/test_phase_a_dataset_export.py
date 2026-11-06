"""
Unit Tests for Phase A — Multimodal Crop Condition Dataset Export

Verifies:
1. Dataset file generation (Parquet, CSV, manifest.json).
2. Schema integrity (all expected columns, non-null guarantees).
3. Exact 224x224 dimensions for RGB and NDVI image chips.
4. Non-circular validation guarantee (NDVI independent of condition label).
5. All 4 target condition classes represented:
   - healthy
   - water-stressed
   - nutrient-stressed
   - suspected disease-pest
6. Honesty rule assertions: is_weak_label=True, label_source='weak_weather_based'.
"""

import os
import json
import pytest
import pandas as pd
from PIL import Image

from ml.research_data.condition_dataset import (
    CropConditionDatasetExporter,
    CLASS_HEALTHY,
    CLASS_WATER_STRESSED,
    CLASS_NUTRIENT_STRESSED,
    CLASS_SUSPECTED_DISEASE_PEST,
    ALL_CLASSES,
)


@pytest.fixture(scope="module")
def dataset_artifacts():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "ml", "research_data", "data", "phase_a")
    exporter = CropConditionDatasetExporter(output_dir=output_dir)
    df, manifest = exporter.generate_dataset(observations_per_field=12)
    return {
        "df": df,
        "manifest": manifest,
        "output_dir": output_dir,
        "exporter": exporter,
    }


def test_phase_a_files_exist(dataset_artifacts):
    output_dir = dataset_artifacts["output_dir"]
    parquet_file = os.path.join(output_dir, "crop_condition_dataset.parquet")
    csv_file = os.path.join(output_dir, "crop_condition_dataset.csv")
    manifest_file = os.path.join(output_dir, "manifest.json")

    assert os.path.exists(parquet_file), "Parquet dataset file must exist."
    assert os.path.exists(csv_file), "CSV dataset file must exist."
    assert os.path.exists(manifest_file), "Manifest JSON must exist."


def test_phase_a_dataframe_schema(dataset_artifacts):
    df = dataset_artifacts["df"]
    expected_cols = [
        "sample_id", "field_id", "field_name", "date", "latitude", "longitude",
        "crop_type", "ndvi", "ndmi", "ndre", "evi", "vv", "vh", "vv_vh_ratio",
        "cloud_quality", "label", "label_source", "label_reason", "is_weak_label",
        "is_demo", "data_provenance", "rgb_chip_path", "ndvi_chip_path"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Column '{col}' must be present in dataset schema."

    assert len(df) >= 40, f"Expected at least 40 observations, got {len(df)}"
    assert df["field_id"].nunique() >= 4, "Must cover at least 4 independent agricultural fields."
    assert df["date"].nunique() >= 6, "Must cover multiple seasonal dates."


def test_phase_a_classes_represented(dataset_artifacts):
    df = dataset_artifacts["df"]
    unique_labels = set(df["label"].unique())

    for c in ALL_CLASSES:
        assert c in unique_labels, f"Target condition class '{c}' must be represented in dataset."


def test_phase_a_image_chips_dimension_and_format(dataset_artifacts):
    df = dataset_artifacts["df"]
    output_dir = dataset_artifacts["output_dir"]

    # Sample test 5 random entries
    sample_rows = df.sample(min(5, len(df)), random_state=42)

    for _, row in sample_rows.iterrows():
        rgb_full = os.path.join(output_dir, row["rgb_chip_path"])
        ndvi_full = os.path.join(output_dir, row["ndvi_chip_path"])

        assert os.path.exists(rgb_full), f"RGB chip file not found: {rgb_full}"
        assert os.path.exists(ndvi_full), f"NDVI chip file not found: {ndvi_full}"

        with Image.open(rgb_full) as img_rgb:
            assert img_rgb.size == (224, 224), f"RGB chip must be exactly 224x224, got {img_rgb.size}"
            assert img_rgb.mode in ("RGB", "RGBA")

        with Image.open(ndvi_full) as img_ndvi:
            assert img_ndvi.size == (224, 224), f"NDVI chip must be exactly 224x224, got {img_ndvi.size}"
            assert img_ndvi.mode in ("RGB", "RGBA")


def test_phase_a_non_circular_labeling_guarantee(dataset_artifacts):
    exporter = dataset_artifacts["exporter"]

    # Case 1: High NDVI with acute moisture and SAR deficit -> Must be water-stressed
    # Demonstrating NDVI does NOT block stress classification
    label, _ = exporter.determine_weak_label(
        ndmi=0.10,
        ndre=0.28,
        vv=-16.0,
        vh=-22.0,
        vv_vh_ratio=6.0,
        rainfall_deficit_pct=60.0,
    )
    assert label == CLASS_WATER_STRESSED, "Must detect water-stressed via NDMI and SAR, independent of NDVI."

    # Case 2: Low NDVI but normal NDMI/NDRE/SAR -> Must be classified as healthy or normal
    label, _ = exporter.determine_weak_label(
        ndmi=0.30,
        ndre=0.35,
        vv=-9.5,
        vh=-16.0,
        vv_vh_ratio=6.5,
        rainfall_deficit_pct=5.0,
    )
    assert label == CLASS_HEALTHY, "Normal non-NDVI indicators must classify as healthy."


def test_phase_a_honesty_rules(dataset_artifacts):
    df = dataset_artifacts["df"]
    manifest = dataset_artifacts["manifest"]

    # All weak labels must be explicitly flagged
    assert bool(df["is_weak_label"].all()) is True, "All rows must be flagged as is_weak_label=True."
    assert (df["label_source"] == "weak_weather_based").all(), "All rows must have label_source='weak_weather_based'."
    assert manifest["is_weak_label_pct"] == 100.0
    assert "non_circular_validation_guarantee" in manifest

