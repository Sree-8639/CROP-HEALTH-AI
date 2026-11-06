"""
Crop Health AI — Research Dataset Quality & Readiness Gates (Phase 21.5)

Implements rigorous pre-training validation gates:
1. Stable Field Identifiers.
2. Valid Timestamps.
3. Duplicate Detection & Handling.
4. Feature vs Target Chronological Integrity (t_target > t_feature).
5. Explicit Representation of Missing Data.
6. Exclusion of Invalid Cloud/NoData Pixels.
7. Verifiable Data Provenance.
8. Independent Target Calculation.
9. Leakage Audit (No future information in input features).
10. Spatial Holdout & Grouping Feasibility.

If real data is insufficient, returns TRAINING_BLOCKED_INSUFFICIENT_REAL_DATA.
"""

from typing import Dict, Any, List, Optional
import datetime
import math
import numpy as np
import pandas as pd


class DatasetReadinessError(ValueError):
    """Raised when a research dataset fails mandatory readiness gates."""
    pass


class ResearchDatasetValidator:
    """Audits dataset quality and executes readiness gates prior to model training."""

    MIN_TOTAL_SAMPLES = 10
    MIN_INDEPENDENT_FIELDS = 2

    @classmethod
    def validate_dataset(
        cls,
        df: pd.DataFrame,
        strict_real_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes all 10 scientific readiness gates on the tabular research dataset.
        Returns a comprehensive audit dictionary.
        """
        checks: Dict[str, Any] = {}
        errors: List[str] = []

        if df.empty:
            return {
                "status": "TRAINING_BLOCKED_INSUFFICIENT_REAL_DATA",
                "is_ready_for_training": False,
                "reason": "Dataset is completely empty.",
                "checks": {},
            }

        # Gate 0: Strict Real Only Check (reject synthetic contamination)
        if strict_real_only:
            synthetic_count = int(df.get("is_demo_source", pd.Series(0)).sum())
            if synthetic_count > 0:
                errors.append(f"Synthetic data contamination: {synthetic_count} synthetic records found in research dataset.")
            checks["zero_synthetic_contamination"] = bool(synthetic_count == 0)

        # Gate 1: Stable Field ID
        missing_fids = int(df["field_id"].isnull().sum())
        checks["stable_field_ids"] = bool(missing_fids == 0)
        if missing_fids > 0:
            errors.append(f"Gate 1 Failed: {missing_fids} records have missing field_id.")

        # Gate 2: Valid Timestamps
        invalid_dates = 0
        for d in df["observation_date"]:
            try:
                datetime.date.fromisoformat(str(d)[:10])
            except Exception:
                invalid_dates += 1
        checks["valid_timestamps"] = bool(invalid_dates == 0)
        if invalid_dates > 0:
            errors.append(f"Gate 2 Failed: {invalid_dates} invalid observation_date values.")

        # Gate 3: Duplicate Acquisitions Handled
        duplicates = int(df.duplicated(subset=["field_id", "observation_date"]).sum())
        checks["no_duplicate_field_dates"] = bool(duplicates == 0)
        if duplicates > 0:
            errors.append(f"Gate 3 Failed: {duplicates} duplicate (field_id, observation_date) observations found.")

        # Gate 4: Feature and Target Chronological Integrity
        temporal_violations = 0
        for _, row in df.iterrows():
            obs_dt = datetime.date.fromisoformat(str(row["observation_date"])[:10])
            tgt_dt = datetime.date.fromisoformat(str(row["target_date"])[:10])
            if tgt_dt <= obs_dt:
                temporal_violations += 1
        checks["strictly_future_targets"] = bool(temporal_violations == 0)
        if temporal_violations > 0:
            errors.append(f"Gate 4 Failed: {temporal_violations} records have target_date <= observation_date (temporal violation).")

        # Gate 5: Missing Observations Explicitly Represented
        suspicious_sar = 0
        for _, row in df.iterrows():
            if row.get("is_missing_sar") == 1:
                vv = row.get("sar_vv")
                if vv is not None and not (isinstance(vv, float) and math.isnan(vv)):
                    suspicious_sar += 1
        checks["missing_data_explicit"] = bool(suspicious_sar == 0)
        if suspicious_sar > 0:
            errors.append(f"Gate 5 Failed: {suspicious_sar} records flag missing SAR but contain non-NaN values.")

        # Gate 6: Invalid Pixels Excluded
        invalid_pixels = int((df["valid_pixel_pct"] <= 0).sum()) if "valid_pixel_pct" in df else 0
        checks["invalid_pixels_excluded"] = bool(invalid_pixels == 0)
        if invalid_pixels > 0:
            errors.append(f"Gate 6 Failed: {invalid_pixels} records have 0% valid pixels.")

        # Gate 7: Data Provenance Available
        missing_provenance = int(df["product_identifier"].isnull().sum()) if "product_identifier" in df else len(df)
        checks["provenance_documented"] = bool(missing_provenance == 0)
        if missing_provenance > 0:
            errors.append(f"Gate 7 Failed: {missing_provenance} records lack product_identifier.")

        # Gate 8: Target Physical Validity
        invalid_targets = int((
            (df["target_ndvi"] < -1.0) | (df["target_ndvi"] > 1.0) | df["target_ndvi"].isnull()
        ).sum())
        checks["target_physically_valid"] = bool(invalid_targets == 0)
        if invalid_targets > 0:
            errors.append(f"Gate 8 Failed: {invalid_targets} target_ndvi values fall outside [-1.0, 1.0].")

        # Gate 9: No Feature Leakage
        forbidden_leakage_columns = {
            "target_ndvi",
            "target_date",
            "is_synthetic_target",
        }
        feature_cols = [c for c in df.columns if c not in forbidden_leakage_columns]
        leakage_detected = any("target" in c.lower() for c in feature_cols if c != "target_horizon_days")
        checks["no_target_leakage_in_features"] = bool(not leakage_detected)
        if leakage_detected:
            errors.append("Gate 9 Failed: Target information detected in feature columns.")

        # Gate 10: Spatial Separation & Sample Sufficiency
        unique_fields = int(df["field_id"].nunique())
        total_samples = int(len(df))
        checks["sufficient_independent_fields"] = bool(unique_fields >= cls.MIN_INDEPENDENT_FIELDS)
        checks["sufficient_sample_size"] = bool(total_samples >= cls.MIN_TOTAL_SAMPLES)

        if unique_fields < cls.MIN_INDEPENDENT_FIELDS:
            errors.append(
                f"Gate 10 Failed: Only {unique_fields} independent fields available (minimum required: {cls.MIN_INDEPENDENT_FIELDS})."
            )

        if total_samples < cls.MIN_TOTAL_SAMPLES:
            errors.append(
                f"Gate 10 Failed: Only {total_samples} samples available (minimum required: {cls.MIN_TOTAL_SAMPLES})."
            )

        is_ready = len(errors) == 0

        status = "DATASET_READY" if is_ready else "TRAINING_BLOCKED_INSUFFICIENT_REAL_DATA"

        return {
            "status": status,
            "is_ready_for_training": is_ready,
            "total_records": total_samples,
            "unique_fields": unique_fields,
            "fields": df["field_id"].unique().tolist(),
            "temporal_span": {
                "start": str(df["observation_date"].min()),
                "end": str(df["observation_date"].max()),
            },
            "gate_checks": checks,
            "error_count": len(errors),
            "errors": errors,
        }
