"""
Crop Health AI — Leak-Free Spatial & Temporal Validation Engine (Phase 21.6)

Implements:
1. Grouped Field Holdout (prevents spatial leakage across parcels).
2. Expanding-Window Walk-Forward Validation (prevents future temporal leakage).
3. Leakage Auditor (validates strict isolation before model fitting).
"""

from typing import Dict, Any, List, Tuple, Generator
import datetime
import pandas as pd
import numpy as np


class DataLeakageError(ValueError):
    """Raised when spatial, temporal, or synthetic-to-real leakage is detected."""
    pass


class SpatialTemporalValidationSplitter:
    """Generates leak-free cross-validation folds and holdout sets."""

    @staticmethod
    def chronological_train_test_split(
        df: pd.DataFrame,
        test_fraction: float = 0.25,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset chronologically:
        Training on earlier observation dates, testing strictly on the latest dates.
        """
        if df.empty:
            return df, df

        df_sorted = df.sort_values(by="observation_date").reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1.0 - test_fraction))

        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()

        return train_df, test_df

    @staticmethod
    def grouped_field_split(
        df: pd.DataFrame,
        holdout_field_id: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Spatial holdout: trains on all other fields, evaluates on holdout field.
        """
        train_df = df[df["field_id"] != holdout_field_id].copy().reset_index(drop=True)
        test_df = df[df["field_id"] == holdout_field_id].copy().reset_index(drop=True)
        return train_df, test_df

    @staticmethod
    def expanding_walk_forward_splits(
        df: pd.DataFrame,
        min_train_size: int = 15,
        step_size: int = 5,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
        """
        Yields expanding chronological folds:
        Fold k: Train on [0 : T_k], Test on [T_k : T_k + step_size].
        """
        df_sorted = df.sort_values(by="observation_date").reset_index(drop=True)
        n = len(df_sorted)

        if n < (min_train_size + step_size):
            # Not enough data for walk-forward, yield single split if possible
            if n > min_train_size:
                yield df_sorted.iloc[:min_train_size], df_sorted.iloc[min_train_size:], 1
            return

        fold_idx = 1
        for split_point in range(min_train_size, n, step_size):
            train_sub = df_sorted.iloc[:split_point].copy()
            test_end = min(split_point + step_size, n)
            test_sub = df_sorted.iloc[split_point:test_end].copy()

            if len(test_sub) == 0:
                break

            yield train_sub, test_sub, fold_idx
            fold_idx += 1


class LeakageAuditor:
    """Verifies that a proposed train/test partition satisfies zero-leakage invariants."""

    @classmethod
    def audit_split(
        cls,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        is_spatial_holdout: bool = False,
        allow_synthetic: bool = False,
    ) -> Dict[str, Any]:
        """
        Audits train and test splits for all forms of leakage.
        Raises DataLeakageError if an invariant fails.
        """
        violations: List[str] = []

        # 1. Zero Intersection of Records
        if "product_identifier" in train_df and "product_identifier" in test_df:
            train_pids = set(train_df["product_identifier"].dropna())
            test_pids = set(test_df["product_identifier"].dropna())
            overlap = train_pids.intersection(test_pids)
            if overlap:
                violations.append(f"Record Leakage: {len(overlap)} product IDs exist in both train and test.")

        # 2. Spatial Field Isolation (if spatial holdout is claimed)
        if is_spatial_holdout:
            train_fids = set(train_df["field_id"].unique())
            test_fids = set(test_df["field_id"].unique())
            field_overlap = train_fids.intersection(test_fids)
            if field_overlap:
                violations.append(f"Spatial Field Leakage: Fields {field_overlap} exist in both train and test.")

        # 3. Temporal Causality (No test observation date prior to latest train date)
        if not is_spatial_holdout and not train_df.empty and not test_df.empty:
            max_train_date = train_df["observation_date"].max()
            min_test_date = test_df["observation_date"].min()
            if min_test_date < max_train_date:
                violations.append(
                    f"Temporal Future Leakage: Test min date ({min_test_date}) is earlier than train max date ({max_train_date})."
                )

        # 4. Zero Synthetic Contamination (Strictly blocked unless allow_synthetic explicitly enabled)
        if not allow_synthetic:
            for name, sub in [("train", train_df), ("test", test_df)]:
                if "is_demo_source" in sub:
                    synthetic_cnt = int(sub["is_demo_source"].sum())
                    if synthetic_cnt > 0:
                        violations.append(f"Synthetic Contamination: {synthetic_cnt} synthetic records present in {name} split.")

        is_clean = len(violations) == 0
        if not is_clean:
            raise DataLeakageError(" ; ".join(violations))

        return {
            "is_clean": True,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "train_fields": int(train_df["field_id"].nunique()) if not train_df.empty else 0,
            "test_fields": int(test_df["field_id"].nunique()) if not test_df.empty else 0,
            "train_date_span": (str(train_df["observation_date"].min()), str(train_df["observation_date"].max())) if not train_df.empty else None,
            "test_date_span": (str(test_df["observation_date"].min()), str(test_df["observation_date"].max())) if not test_df.empty else None,
        }
