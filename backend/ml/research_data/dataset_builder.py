"""
Crop Health AI — Research Dataset Builder (Phase 21.4)

Compiles reproducible, leak-free supervised tabular datasets for satellite
time-series vegetation index forecasting.
Extracts optical bands, SAR backscatter, temporal lag metrics, and future target observations.
Exports to Parquet and CSV formats.
"""

from typing import Dict, Any, List, Optional
import os
import json
import datetime
import math
import numpy as np
import pandas as pd

from ml.research_data.acquisition.collector import ResearchObservationCollector
from ml.research_data.ground_truth.workflow import TargetHorizonGenerator


class ResearchDatasetBuilder:
    """Builds tabular machine learning datasets for satellite time-series forecasting."""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        collector: Optional[ResearchObservationCollector] = None,
        target_generator: Optional[TargetHorizonGenerator] = None,
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = output_dir or os.path.join(base_dir, "data")
        os.makedirs(self.output_dir, exist_ok=True)
        self.collector = collector or ResearchObservationCollector()
        self.target_generator = target_generator or TargetHorizonGenerator(target_index="ndvi")

    def build_features_from_observation_sequence(
        self,
        observations: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Takes raw chronological observations for one or more fields and creates
        a tabular feature matrix with strict temporal lag engineering.
        """
        if not observations:
            return pd.DataFrame()

        # Group observations by field_id
        fields: Dict[str, List[Dict[str, Any]]] = {}
        for obs in observations:
            fid = obs["field_id"]
            if fid not in fields:
                fields[fid] = []
            fields[fid].append(obs)

        all_rows: List[Dict[str, Any]] = []

        for fid, field_obs in fields.items():
            # Sort strictly chronologically
            sorted_obs = sorted(field_obs, key=lambda x: x["acquisition_date"])

            # Generate target pairs
            pairs = self.target_generator.generate_supervised_pairs(sorted_obs)

            # Map from observation product_id or date to target info
            target_map = {
                p["base_observation"]["product_identifier"]: p for p in pairs
            }

            for idx, obs in enumerate(sorted_obs):
                pid = obs["product_identifier"]
                target_pair = target_map.get(pid)

                # Skip if no valid future observation exists within the target horizon
                if target_pair is None:
                    continue

                curr_date = datetime.date.fromisoformat(obs["acquisition_date"][:10])

                # Lag features from previous observation (idx - 1)
                prev_obs = sorted_obs[idx - 1] if idx > 0 else None
                prev_date = (
                    datetime.date.fromisoformat(prev_obs["acquisition_date"][:10])
                    if prev_obs
                    else None
                )

                interval_days = (curr_date - prev_date).days if prev_date else np.nan

                # Compute temporal deltas
                prev_ndvi = prev_obs.get("ndvi") if prev_obs else np.nan
                curr_ndvi = obs.get("ndvi", np.nan)
                delta_ndvi = (
                    (curr_ndvi - prev_ndvi)
                    if (prev_ndvi is not None and not math.isnan(prev_ndvi) and curr_ndvi is not None)
                    else np.nan
                )
                rate_ndvi_per_day = (
                    (delta_ndvi / interval_days)
                    if (interval_days and interval_days > 0 and not math.isnan(delta_ndvi))
                    else np.nan
                )

                prev_ndmi = prev_obs.get("ndmi") if prev_obs else np.nan
                curr_ndmi = obs.get("ndmi", np.nan)
                delta_ndmi = (
                    (curr_ndmi - prev_ndmi)
                    if (prev_ndmi is not None and not math.isnan(prev_ndmi) and curr_ndmi is not None)
                    else np.nan
                )

                row = {
                    "field_id": fid,
                    "observation_date": obs["acquisition_date"][:10],
                    "crop_type": obs.get("crop_type", "Paddy / Rice"),
                    "latitude": obs["aoi_geometry"]["latitude"],
                    "longitude": obs["aoi_geometry"]["longitude"],
                    # Core optical spectral features
                    "ndvi": curr_ndvi,
                    "ndmi": curr_ndmi,
                    "ndre": obs.get("ndre", np.nan),
                    "evi": obs.get("evi", np.nan),
                    "b02": obs.get("b02", np.nan),
                    "b03": obs.get("b03", np.nan),
                    "b04": obs.get("b04", np.nan),
                    "b05": obs.get("b05", np.nan),
                    "b08": obs.get("b08", np.nan),
                    "b11": obs.get("b11", np.nan),
                    # SAR features
                    "sar_vv": obs.get("sar_vv", np.nan),
                    "sar_vh": obs.get("sar_vh", np.nan),
                    "sar_vv_vh_ratio": obs.get("sar_vv_vh_ratio", np.nan),
                    # Temporal features
                    "prev_ndvi": prev_ndvi,
                    "prev_ndmi": prev_ndmi,
                    "delta_ndvi": delta_ndvi,
                    "delta_ndmi": delta_ndmi,
                    "rate_ndvi_per_day": rate_ndvi_per_day,
                    "observation_interval_days": interval_days,
                    "history_observation_count": idx + 1,
                    # Quality features
                    "cloud_coverage": obs.get("cloud_coverage", 0.0),
                    "valid_pixel_pct": obs.get("valid_pixel_percentage", 100.0),
                    "is_missing_sar": int(obs.get("is_missing_sar", True)),
                    "is_missing_optical": int(obs.get("is_missing_optical", False)),
                    "is_demo_source": int(obs.get("is_demo", False)),
                    "product_identifier": pid,
                    # Target information
                    "target_date": target_pair["target_date"],
                    "target_horizon_days": target_pair["target_horizon_days"],
                    "target_ndvi": target_pair["target_value"],
                    "is_synthetic_target": int(target_pair["is_synthetic_target"]),
                }
                all_rows.append(row)

        df = pd.DataFrame(all_rows)
        return df

    def save_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str = "research_timeseries_dataset",
        version: str = "v1.0.0",
    ) -> Dict[str, str]:
        """Saves dataframe as Parquet and CSV artifacts."""
        parquet_path = os.path.join(self.output_dir, f"{dataset_name}_{version}.parquet")
        csv_path = os.path.join(self.output_dir, f"{dataset_name}_{version}.csv")

        # Save Parquet
        df.to_parquet(parquet_path, index=False)
        # Save CSV for human inspection
        df.to_csv(csv_path, index=False)

        return {
            "parquet_path": parquet_path,
            "csv_path": csv_path,
        }

    def generate_manifest(
        self,
        df: pd.DataFrame,
        dataset_version: str = "v1.0.0",
        dataset_name: str = "Sentinel_Multimodal_NDVI_Forecast_Dataset",
        notes: str = "Research dataset for satellite vegetation-index time-series forecasting.",
    ) -> Dict[str, Any]:
        """Generates comprehensive dataset manifest."""
        manifest = {
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "notes": notes,
            "total_records": len(df),
            "unique_fields_count": int(df["field_id"].nunique()) if not df.empty else 0,
            "fields": df["field_id"].unique().tolist() if not df.empty else [],
            "temporal_range": {
                "start_date": str(df["observation_date"].min()) if not df.empty else None,
                "end_date": str(df["observation_date"].max()) if not df.empty else None,
            },
            "synthetic_records_count": int(df["is_demo_source"].sum()) if not df.empty else 0,
            "real_records_count": int((df["is_demo_source"] == 0).sum()) if not df.empty else 0,
            "synthetic_target_count": int(df["is_synthetic_target"].sum()) if not df.empty else 0,
            "real_target_count": int((df["is_synthetic_target"] == 0).sum()) if not df.empty else 0,
            "target_variable": "target_ndvi",
            "target_horizon_range_days": [
                int(df["target_horizon_days"].min()) if not df.empty else None,
                int(df["target_horizon_days"].max()) if not df.empty else None,
            ],
            "feature_columns": [c for c in df.columns if c not in {"target_ndvi", "is_synthetic_target", "target_date"}],
            "missing_data_summary": df.isnull().sum().to_dict() if not df.empty else {},
        }

        manifest_path = os.path.join(self.output_dir, "dataset_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

