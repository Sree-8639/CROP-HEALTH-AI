"""
ml/crop/time_series.py:
Builds and manages per-pixel multimodal time series from Sentinel-2 (NDVI, NDRE, NDMI)
and Sentinel-1 (VV, VH, ratio) for the Area of Interest (AOI).
Supports optional ground-truth reference labels via labels.csv.
"""

import os
import csv
import json
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd


class PixelTimeSeriesDataset:
    """
    Multimodal per-pixel time series container for agricultural monitoring.
    
    Channels:
      - Optical (Sentinel-2): NDVI, NDRE, NDMI
      - SAR (Sentinel-1): VV (dB), VH (dB), Ratio (VV - VH in dB)
    """

    CHANNELS_S2 = ["ndvi", "ndre", "ndmi"]
    CHANNELS_S1 = ["vv_db", "vh_db", "ratio_db"]
    CHANNELS_ALL = CHANNELS_S2 + CHANNELS_S1

    def __init__(
        self,
        aoi_name: str = "Tadepalligudem",
        bbox: Optional[List[float]] = None,
        dates: Optional[List[str]] = None,
        grid_shape: Tuple[int, int] = (64, 64),
        labels_path: Optional[str] = None,
        random_seed: int = 42,
    ):
        self.aoi_name = aoi_name
        self.bbox = bbox or [81.503, 16.793, 81.543, 16.833]
        self.grid_shape = grid_shape
        self.random_seed = random_seed
        self.labels_path = labels_path

        # Default multi-temporal observation dates (covering Rabi phenological progression)
        self.dates = dates or [
            "2024-01-05", "2024-01-10", "2024-01-17", "2024-01-24", "2024-01-29",
            "2024-02-05", "2024-02-12", "2024-02-19", "2024-02-24", "2024-02-29"
        ]
        self.num_timesteps = len(self.dates)
        self.num_pixels = grid_shape[0] * grid_shape[1]

        # Load or generate multimodal tensor: shape (T, C, H, W)
        self.tensor = self._build_multimodal_tensor()
        
        # Load optional ground truth labels
        self.labels: Optional[np.ndarray] = None
        self.has_labels: bool = False
        if labels_path and os.path.exists(labels_path):
            self.load_labels(labels_path)

    def _build_multimodal_tensor(self) -> np.ndarray:
        """
        Loads the real Copernicus Sentinel-2 and Sentinel-1 multimodal tensor.
        Data source: Real Copernicus CDSE Level-2A optical and GRD C-band SAR observations
        over Tadepalligudem AOI ([81.503, 16.793, 81.543, 16.833]).
        """
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        npz_path = os.path.join(data_dir, "real_tadepalligudem_multimodal_tensor.npz")
        meta_path = os.path.join(data_dir, "real_tadepalligudem_meta.json")

        if not os.path.exists(npz_path):
            from ml.crop.build_real_time_series import build_and_save_real_time_series
            build_and_save_real_time_series()

        with np.load(npz_path) as npz:
            tensor = npz["tensor"]
            if "optical_masks" in npz:
                self.optical_masks = npz["optical_masks"]
            else:
                self.optical_masks = np.ones((tensor.shape[0], tensor.shape[2], tensor.shape[3]), dtype=bool)

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                obs = meta.get("observation_windows", [])
                if obs:
                    self.dates = [o.get("sentinel2_acquisition_date", "")[:10] for o in obs]
                    self.num_timesteps = len(self.dates)

        return tensor.astype(np.float32)

    def load_labels(self, path: str) -> None:
        """
        Loads optional ground-truth labels from CSV.
        Expected format: CSV with header, rows containing (pixel_idx or row, col, and label).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"labels.csv not found at {path}")

        labels_list = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("label") or row.get("class") or row.get("target")
                if val is not None:
                    try:
                        labels_list.append(int(val))
                    except ValueError:
                        labels_list.append(val)
        
        if labels_list:
            self.labels = np.array(labels_list)
            self.has_labels = True

    def get_pixel_feature_matrix(self, mode: str = "fused") -> np.ndarray:
        """
        Flattens spatial dimensions into (N_pixels, D_features).
        
        Modes:
          - "fused": S1 + S2 (T * 6 features)
          - "s2_only": NDVI, NDRE, NDMI (T * 3 features)
          - "s1_only": VV, VH, Ratio (T * 3 features)
        """
        T, C, H, W = self.tensor.shape
        N = H * W

        if mode == "s2_only":
            sub_tensor = self.tensor[:, 0:3, :, :] # (T, 3, H, W)
        elif mode == "s1_only":
            sub_tensor = self.tensor[:, 3:6, :, :] # (T, 3, H, W)
        elif mode == "fused":
            sub_tensor = self.tensor              # (T, 6, H, W)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'fused', 's2_only', or 's1_only'.")

        # Transpose to (H, W, T, C_sub) -> reshape to (N, T * C_sub)
        C_sub = sub_tensor.shape[1]
        reordered = np.transpose(sub_tensor, (2, 3, 0, 1)) # (H, W, T, C_sub)
        matrix = reordered.reshape(N, T * C_sub)
        return matrix.astype(np.float32)

    def get_spatial_blocks(self, n_blocks: int = 4) -> np.ndarray:
        """
        Partitions the spatial grid into non-overlapping spatial blocks (e.g. quadrants).
        Ensures zero spatial leakage across train/test splits.
        """
        H, W = self.grid_shape
        row_indices, col_indices = np.indices((H, W))
        r_flat = row_indices.ravel()
        c_flat = col_indices.ravel()

        if n_blocks == 4:
            blocks = np.zeros(H * W, dtype=int)
            blocks[(r_flat < H // 2) & (c_flat >= W // 2)] = 1  # NE
            blocks[(r_flat >= H // 2) & (c_flat < W // 2)] = 2  # SW
            blocks[(r_flat >= H // 2) & (c_flat >= W // 2)] = 3  # SE
            return blocks
        else:
            # General n-strip partitioning along latitude
            strip_height = H / n_blocks
            return np.clip((r_flat / strip_height).astype(int), 0, n_blocks - 1)

    def extract_descals_optical_features(self) -> pd.DataFrame:
        """
        Extracts temporal phenology window features reproducing Descals et al. 2025
        strictly from optical Sentinel-2 observations (NDVI, NDRE, NDMI).
        """
        T, C, H, W = self.tensor.shape
        N = H * W
        ndvi = self.tensor[:, 0, :, :].reshape(T, N)
        ndre = self.tensor[:, 1, :, :].reshape(T, N)
        ndmi = self.tensor[:, 2, :, :].reshape(T, N)

        # Early stage: Jan 5 & Jan 10 (t0, t1)
        early_ndvi = np.mean(ndvi[:2, :], axis=0)
        # Peak / reproductive stage: Feb 19 & Feb 29 (t2, t3)
        peak_ndvi = np.max(ndvi[2:, :], axis=0)
        senescence_ndvi = np.min(ndvi, axis=0)
        amplitude_ndvi = np.maximum(0.0, peak_ndvi - early_ndvi)
        greenup_rate = (ndvi[-1, :] - ndvi[0, :]) / max(1.0, float(T - 1))

        early_ndre = np.mean(ndre[:2, :], axis=0)
        peak_ndre = np.max(ndre[2:, :], axis=0)
        mean_ndmi = np.mean(ndmi, axis=0)

        df = pd.DataFrame({
            "early_stage_mean_ndvi": early_ndvi,
            "mid_stage_peak_ndvi": peak_ndvi,
            "late_stage_min_ndvi": senescence_ndvi,
            "seasonal_ndvi_amplitude": amplitude_ndvi,
            "greenup_rate": greenup_rate,
            "seasonal_mean_ndvi": np.mean(ndvi, axis=0),
            "seasonal_std_ndvi": np.std(ndvi, axis=0),
            "early_stage_mean_ndre": early_ndre,
            "peak_stage_max_ndre": peak_ndre,
            "seasonal_mean_ndmi": mean_ndmi,
        })
        return df

    def extract_fused_features(self) -> pd.DataFrame:
        """
        Extracts full multimodal fusion features combining optical Sentinel-2
        and Sentinel-1 C-SAR backscatter dynamics.
        """
        df_descals = self.extract_descals_optical_features()
        T, C, H, W = self.tensor.shape
        N = H * W
        vv = self.tensor[:, 3, :, :].reshape(T, N)
        vh = self.tensor[:, 4, :, :].reshape(T, N)
        ratio = self.tensor[:, 5, :, :].reshape(T, N)

        sar_dict = {
            "sar_vv_mean": np.mean(vv, axis=0),
            "sar_vv_std": np.std(vv, axis=0),
            "sar_vh_mean": np.mean(vh, axis=0),
            "sar_vh_std": np.std(vh, axis=0),
            "sar_ratio_mean": np.mean(ratio, axis=0),
            "sar_ratio_std": np.std(ratio, axis=0),
            "sar_vh_biomass_expansion": vh[-1, :] - vh[0, :],
            "sar_vv_amplitude": np.max(vv, axis=0) - np.min(vv, axis=0),
            "canopy_radar_optical_ratio": df_descals["seasonal_mean_ndvi"].values / (np.abs(np.mean(vh, axis=0)) + 1e-4),
        }
        sar_df = pd.DataFrame(sar_dict)
        return pd.concat([df_descals, sar_df], axis=1)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Returns statistical overview of the time series dataset."""
        T, C, H, W = self.tensor.shape
        stats = {
            "aoi_name": self.aoi_name,
            "bbox": self.bbox,
            "grid_dimensions": {"height": H, "width": W, "total_pixels": H * W},
            "timesteps": T,
            "observation_dates": self.dates,
            "features_per_timestep": 6,
            "channels": self.CHANNELS_ALL,
            "has_ground_truth_labels": self.has_labels,
        }
        for i, ch in enumerate(self.CHANNELS_ALL):
            arr = self.tensor[:, i, :, :]
            stats[f"{ch}_mean"] = round(float(np.mean(arr)), 4)
            stats[f"{ch}_std"] = round(float(np.std(arr)), 4)
            stats[f"{ch}_min"] = round(float(np.min(arr)), 4)
            stats[f"{ch}_max"] = round(float(np.max(arr)), 4)
        return stats

