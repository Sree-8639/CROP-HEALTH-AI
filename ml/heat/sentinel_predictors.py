"""
ml/heat/sentinel_predictors.py:
Computes and manages Sentinel-2, Copernicus DEM, and Sentinel-1 SAR predictors for LST downscaling:
Baseline 4 features:
- NDVI: Normalized Difference Vegetation Index (B8 - B4) / (B8 + B4)
- NDBI: Normalized Difference Built-up Index (B11 - B8) / (B11 + B8)
- MNDWI: Modified Normalized Difference Water Index (B3 - B11) / (B3 + B11)
- Albedo: Broadband surface shortwave albedo (Liang 2001)

Expanded 9 features:
- DEM: Copernicus GLO-30 30m Digital Elevation Model (m)
- Slope: Topographic slope in degrees derived from DEM
- Distance to Coast: Euclidean distance to Bay of Bengal shoreline (km)
- SAR Texture VV: Local standard deviation of Sentinel-1 VV backscatter (dB)
- SAR Texture VH: Local standard deviation of Sentinel-1 VH backscatter (dB)

Provides 10m fine-grid predictors and 30m spatially aggregated predictors as in Yao et al. 2020.
"""

import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2

REAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "real_visakhapatnam_heat_benchmark.npz")


def calculate_ndvi(b8_nir: np.ndarray, b4_red: np.ndarray) -> np.ndarray:
    denom = b8_nir + b4_red + 1e-7
    return np.clip((b8_nir - b4_red) / denom, -1.0, 1.0).astype(np.float32)


def calculate_ndbi(b11_swir: np.ndarray, b8_nir: np.ndarray) -> np.ndarray:
    denom = b11_swir + b8_nir + 1e-7
    return np.clip((b11_swir - b8_nir) / denom, -1.0, 1.0).astype(np.float32)


def calculate_mndwi(b3_green: np.ndarray, b11_swir: np.ndarray) -> np.ndarray:
    denom = b3_green + b11_swir + 1e-7
    return np.clip((b3_green - b11_swir) / denom, -1.0, 1.0).astype(np.float32)


def calculate_broadband_albedo(
    b2_blue: np.ndarray,
    b4_red: np.ndarray,
    b8_nir: np.ndarray,
    b11_swir1: np.ndarray,
    b12_swir2: Optional[np.ndarray] = None,
) -> np.ndarray:
    if b12_swir2 is None:
        b12_swir2 = b11_swir1 * 0.85
    albedo = 0.356 * b2_blue + 0.130 * b4_red + 0.373 * b8_nir + 0.085 * b11_swir1 + 0.072 * b12_swir2 - 0.0018
    return np.clip(albedo, 0.02, 0.80).astype(np.float32)


def aggregate_10m_to_30m(array_10m: np.ndarray) -> np.ndarray:
    H10, W10 = array_10m.shape
    H30, W30 = H10 // 3, W10 // 3
    cropped = array_10m[:H30 * 3, :W30 * 3]
    reshaped = cropped.reshape(H30, 3, W30, 3)
    return reshaped.mean(axis=(1, 3)).astype(np.float32)


class Sentinel2PredictorSuite:
    """
    Generates aligned 10m and 30m multispectral, topographic, and radar predictors for Visakhapatnam.
    Grid shape: 768x768 at 10m corresponds exactly to 256x256 at 30m (7.68 km extent).
    """

    BASELINE_KEYS = ["ndvi", "ndbi", "mndwi", "albedo"]
    EXPANDED_KEYS = [
        "ndvi", "ndbi", "mndwi", "albedo",
        "dem", "slope", "dist_coast",
        "sar_tex_vv", "sar_tex_vh"
    ]

    def __init__(
        self,
        grid_shape_10m: Tuple[int, int] = (768, 768),
        random_seed: int = 42,
    ):
        self.grid_shape_10m = grid_shape_10m
        self.grid_shape_30m = (grid_shape_10m[0] // 3, grid_shape_10m[1] // 3)
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        self.predictors_10m: Dict[str, np.ndarray] = {}
        self.predictors_30m: Dict[str, np.ndarray] = {}

        if os.path.exists(REAL_DATA_PATH):
            data = np.load(REAL_DATA_PATH)
            # Base 4
            self.predictors_10m["ndvi"] = data["features_10m_ndvi"]
            self.predictors_10m["ndbi"] = data["features_10m_ndbi"]
            self.predictors_10m["mndwi"] = data["features_10m_mndwi"]
            self.predictors_10m["albedo"] = data["features_10m_albedo"]

            self.predictors_30m["ndvi"] = data["features_30m_ndvi"]
            self.predictors_30m["ndbi"] = data["features_30m_ndbi"]
            self.predictors_30m["mndwi"] = data["features_30m_mndwi"]
            self.predictors_30m["albedo"] = data["features_30m_albedo"]

            # Additional 5 predictors if present
            for extra in ["dem", "slope", "dist_coast", "sar_tex_vv", "sar_tex_vh"]:
                k10 = f"features_10m_{extra}"
                k30 = f"features_30m_{extra}"
                if k10 in data and k30 in data:
                    self.predictors_10m[extra] = data[k10]
                    self.predictors_30m[extra] = data[k30]
        else:
            self.predictors_10m = self._generate_fallback_10m_predictors()
            self.predictors_30m = {k: aggregate_10m_to_30m(v) for k, v in self.predictors_10m.items()}

    def _generate_fallback_10m_predictors(self) -> Dict[str, np.ndarray]:
        H, W = self.grid_shape_10m
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        xx, yy = np.meshgrid(x, y)

        ridge = np.exp(-((xx - 0.4)**2 + (yy - 0.3)**2) / 0.03)
        ndvi = np.clip(0.25 + 0.5 * ridge + self.rng.normal(0, 0.02, (H, W)), -0.1, 0.85).astype(np.float32)
        ndbi = np.clip(0.10 - 0.4 * ridge + self.rng.normal(0, 0.03, (H, W)), -0.5, 0.70).astype(np.float32)
        mndwi = np.full((H, W), -0.3, dtype=np.float32)
        mndwi[xx > 0.85] = 0.8
        albedo = np.clip(0.18 + self.rng.normal(0, 0.02, (H, W)), 0.05, 0.60).astype(np.float32)
        dem = (ridge * 400.0).astype(np.float32)
        slope = (ridge * 25.0).astype(np.float32)
        dist_coast = ((1.0 - xx) * 10.0).astype(np.float32)
        sar_tex_vv = np.full((H, W), 1.2, dtype=np.float32)
        sar_tex_vh = np.full((H, W), 1.5, dtype=np.float32)

        return {
            "ndvi": ndvi, "ndbi": ndbi, "mndwi": mndwi, "albedo": albedo,
            "dem": dem, "slope": slope, "dist_coast": dist_coast,
            "sar_tex_vv": sar_tex_vv, "sar_tex_vh": sar_tex_vh
        }

    def get_feature_matrix_30m(self, expanded: bool = False) -> Tuple[np.ndarray, List[str]]:
        keys = self.EXPANDED_KEYS if (expanded and "dem" in self.predictors_30m) else self.BASELINE_KEYS
        cols = [self.predictors_30m[k].ravel() for k in keys]
        return np.column_stack(cols).astype(np.float32), keys

    def get_feature_matrix_10m(self, expanded: bool = False) -> Tuple[np.ndarray, List[str]]:
        keys = self.EXPANDED_KEYS if (expanded and "dem" in self.predictors_10m) else self.BASELINE_KEYS
        cols = [self.predictors_10m[k].ravel() for k in keys]
        return np.column_stack(cols).astype(np.float32), keys
