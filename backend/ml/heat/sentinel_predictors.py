"""
ml/heat/sentinel_predictors.py:
Computes and manages Sentinel-2 high-resolution 10m thermal predictors for LST downscaling:
- NDVI: Normalized Difference Vegetation Index (B8 - B4) / (B8 + B4)
- NDBI: Normalized Difference Built-up Index (B11 - B8) / (B11 + B8)
- MNDWI: Modified Normalized Difference Water Index (B3 - B11) / (B3 + B11)
- Albedo: Broadband surface shortwave albedo
Provides 10m fine-grid predictors and 30m spatially aggregated predictors as in Yao et al. 2020.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2


def calculate_ndvi(b8_nir: np.ndarray, b4_red: np.ndarray) -> np.ndarray:
    """Computes Normalized Difference Vegetation Index (NDVI)."""
    denom = b8_nir + b4_red + 1e-7
    return np.clip((b8_nir - b4_red) / denom, -1.0, 1.0).astype(np.float32)


def calculate_ndbi(b11_swir: np.ndarray, b8_nir: np.ndarray) -> np.ndarray:
    """Computes Normalized Difference Built-up Index (NDBI)."""
    denom = b11_swir + b8_nir + 1e-7
    return np.clip((b11_swir - b8_nir) / denom, -1.0, 1.0).astype(np.float32)


def calculate_mndwi(b3_green: np.ndarray, b11_swir: np.ndarray) -> np.ndarray:
    """Computes Modified Normalized Difference Water Index (MNDWI)."""
    denom = b3_green + b11_swir + 1e-7
    return np.clip((b3_green - b11_swir) / denom, -1.0, 1.0).astype(np.float32)


def calculate_broadband_albedo(
    b2_blue: np.ndarray,
    b4_red: np.ndarray,
    b8_nir: np.ndarray,
    b11_swir1: np.ndarray,
    b12_swir2: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes broadband surface shortwave albedo from Sentinel-2 multispectral bands
    using Liang (2001) empirical coefficients adapted for Sentinel-2.
    """
    if b12_swir2 is None:
        b12_swir2 = b11_swir1 * 0.85
    albedo = 0.356 * b2_blue + 0.130 * b4_red + 0.373 * b8_nir + 0.085 * b11_swir1 + 0.072 * b12_swir2 - 0.0018
    return np.clip(albedo, 0.02, 0.80).astype(np.float32)


def aggregate_10m_to_30m(array_10m: np.ndarray) -> np.ndarray:
    """
    Spatially aggregates a 10m grid to 30m resolution (3x3 area averaging)
    to match Landsat 8/9 ST_B10 pixel dimensions.
    """
    H10, W10 = array_10m.shape
    H30, W30 = H10 // 3, W10 // 3
    # Crop to multiple of 3
    cropped = array_10m[:H30 * 3, :W30 * 3]
    # Reshape and average
    reshaped = cropped.reshape(H30, 3, W30, 3)
    return reshaped.mean(axis=(1, 3)).astype(np.float32)


class Sentinel2PredictorSuite:
    """
    Generates aligned 10m and 30m multispectral predictors for Visakhapatnam.
    Grid shape: 768x768 at 10m corresponds exactly to 256x256 at 30m (7.68 km extent).
    """

    def __init__(
        self,
        grid_shape_10m: Tuple[int, int] = (768, 768),
        random_seed: int = 42,
    ):
        self.grid_shape_10m = grid_shape_10m
        self.grid_shape_30m = (grid_shape_10m[0] // 3, grid_shape_10m[1] // 3)
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        self.predictors_10m = self._generate_10m_predictors()
        self.predictors_30m = self._aggregate_to_30m(self.predictors_10m)

    def _generate_10m_predictors(self) -> Dict[str, np.ndarray]:
        """
        Synthesizes high-resolution 10m Sentinel-2 indices matching Visakhapatnam morphology:
          - High NDBI & low NDVI in Industrial (Steel Plant/Port) and Urban Core (Jagadamba/Dwaraka)
          - High NDVI & negative NDBI in Hill Ridges (Simhachalam & Kailasagiri)
          - High MNDWI & low Albedo in Bay of Bengal and reservoirs
        """
        H, W = self.grid_shape_10m
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        xx, yy = np.meshgrid(x, y)

        # 1. Base Vegetation Pattern (NDVI)
        # Dense forest in NW (Simhachalam) and NE (Kailasagiri)
        ridge1 = np.exp(-((xx - 0.35)**2 + (yy - 0.25)**2) / 0.02)
        ridge2 = np.exp(-((xx - 0.80)**2 + (yy - 0.25)**2) / 0.02)
        forest_canopy = np.clip(0.65 * (ridge1 + ridge2), 0.0, 0.85)

        # Urban / Industrial canopy deficit
        urban_core = np.exp(-((xx - 0.65)**2 + (yy - 0.55)**2) / 0.03)
        industrial = np.exp(-((xx - 0.25)**2 + (yy - 0.70)**2) / 0.05)
        
        ndvi_base = 0.22 + forest_canopy - 0.14 * urban_core - 0.16 * industrial
        ndvi = np.clip(ndvi_base + self.rng.normal(0, 0.03, (H, W)), -0.1, 0.88).astype(np.float32)

        # 2. Built-up Pattern (NDBI)
        ndbi_base = -0.18 - 0.35 * forest_canopy + 0.45 * urban_core + 0.55 * industrial
        ndbi = np.clip(ndbi_base + self.rng.normal(0, 0.04, (H, W)), -0.6, 0.75).astype(np.float32)

        # 3. Water Body Pattern (MNDWI)
        sea_mask = xx > 0.88
        mndwi = np.full((H, W), -0.35, dtype=np.float32)
        mndwi[sea_mask] = 0.75
        mndwi += self.rng.normal(0, 0.03, (H, W)).astype(np.float32)

        # Overwrite water in NDVI / NDBI
        ndvi[sea_mask] = -0.45
        ndbi[sea_mask] = -0.50

        # 4. Albedo
        # High albedo in bare metal industrial roofs and light concrete, low in water & dense forest
        albedo_base = 0.16 + 0.18 * industrial + 0.08 * urban_core - 0.07 * forest_canopy
        albedo = np.clip(albedo_base + self.rng.normal(0, 0.02, (H, W)), 0.04, 0.65).astype(np.float32)
        albedo[sea_mask] = 0.05

        return {
            "ndvi": ndvi,
            "ndbi": ndbi,
            "mndwi": mndwi,
            "albedo": albedo,
        }

    def _aggregate_to_30m(self, preds_10m: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Aggregates all 10m predictors to 30m using 3x3 pixel area averaging."""
        return {
            key: aggregate_10m_to_30m(arr) for key, arr in preds_10m.items()
        }

    def get_feature_matrix_30m(self) -> np.ndarray:
        """Returns (N_pixels, 4) feature matrix at 30m resolution [NDVI, NDBI, MNDWI, Albedo]."""
        p = self.predictors_30m
        return np.column_stack([
            p["ndvi"].ravel(),
            p["ndbi"].ravel(),
            p["mndwi"].ravel(),
            p["albedo"].ravel(),
        ])

    def get_feature_matrix_10m(self) -> np.ndarray:
        """Returns (N_pixels, 4) feature matrix at 10m resolution [NDVI, NDBI, MNDWI, Albedo]."""
        p = self.predictors_10m
        return np.column_stack([
            p["ndvi"].ravel(),
            p["ndbi"].ravel(),
            p["mndwi"].ravel(),
            p["albedo"].ravel(),
        ])
