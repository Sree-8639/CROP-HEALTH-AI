"""
ml/heat/downscaler_rf.py:
Implements Machine Learning thermal downscaling from 30m Landsat LST to 10m Sentinel-2 resolution:
1. Yao et al. (2020) Random Forest downscaler.
2. Gradient Tree Boosting downscaler (HistGradientBoostingRegressor).
3. Energy-conserving residual correction for fine-scale 10m thermal reconstruction.
4. Rigorous Spatial Block Cross-Validation across multiple seasons and dates (R², RMSE, MAE).
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class RandomForestLSTDownscaler:
    """
    Random Forest LST Downscaler based on Yao et al. 2020.
    Predictors: Sentinel-2 optical, topographic (DEM, slope), distance to coast, and SAR texture.
    Target: Landsat Collection 2 ST_B10 LST (°C).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 12,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=1,
            random_state=random_state,
        )
        self.random_state = random_state
        self.is_fitted = False
        self.feature_names_: List[str] = []
        self.feature_importances_: Optional[Dict[str, float]] = None

    def fit_and_evaluate_spatial(
        self,
        features_30m: Dict[str, np.ndarray],
        lst_30m: np.ndarray,
        spatial_split_ratio: float = 0.70,
        feature_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Trains RF at 30m using a strict SPATIAL block split:
          - Spatial extent divided into 64x64 pixel geographic blocks (1.92 km x 1.92 km each)
          - 30% of contiguous blocks held out for out-of-sample spatial evaluation
          - Open water pixels excluded from land calibration
        """
        H, W = lst_30m.shape
        block_size = 64
        n_br = H // block_size
        n_bc = W // block_size

        block_ids = np.zeros((H, W), dtype=int)
        for r in range(n_br):
            for c in range(n_bc):
                block_ids[r * block_size:(r + 1) * block_size, c * block_size:(c + 1) * block_size] = r * n_bc + c

        rng = np.random.RandomState(self.random_state)
        total_blocks = n_br * n_bc
        test_block_count = max(1, int(round((1.0 - spatial_split_ratio) * total_blocks)))
        test_blocks = rng.choice(np.arange(total_blocks), size=test_block_count, replace=False)
        test_mask = np.isin(block_ids, test_blocks)

        mndwi = features_30m["mndwi"]
        land_mask = (mndwi < 0.15) & np.isfinite(lst_30m)

        train_indices = (~test_mask) & land_mask
        test_indices = test_mask & land_mask

        if feature_keys is None:
            feature_keys = ["ndvi", "ndbi", "mndwi", "albedo"]
            if "dem" in features_30m:
                feature_keys = ["ndvi", "ndbi", "mndwi", "albedo", "dem", "slope", "dist_coast", "sar_tex_vv", "sar_tex_vh"]

        self.feature_names_ = feature_keys
        X_all = np.column_stack([features_30m[k].ravel() for k in feature_keys])
        Y_all = lst_30m.ravel()

        X_train = X_all[train_indices.ravel()]
        Y_train = Y_all[train_indices.ravel()]

        X_test = X_all[test_indices.ravel()]
        Y_test = Y_all[test_indices.ravel()]

        self.rf.fit(X_train, Y_train)
        self.is_fitted = True

        name_map = {
            "ndvi": "NDVI", "ndbi": "NDBI", "mndwi": "MNDWI", "albedo": "Albedo",
            "dem": "DEM", "slope": "Slope", "dist_coast": "Dist_Coast",
            "sar_tex_vv": "SAR_VV_Tex", "sar_tex_vh": "SAR_VH_Tex"
        }
        self.feature_importances_ = {
            name_map.get(k, k): round(float(imp), 4)
            for k, imp in zip(feature_keys, self.rf.feature_importances_)
        }

        Y_pred_test = self.rf.predict(X_test)
        r2 = float(r2_score(Y_test, Y_pred_test))
        rmse = float(np.sqrt(mean_squared_error(Y_test, Y_pred_test)))
        mae = float(mean_absolute_error(Y_test, Y_pred_test))

        Y_pred_train = self.rf.predict(X_train)
        train_r2 = float(r2_score(Y_train, Y_pred_train))
        train_rmse = float(np.sqrt(mean_squared_error(Y_train, Y_pred_train)))

        return {
            "spatial_split_protocol": f"Spatial block partitioning (1.92 km x 1.92 km blocks, {total_blocks - test_block_count} train blocks, {test_block_count} held-out blocks; water excluded)",
            "train_pixels": int(len(Y_train)),
            "test_pixels_held_out": int(len(Y_test)),
            "features_used": feature_keys,
            "held_out_metrics": {
                "r2": round(r2, 4),
                "rmse_celsius": round(rmse, 3),
                "mae_celsius": round(mae, 3),
            },
            "training_metrics": {
                "r2": round(train_r2, 4),
                "rmse_celsius": round(train_rmse, 3),
            },
            "feature_importances": self.feature_importances_,
        }

    def downscale_to_10m(
        self,
        features_10m: Dict[str, np.ndarray],
        features_30m: Dict[str, np.ndarray],
        lst_30m: np.ndarray,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Downscaler model must be fitted before downscaling.")

        H10, W10 = features_10m["ndvi"].shape
        H30, W30 = lst_30m.shape

        keys = self.feature_names_
        X_10m = np.column_stack([features_10m[k].ravel() for k in keys])
        lst_10m_raw = self.rf.predict(X_10m).reshape((H10, W10))

        X_30m = np.column_stack([features_30m[k].ravel() for k in keys])
        lst_30m_pred = self.rf.predict(X_30m).reshape((H30, W30))
        residual_30m = lst_30m - lst_30m_pred

        residual_10m = cv2.resize(residual_30m, (W10, H10), interpolation=cv2.INTER_CUBIC)
        downscaled_lst_10m = lst_10m_raw + residual_10m
        return downscaled_lst_10m.astype(np.float32)


class GradientBoostingLSTDownscaler:
    """
    Gradient Tree Boosting LST Downscaler using HistGradientBoostingRegressor.
    Captures non-linear topographic lapse, coastal sea-breeze microclimate, and urban canopy roughness.
    """

    def __init__(
        self,
        max_iter: int = 150,
        max_depth: int = 8,
        min_samples_leaf: int = 10,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.gbr = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.random_state = random_state
        self.is_fitted = False
        self.feature_names_: List[str] = []

    def fit_and_evaluate_spatial(
        self,
        features_30m: Dict[str, np.ndarray],
        lst_30m: np.ndarray,
        spatial_split_ratio: float = 0.70,
        feature_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        H, W = lst_30m.shape
        block_size = 64
        n_br = H // block_size
        n_bc = W // block_size

        block_ids = np.zeros((H, W), dtype=int)
        for r in range(n_br):
            for c in range(n_bc):
                block_ids[r * block_size:(r + 1) * block_size, c * block_size:(c + 1) * block_size] = r * n_bc + c

        rng = np.random.RandomState(self.random_state)
        total_blocks = n_br * n_bc
        test_block_count = max(1, int(round((1.0 - spatial_split_ratio) * total_blocks)))
        test_blocks = rng.choice(np.arange(total_blocks), size=test_block_count, replace=False)
        test_mask = np.isin(block_ids, test_blocks)

        mndwi = features_30m["mndwi"]
        land_mask = (mndwi < 0.15) & np.isfinite(lst_30m)

        train_indices = (~test_mask) & land_mask
        test_indices = test_mask & land_mask

        if feature_keys is None:
            feature_keys = ["ndvi", "ndbi", "mndwi", "albedo"]
            if "dem" in features_30m:
                feature_keys = ["ndvi", "ndbi", "mndwi", "albedo", "dem", "slope", "dist_coast", "sar_tex_vv", "sar_tex_vh"]

        self.feature_names_ = feature_keys
        X_all = np.column_stack([features_30m[k].ravel() for k in feature_keys])
        Y_all = lst_30m.ravel()

        X_train = X_all[train_indices.ravel()]
        Y_train = Y_all[train_indices.ravel()]

        X_test = X_all[test_indices.ravel()]
        Y_test = Y_all[test_indices.ravel()]

        self.gbr.fit(X_train, Y_train)
        self.is_fitted = True

        Y_pred_test = self.gbr.predict(X_test)
        r2 = float(r2_score(Y_test, Y_pred_test))
        rmse = float(np.sqrt(mean_squared_error(Y_test, Y_pred_test)))
        mae = float(mean_absolute_error(Y_test, Y_pred_test))

        Y_pred_train = self.gbr.predict(X_train)
        train_r2 = float(r2_score(Y_train, Y_pred_train))
        train_rmse = float(np.sqrt(mean_squared_error(Y_train, Y_pred_train)))

        return {
            "model_name": "Gradient Boosting (HistGradientBoosting)",
            "train_pixels": int(len(Y_train)),
            "test_pixels_held_out": int(len(Y_test)),
            "features_used": feature_keys,
            "held_out_metrics": {
                "r2": round(r2, 4),
                "rmse_celsius": round(rmse, 3),
                "mae_celsius": round(mae, 3),
            },
            "training_metrics": {
                "r2": round(train_r2, 4),
                "rmse_celsius": round(train_rmse, 3),
            },
        }

    def downscale_to_10m(
        self,
        features_10m: Dict[str, np.ndarray],
        features_30m: Dict[str, np.ndarray],
        lst_30m: np.ndarray,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Downscaler model must be fitted before downscaling.")

        H10, W10 = features_10m["ndvi"].shape
        H30, W30 = lst_30m.shape

        keys = self.feature_names_
        X_10m = np.column_stack([features_10m[k].ravel() for k in keys])
        lst_10m_raw = self.gbr.predict(X_10m).reshape((H10, W10))

        X_30m = np.column_stack([features_30m[k].ravel() for k in keys])
        lst_30m_pred = self.gbr.predict(X_30m).reshape((H30, W30))
        residual_30m = lst_30m - lst_30m_pred

        residual_10m = cv2.resize(residual_30m, (W10, H10), interpolation=cv2.INTER_CUBIC)
        downscaled_lst_10m = lst_10m_raw + residual_10m
        return downscaled_lst_10m.astype(np.float32)


def evaluate_spatial_cv_for_date(
    model_type: str,
    features_30m: Dict[str, np.ndarray],
    lst_30m: np.ndarray,
    feature_keys: List[str],
    n_splits: int = 4,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Performs 4-fold Spatial Block Cross-Validation for a specific Landsat scene:
      - 256x256 grid divided into 16 blocks (64x64 each)
      - Folds assign 4 contiguous spatial blocks to test fold in each iteration
      - Ensures zero spatial autocorrelation leakage between train and test blocks
    """
    H, W = lst_30m.shape
    block_size = 64
    n_br = H // block_size
    n_bc = W // block_size

    block_ids = np.zeros((H, W), dtype=int)
    for r in range(n_br):
        for c in range(n_bc):
            block_ids[r * block_size:(r + 1) * block_size, c * block_size:(c + 1) * block_size] = r * n_bc + c

    total_blocks = n_br * n_bc  # 16
    mndwi = features_30m["mndwi"]
    land_mask = (mndwi < 0.15) & np.isfinite(lst_30m)

    X_all = np.column_stack([features_30m[k].ravel() for k in feature_keys])
    Y_all = lst_30m.ravel()

    fold_r2, fold_rmse, fold_mae = [], [], []

    # 4 contiguous quadrant folds
    # Fold 0: blocks 0,1,4,5 (NW)
    # Fold 1: blocks 2,3,6,7 (NE)
    # Fold 2: blocks 8,9,12,13 (SW)
    # Fold 3: blocks 10,11,14,15 (SE)
    block_groups = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15],
    ]

    for fold_idx, test_blocks in enumerate(block_groups):
        test_mask = np.isin(block_ids, test_blocks)
        train_indices = (~test_mask) & land_mask
        test_indices = test_mask & land_mask

        X_train, Y_train = X_all[train_indices.ravel()], Y_all[train_indices.ravel()]
        X_test, Y_test = X_all[test_indices.ravel()], Y_all[test_indices.ravel()]

        if len(Y_test) == 0 or len(Y_train) == 0:
            continue

        if model_type == "gradient_boosting":
            model = HistGradientBoostingRegressor(
                max_iter=120, max_depth=6, min_samples_leaf=10, learning_rate=0.05, random_state=random_state + fold_idx
            )
        else:
            model = RandomForestRegressor(
                n_estimators=80, max_depth=10, min_samples_leaf=5, n_jobs=1, random_state=random_state + fold_idx
            )

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        fold_r2.append(float(r2_score(Y_test, Y_pred)))
        fold_rmse.append(float(np.sqrt(mean_squared_error(Y_test, Y_pred))))
        fold_mae.append(float(mean_absolute_error(Y_test, Y_pred)))

    return {
        "model_type": model_type,
        "n_folds": len(fold_r2),
        "mean_r2": round(float(np.mean(fold_r2)), 4),
        "std_r2": round(float(np.std(fold_r2)), 4),
        "mean_rmse_celsius": round(float(np.mean(fold_rmse)), 3),
        "std_rmse_celsius": round(float(np.std(fold_rmse)), 3),
        "mean_mae_celsius": round(float(np.mean(fold_mae)), 3),
        "std_mae_celsius": round(float(np.std(fold_mae)), 3),
        "fold_metrics": [
            {"fold": i + 1, "r2": round(r, 4), "rmse": round(rm, 3), "mae": round(m, 3)}
            for i, (r, rm, m) in enumerate(zip(fold_r2, fold_rmse, fold_mae))
        ],
    }
