"""
ml/heat/downscaler_rf.py:
Implements Random Forest thermal downscaling from 30m Landsat LST to 10m Sentinel-2 resolution
following the methodology of Yao et al. 2020.
Evaluates model on held-out pixels using a rigorous SPATIAL split (R², RMSE, MAE).
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



class RandomForestLSTDownscaler:
    """
    Random Forest LST Downscaler based on Yao et al. 2020:
    "A statistical downscaling method for thermal infrared imagery based on random forest regressor".
    
    Predictors: Sentinel-2 NDVI, NDBI, MNDWI, Albedo.
    Target: Landsat Collection 2 ST_B10 LST (°C).
    """

    FEATURE_NAMES = ["NDVI", "NDBI", "MNDWI", "Albedo"]

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
            n_jobs=-1,
            random_state=random_state,
        )
        self.random_state = random_state
        self.is_fitted = False
        self.feature_importances_: Optional[Dict[str, float]] = None

    def fit_and_evaluate_spatial(
        self,
        features_30m: Dict[str, np.ndarray],
        lst_30m: np.ndarray,
        spatial_split_ratio: float = 0.70,
    ) -> Dict[str, Any]:
        """
        Trains RF at 30m using a strict SPATIAL split:
          - Train region: Northern & Western quadrants (row < 70% or col < 70%)
          - Held-out test region: South-Eastern spatial block (row >= 70% and col >= 70%)
        Returns R², RMSE, MAE on the held-out spatial block.
        """
        H, W = lst_30m.shape
        split_r = int(H * spatial_split_ratio)
        split_c = int(W * spatial_split_ratio)

        # Spatial split masks
        # Test mask: distinct geographic block
        test_mask = np.zeros((H, W), dtype=bool)
        test_mask[split_r:, split_c:] = True
        
        # Exclude open water bodies where MNDWI > 0.4 from land LST calibration
        mndwi = features_30m["mndwi"]
        land_mask = (mndwi < 0.40) & np.isfinite(lst_30m)

        train_indices = (~test_mask) & land_mask
        test_indices = test_mask & land_mask

        # Assemble feature vectors
        X_all = np.column_stack([
            features_30m["ndvi"].ravel(),
            features_30m["ndbi"].ravel(),
            features_30m["mndwi"].ravel(),
            features_30m["albedo"].ravel(),
        ])
        Y_all = lst_30m.ravel()

        X_train = X_all[train_indices.ravel()]
        Y_train = Y_all[train_indices.ravel()]

        X_test = X_all[test_indices.ravel()]
        Y_test = Y_all[test_indices.ravel()]

        # Fit model
        self.rf.fit(X_train, Y_train)
        self.is_fitted = True

        # Feature importances
        self.feature_importances_ = {
            name: round(float(imp), 4)
            for name, imp in zip(self.FEATURE_NAMES, self.rf.feature_importances_)
        }

        # Predict on held-out spatial test block
        Y_pred_test = self.rf.predict(X_test)

        r2 = float(r2_score(Y_test, Y_pred_test))
        rmse = float(np.sqrt(mean_squared_error(Y_test, Y_pred_test)))
        mae = float(mean_absolute_error(Y_test, Y_pred_test))

        # Predict on training region for baseline comparison
        Y_pred_train = self.rf.predict(X_train)
        train_r2 = float(r2_score(Y_train, Y_pred_train))
        train_rmse = float(np.sqrt(mean_squared_error(Y_train, Y_pred_train)))

        return {
            "spatial_split_protocol": "Strict geographic block partition: 70% North/West for training, 30% South-East held-out test block.",
            "train_pixels": int(len(Y_train)),
            "test_pixels_held_out": int(len(Y_test)),
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
        """
        Applies trained Random Forest to 10m Sentinel-2 predictors
        and performs energy-conserving residual correction (Yao et al. 2020).
        """
        if not self.is_fitted:
            raise RuntimeError("Downscaler model must be fitted before downscaling.")

        H10, W10 = features_10m["ndvi"].shape
        H30, W30 = lst_30m.shape

        # 1. High-resolution raw prediction at 10m
        X_10m = np.column_stack([
            features_10m["ndvi"].ravel(),
            features_10m["ndbi"].ravel(),
            features_10m["mndwi"].ravel(),
            features_10m["albedo"].ravel(),
        ])
        lst_10m_raw = self.rf.predict(X_10m).reshape((H10, W10))

        # 2. Residual calculation at 30m
        X_30m = np.column_stack([
            features_30m["ndvi"].ravel(),
            features_30m["ndbi"].ravel(),
            features_30m["mndwi"].ravel(),
            features_30m["albedo"].ravel(),
        ])
        lst_30m_pred = self.rf.predict(X_30m).reshape((H30, W30))
        residual_30m = lst_30m - lst_30m_pred

        # 3. Bicubic interpolation of residuals from 30m to 10m
        residual_10m = cv2.resize(residual_30m, (W10, H10), interpolation=cv2.INTER_CUBIC)

        # 4. Final Energy-Conserving Downscaled LST at 10m
        downscaled_lst_10m = lst_10m_raw + residual_10m

        return downscaled_lst_10m.astype(np.float32)
