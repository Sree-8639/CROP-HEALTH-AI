"""
Crop Health AI — HistGradientBoosting Satellite Time-Series Forecaster (Phase 21.8)

Model A for Satellite Vegetation-Index Forecasting:
- Uses scikit-learn HistGradientBoostingRegressor.
- Natively supports missing values (crucial for real SAR/cloud data).
- Versioned feature schema and fixed random seed for strict reproducibility.
- Clamped outputs to physical vegetation boundaries [-1.0, 1.0].
"""

from typing import Dict, Any, List, Optional, Tuple
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import joblib

FEATURE_SCHEMA_V1 = [
    "ndvi",
    "ndmi",
    "ndre",
    "evi",
    "b02",
    "b03",
    "b04",
    "b05",
    "b08",
    "b11",
    "sar_vv",
    "sar_vh",
    "sar_vv_vh_ratio",
    "prev_ndvi",
    "prev_ndmi",
    "delta_ndvi",
    "delta_ndmi",
    "rate_ndvi_per_day",
    "observation_interval_days",
    "history_observation_count",
    "cloud_coverage",
    "valid_pixel_pct",
    "is_missing_sar",
    "is_missing_optical",
    "target_horizon_days",
]


class HistGradientBoostingForecaster:
    """Gradient boosting regressor for satellite NDVI time-series forecasting."""

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.08,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ):
        self.model_name = "HistGradientBoosting_NDVI_Forecaster"
        self.model_version = "v1.0.0-research"
        self.feature_schema = FEATURE_SCHEMA_V1
        self.random_state = random_state

        self.regressor = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=max_iter,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self.is_fitted = False
        self.training_metadata: Dict[str, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        dataset_version: str = "v1.0.0",
        notes: Optional[str] = None,
    ) -> "HistGradientBoostingForecaster":
        """Fits the gradient boosting model on verified feature matrix X and target y."""
        X_sub = X[self.feature_schema].copy()
        y_arr = np.asarray(y, dtype=np.float64)

        # Validate non-empty
        if len(X_sub) == 0:
            raise ValueError("Cannot fit HistGradientBoostingForecaster on empty dataset.")

        start_time = datetime.datetime.now(datetime.timezone.utc)
        self.regressor.fit(X_sub, y_arr)
        duration_s = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()

        self.is_fitted = True

        self.training_metadata = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": "HistGradientBoostingRegressor",
            "trained_at": start_time.isoformat(),
            "training_duration_seconds": round(duration_s, 3),
            "training_samples_count": len(X_sub),
            "dataset_version": dataset_version,
            "random_seed": self.random_state,
            "hyperparameters": {
                "max_iter": self.regressor.max_iter,
                "learning_rate": self.regressor.learning_rate,
                "min_samples_leaf": self.regressor.min_samples_leaf,
            },
            "feature_schema": self.feature_schema,
            "notes": notes or "Satellite NDVI time-series regression.",
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates bounded NDVI forecasts [-1.0, 1.0]."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        X_sub = X[self.feature_schema].copy()
        raw_preds = self.regressor.predict(X_sub)
        # Clamped to physical NDVI limits
        return np.clip(raw_preds, -1.0, 1.0)

    def compute_feature_importances(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Computes permutation feature importance on validation set."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        X_sub = X_val[self.feature_schema].copy()
        res = permutation_importance(
            self.regressor,
            X_sub,
            y_val,
            n_repeats=5,
            random_state=self.random_state,
        )

        importances = {}
        for feat, imp in zip(self.feature_schema, res.importances_mean):
            importances[feat] = round(float(imp), 5)

        # Sort descending
        return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    def save_artifact(self, filepath: str) -> None:
        """Saves model and metadata to artifact file."""
        payload = {
            "regressor": self.regressor,
            "metadata": self.training_metadata,
            "feature_schema": self.feature_schema,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load_artifact(cls, filepath: str) -> "HistGradientBoostingForecaster":
        """Loads trained model artifact."""
        payload = joblib.load(filepath)
        forecaster = cls()
        forecaster.regressor = payload["regressor"]
        forecaster.training_metadata = payload["metadata"]
        forecaster.feature_schema = payload["feature_schema"]
        forecaster.model_name = payload["model_name"]
        forecaster.model_version = payload["model_version"]
        forecaster.is_fitted = True
        return forecaster

