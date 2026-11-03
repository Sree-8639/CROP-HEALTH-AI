"""
Modular time-series forecasting algorithms, walk-forward validation engine,
and empirical prediction intervals for Crop Growth & Trend Prediction (Phase 14).
"""

import math
import datetime
from typing import List, Dict, Any, Optional, Tuple, Type
import numpy as np

from ml.growth_prediction.schema import (
    PHYSICAL_BOUNDS,
    SUPPORTED_INDICES,
)

# -------------------------------------------------------------
# Base Forecaster Interface
# -------------------------------------------------------------

class BaseForecaster:
    """Abstract base class for modular vegetation indicator forecasters."""
    
    def __init__(self, index_name: str = "NDVI"):
        self.index_name = index_name
        self.is_fitted = False
        self.last_value: Optional[float] = None
        self.model_name = "BaseForecaster"
        self.version = "v1.0.0"

    def fit(self, times_days: np.ndarray, values: np.ndarray, dates: Optional[List[str]] = None) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, future_times_days: np.ndarray, future_dates: Optional[List[str]] = None) -> np.ndarray:
        raise NotImplementedError

    def clamp(self, preds: np.ndarray) -> np.ndarray:
        """Clamps forecasts to physically valid boundaries for the given index."""
        bounds = PHYSICAL_BOUNDS.get(self.index_name, (-1.0, 1.0))
        return np.clip(preds, bounds[0], bounds[1])


# -------------------------------------------------------------
# 1. Last Observation Forecaster (Persistence / Naive Baseline)
# -------------------------------------------------------------

class LastObservationForecaster(BaseForecaster):
    """
    Persistence baseline: projects future values as the last observed valid value.
    y_hat_{t+h} = y_t
    """
    
    def __init__(self, index_name: str = "NDVI"):
        super().__init__(index_name)
        self.model_name = "LastObservation_Naive"

    def fit(self, times_days: np.ndarray, values: np.ndarray, dates: Optional[List[str]] = None) -> "LastObservationForecaster":
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            raise ValueError(f"No valid observations provided to fit {self.model_name}")
        self.last_value = float(values[valid_mask][-1])
        self.is_fitted = True
        return self

    def predict(self, future_times_days: np.ndarray, future_dates: Optional[List[str]] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.model_name} is not fitted.")
        preds = np.full(shape=future_times_days.shape, fill_value=self.last_value, dtype=np.float64)
        return self.clamp(preds)


# -------------------------------------------------------------
# 2. Moving Average Forecaster
# -------------------------------------------------------------

class MovingAverageForecaster(BaseForecaster):
    """
    Moving average baseline: projects future values as the mean of the last k valid observations.
    k = min(window_size, n_valid)
    """

    def __init__(self, index_name: str = "NDVI", window_size: int = 3):
        super().__init__(index_name)
        self.window_size = window_size
        self.model_name = "MovingAverage"
        self.ma_value: Optional[float] = None

    def fit(self, times_days: np.ndarray, values: np.ndarray, dates: Optional[List[str]] = None) -> "MovingAverageForecaster":
        valid_vals = values[~np.isnan(values)]
        if len(valid_vals) == 0:
            raise ValueError(f"No valid observations provided to fit {self.model_name}")
        k = min(self.window_size, len(valid_vals))
        self.ma_value = float(np.mean(valid_vals[-k:]))
        self.last_value = float(valid_vals[-1])
        self.is_fitted = True
        return self

    def predict(self, future_times_days: np.ndarray, future_dates: Optional[List[str]] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.model_name} is not fitted.")
        preds = np.full(shape=future_times_days.shape, fill_value=self.ma_value, dtype=np.float64)
        return self.clamp(preds)


# -------------------------------------------------------------
# 3. Linear Trend Forecaster
# -------------------------------------------------------------

class LinearTrendForecaster(BaseForecaster):
    """
    Ordinary least squares linear trend forecaster:
    y_hat(t) = alpha + beta * (t - t_0)
    Includes progressive slope damping over multi-step horizons to prevent divergence.
    """

    def __init__(self, index_name: str = "NDVI", slope_damping: float = 0.88):
        super().__init__(index_name)
        self.slope_damping = slope_damping
        self.model_name = "LinearTrend"
        self.slope: float = 0.0
        self.intercept: float = 0.0
        self.t0: float = 0.0
        self.last_t: float = 0.0

    def fit(self, times_days: np.ndarray, values: np.ndarray, dates: Optional[List[str]] = None) -> "LinearTrendForecaster":
        valid_mask = ~np.isnan(values) & ~np.isnan(times_days)
        t_valid = times_days[valid_mask]
        v_valid = values[valid_mask]

        if len(v_valid) == 0:
            raise ValueError(f"No valid observations to fit {self.model_name}")

        self.last_value = float(v_valid[-1])
        self.last_t = float(t_valid[-1])

        if len(v_valid) < 2 or (t_valid[-1] - t_valid[0]) < 1e-4:
            # Fallback to horizontal line if single observation or zero time span
            self.slope = 0.0
            self.intercept = self.last_value
            self.t0 = self.last_t
            self.is_fitted = True
            return self

        self.t0 = float(t_valid[0])
        dt = t_valid - self.t0

        # OLS regression: v = alpha + beta * dt
        # Clamp slope to sensible agronomic bounds per day
        max_daily_rate = 0.02 if self.index_name in ("NDVI", "NDRE", "EVI") else 0.04
        if self.index_name in ("VV", "VH"):
            max_daily_rate = 0.35  # dB per day max

        # Calculate slope
        dt_var = np.var(dt)
        if dt_var > 1e-6:
            cov = np.cov(dt, v_valid)[0, 1]
            raw_slope = cov / dt_var
            self.slope = float(np.clip(raw_slope, -max_daily_rate, max_daily_rate))
            self.intercept = float(np.mean(v_valid) - self.slope * np.mean(dt))
        else:
            self.slope = 0.0
            self.intercept = self.last_value

        self.is_fitted = True
        return self

    def predict(self, future_times_days: np.ndarray, future_dates: Optional[List[str]] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.model_name} is not fitted.")

        preds = []
        for step_idx, t_fut in enumerate(future_times_days):
            delta_from_last = max(0.0, float(t_fut - self.last_t))
            # Progressive damping over extended horizons
            damped_slope = self.slope * (self.slope_damping ** step_idx)
            val = self.last_value + damped_slope * delta_from_last
            preds.append(val)

        return self.clamp(np.array(preds, dtype=np.float64))


# -------------------------------------------------------------
# 4. Seasonal Naive Forecaster
# -------------------------------------------------------------

class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal naive forecaster:
    Matches a future date's day-of-year (DOY) with historical observations from prior cycles.
    Only available when historical timeline spans >= 180 days with >= 10 observations.
    """

    def __init__(self, index_name: str = "NDVI"):
        super().__init__(index_name)
        self.model_name = "SeasonalNaive"
        self.is_available = False
        self.history_doy_values: List[Tuple[int, float]] = []

    def fit(self, times_days: np.ndarray, values: np.ndarray, dates: Optional[List[str]] = None) -> "SeasonalNaiveForecaster":
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            self.last_value = float(values[valid_mask][-1])
        else:
            self.last_value = 0.0

        if dates is None or np.count_nonzero(valid_mask) < 10:
            self.is_available = False
            self.is_fitted = True
            return self

        valid_dates = [d for i, d in enumerate(dates) if valid_mask[i]]
        valid_vals = values[valid_mask]

        # Check total time span
        try:
            d_start = datetime.date.fromisoformat(valid_dates[0][:10])
            d_end = datetime.date.fromisoformat(valid_dates[-1][:10])
            span_days = (d_end - d_start).days
        except Exception:
            self.is_available = False
            self.is_fitted = True
            return self

        if span_days < 180:
            self.is_available = False
            self.is_fitted = True
            return self

        self.history_doy_values = []
        for d_str, v in zip(valid_dates, valid_vals):
            try:
                dt = datetime.date.fromisoformat(d_str[:10])
                doy = dt.timetuple().tm_yday
                self.history_doy_values.append((doy, float(v)))
            except Exception:
                continue

        self.last_value = float(valid_vals[-1])
        self.is_available = len(self.history_doy_values) >= 10
        self.is_fitted = True
        return self

    def predict(self, future_times_days: np.ndarray, future_dates: Optional[List[str]] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.model_name} is not fitted.")

        if not self.is_available or not future_dates:
            # Fallback to persistence if seasonal data insufficient
            fallback = np.full(shape=future_times_days.shape, fill_value=self.last_value, dtype=np.float64)
            return self.clamp(fallback)

        preds = []
        for d_str in future_dates:
            try:
                dt = datetime.date.fromisoformat(d_str[:10])
                target_doy = dt.timetuple().tm_yday
                # Find nearest historical DOY in circular 365-day calendar
                best_val = self.last_value
                min_dist = 999
                for hist_doy, val in self.history_doy_values:
                    # Circular distance
                    dist = abs(target_doy - hist_doy)
                    circ_dist = min(dist, 365 - dist)
                    if circ_dist < min_dist:
                        min_dist = circ_dist
                        best_val = val
                preds.append(best_val)
            except Exception:
                preds.append(self.last_value)

        return self.clamp(np.array(preds, dtype=np.float64))


# -------------------------------------------------------------
# Walk-Forward Chronological Validation Engine
# -------------------------------------------------------------

def walk_forward_validate(
    dates: List[str],
    values: np.ndarray,
    times_days: np.ndarray,
    forecaster_cls: Type[BaseForecaster],
    index_name: str = "NDVI",
    min_train_size: int = 3,
    horizon_steps: int = 1,
) -> Dict[str, Any]:
    """
    Performs strict chronological expanding-window walk-forward validation.
    Never uses random or future-leaking splits.
    
    Returns:
    - mae: Mean Absolute Error
    - rmse: Root Mean Squared Error
    - r2: R-squared coefficient of determination
    - mape: Mean Absolute Percentage Error (computed when |y| > 0.05)
    - residual_std: Standard deviation of residuals for prediction intervals
    - validation_windows_count: Total evaluated cutoffs
    """
    valid_indices = np.where(~np.isnan(values))[0]
    n_valid = len(valid_indices)

    if n_valid < min_train_size + horizon_steps:
        return {
            "model_name": forecaster_cls(index_name).model_name,
            "index_name": index_name,
            "mae": None,
            "rmse": None,
            "r2": None,
            "mape": None,
            "residual_std": None,
            "validation_windows_count": 0,
            "status": "INSUFFICIENT_OBSERVATIONS_FOR_VALIDATION",
        }

    actuals: List[float] = []
    predictions: List[float] = []

    for split_idx in range(min_train_size, n_valid - horizon_steps + 1):
        train_idxs = valid_indices[:split_idx]
        test_idx = valid_indices[split_idx + horizon_steps - 1]

        t_train = times_days[train_idxs]
        v_train = values[train_idxs]
        d_train = [dates[i] for i in train_idxs]

        t_test = np.array([times_days[test_idx]], dtype=np.float64)
        d_test = [dates[test_idx]]
        v_actual = float(values[test_idx])

        forecaster = forecaster_cls(index_name=index_name)
        try:
            forecaster.fit(t_train, v_train, d_train)
            pred = float(forecaster.predict(t_test, d_test)[0])
            actuals.append(v_actual)
            predictions.append(pred)
        except Exception:
            continue

    if len(actuals) == 0:
        return {
            "model_name": forecaster_cls(index_name).model_name,
            "index_name": index_name,
            "mae": None,
            "rmse": None,
            "r2": None,
            "mape": None,
            "residual_std": None,
            "validation_windows_count": 0,
            "status": "NO_VALID_EVALUATION_WINDOWS",
        }

    y_arr = np.array(actuals, dtype=np.float64)
    y_hat_arr = np.array(predictions, dtype=np.float64)
    errors = y_arr - y_hat_arr

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    residual_std = float(np.std(errors, ddof=1)) if len(errors) > 1 else rmse

    # R2 calculation
    y_var = np.var(y_arr)
    if y_var > 1e-6 and len(y_arr) >= 3:
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = round(float(1.0 - (ss_res / ss_tot)), 4)
    else:
        r2 = None

    # MAPE: compute only when |y| >= 0.05 to prevent division by near-zero
    nz_mask = np.abs(y_arr) >= 0.05
    if np.any(nz_mask):
        mape = round(float(np.mean(np.abs(errors[nz_mask] / y_arr[nz_mask])) * 100.0), 2)
    else:
        mape = None

    return {
        "model_name": forecaster_cls(index_name).model_name,
        "index_name": index_name,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": r2,
        "mape": mape,
        "residual_std": round(residual_std, 4),
        "validation_windows_count": len(actuals),
        "status": "VALIDATED",
    }


# -------------------------------------------------------------
# Empirical Prediction Interval Calculation
# -------------------------------------------------------------

def compute_prediction_intervals(
    forecast_values: List[float],
    residual_std: Optional[float],
    index_name: str = "NDVI",
    confidence_level: float = 0.80,
) -> Optional[List[Dict[str, float]]]:
    """
    Calculates calibrated prediction intervals expanding with forecast horizon h:
    margin_h = z * residual_std * sqrt(1 + 0.25 * (h - 1))
    
    z = 1.28 for 80% confidence level.
    Results are strictly clamped to physical sensor index bounds.
    """
    if residual_std is None or math.isnan(residual_std) or residual_std <= 1e-5:
        return None

    # Normal distribution critical value: 80% CI -> z ~ 1.282, 90% CI -> z ~ 1.645
    z = 1.282 if confidence_level <= 0.80 else 1.645
    bounds = PHYSICAL_BOUNDS.get(index_name, (-1.0, 1.0))

    intervals = []
    for step_idx, f_val in enumerate(forecast_values):
        # Forecast uncertainty expands over multi-step horizon
        horizon_expansion = math.sqrt(1.0 + 0.25 * step_idx)
        margin = z * residual_std * horizon_expansion
        lower = max(bounds[0], round(f_val - margin, 4))
        upper = min(bounds[1], round(f_val + margin, 4))
        intervals.append({
            "lower": lower,
            "upper": upper,
            "confidence_level": confidence_level,
        })

    return intervals
