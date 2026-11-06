"""
Crop Health AI — Research Evaluation Metrics (Phase 21.9)

Scientific metrics for satellite vegetation-index time-series forecasting:
- MAE, RMSE, R², Mean Bias.
- Per-horizon breakdown.
- Field-level clustered bootstrap confidence intervals.
- Comparative delta vs persistence baseline.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np


class ForecastEvaluationMetrics:
    """Computes comprehensive regression metrics with bootstrap confidence intervals."""

    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Calculates standard deterministic regression metrics."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if np.sum(mask) == 0:
            return {
                "sample_count": 0,
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "mean_bias": float("nan"),
                "mape_pct": float("nan"),
            }

        yt = y_true[mask]
        yp = y_pred[mask]
        n = len(yt)

        errors = yp - yt
        abs_errors = np.abs(errors)
        sq_errors = errors ** 2

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(sq_errors)))
        mean_bias = float(np.mean(errors))

        # R² Calculation
        ss_res = np.sum(sq_errors)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        if ss_tot > 1e-9:
            r2 = float(1.0 - (ss_res / ss_tot))
        else:
            r2 = 0.0

        # Mean Absolute Percentage Error (excluding near-zero denominators)
        nz_mask = np.abs(yt) > 1e-4
        if np.sum(nz_mask) > 0:
            mape_pct = float(np.mean(abs_errors[nz_mask] / np.abs(yt[nz_mask])) * 100.0)
        else:
            mape_pct = float("nan")

        return {
            "sample_count": n,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "mean_bias": round(mean_bias, 4),
            "mape_pct": round(mape_pct, 2),
        }

    @classmethod
    def calculate_field_bootstrap_ci(
        cls,
        field_ids: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bootstraps: int = 500,
        alpha: float = 0.05,
        random_seed: int = 42,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Computes 95% bootstrap confidence intervals with resampling at the
        independent FIELD level to respect spatial autocorrelation.
        """
        rng = np.random.RandomState(random_seed)
        unique_fields = np.unique(field_ids)
        n_fields = len(unique_fields)

        if n_fields < 2:
            # Cannot bootstrap cluster with fewer than 2 independent fields
            base_metrics = cls.calculate_regression_metrics(y_true, y_pred)
            return {
                "mae_ci": (base_metrics["mae"], base_metrics["mae"]),
                "rmse_ci": (base_metrics["rmse"], base_metrics["rmse"]),
                "r2_ci": (base_metrics["r2"], base_metrics["r2"]),
                "bootstrap_resamples": 0,
                "note": "Field-level bootstrap unavailable with fewer than 2 fields.",
            }

        boot_maes = []
        boot_rmses = []
        boot_r2s = []

        for _ in range(n_bootstraps):
            # Resample fields with replacement
            sampled_fields = rng.choice(unique_fields, size=n_fields, replace=True)
            indices = []
            for f in sampled_fields:
                indices.extend(np.where(field_ids == f)[0])

            if len(indices) == 0:
                continue

            sub_yt = y_true[indices]
            sub_yp = y_pred[indices]

            m = cls.calculate_regression_metrics(sub_yt, sub_yp)
            if not math.isnan(m["mae"]):
                boot_maes.append(m["mae"])
                boot_rmses.append(m["rmse"])
                boot_r2s.append(m["r2"])

        if len(boot_maes) < 10:
            base_metrics = cls.calculate_regression_metrics(y_true, y_pred)
            return {
                "mae_ci": (base_metrics["mae"], base_metrics["mae"]),
                "rmse_ci": (base_metrics["rmse"], base_metrics["rmse"]),
                "r2_ci": (base_metrics["r2"], base_metrics["r2"]),
                "bootstrap_resamples": len(boot_maes),
            }

        lower_pct = (alpha / 2.0) * 100.0
        upper_pct = (1.0 - alpha / 2.0) * 100.0

        return {
            "mae_ci": (
                round(float(np.percentile(boot_maes, lower_pct)), 4),
                round(float(np.percentile(boot_maes, upper_pct)), 4),
            ),
            "rmse_ci": (
                round(float(np.percentile(boot_rmses, lower_pct)), 4),
                round(float(np.percentile(boot_rmses, upper_pct)), 4),
            ),
            "r2_ci": (
                round(float(np.percentile(boot_r2s, lower_pct)), 4),
                round(float(np.percentile(boot_r2s, upper_pct)), 4),
            ),
            "bootstrap_resamples": len(boot_maes),
        }

