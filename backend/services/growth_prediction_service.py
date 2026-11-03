"""
Crop Growth & Trend Prediction Service (Phase 14).

Orchestrates:
- Multi-temporal Sentinel-2 optical and Sentinel-1 SAR observations retrieval
- Chronological time-series cleaning and leakage-free feature extraction
- Trend classification with configurable agronomic thresholds
- Multi-horizon forecasting using modular baselines (Last Observation, Moving Average, Linear Trend, Seasonal Naive)
- Chronological walk-forward validation and calibrated prediction intervals
- Multi-period historical comparison with transparent scientific disclaimers
"""

import os
import math
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union, Type

import numpy as np

from services.historical_service import (
    historical_monitoring_service,
    HistoricalMonitoringService,
    HistoricalMonitoringError,
)
from services.fusion_service import (
    multimodal_fusion_service,
    MultimodalFusionService,
)
from ml.growth_prediction.schema import (
    SUPPORTED_INDICES,
    OPTICAL_INDICES,
    SAR_INDICES,
    PHYSICAL_BOUNDS,
    TREND_INCREASING,
    TREND_DECREASING,
    TREND_STABLE,
    TREND_INSUFFICIENT,
    DEFAULT_TREND_THRESHOLDS,
    FORECAST_HORIZONS,
    GrowthPredictionResponse,
)
from ml.growth_prediction.forecaster import (
    BaseForecaster,
    LastObservationForecaster,
    MovingAverageForecaster,
    LinearTrendForecaster,
    SeasonalNaiveForecaster,
    walk_forward_validate,
    compute_prediction_intervals,
)

class GrowthPredictionError(Exception):
    """Base exception for growth prediction service errors."""
    pass

class InsufficientObservationsError(GrowthPredictionError):
    """Raised when historical observations are inadequate for forecasting."""
    pass


class GrowthPredictionService:
    """
    Crop Growth & Trend Prediction Service.
    
    Provides time-series trend analysis, walk-forward validated baselines,
    and multi-horizon projections for optical and radar canopy indicators.
    """

    def __init__(
        self,
        historical_svc: Optional[HistoricalMonitoringService] = None,
        fusion_svc: Optional[MultimodalFusionService] = None,
    ):
        self.historical_svc = historical_svc or historical_monitoring_service
        self.fusion_svc = fusion_svc or multimodal_fusion_service

    @property
    def is_demo_mode(self) -> bool:
        demo_env = os.getenv("DEMO_MODE", "true").lower()
        return demo_env in ("1", "true", "yes")

    async def predict_growth_and_trends(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        forecast_horizon: str = "medium",
        selected_indices: Optional[List[str]] = None,
        model_type: str = "linear_trend",
        max_cloud_coverage: float = 35.0,
        aoi_area_km2: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end crop growth trend analysis and forecasting:
        1. Retrieves historical multi-temporal optical and SAR observations.
        2. Cleans, deduplicates, and sorts observations chronologically.
        3. Extracts sequential time-series features with strict temporal leakage prevention.
        4. Calculates trend directions and empirical rates of change.
        5. Computes realistic revisit cadence to determine future forecast dates.
        6. Runs walk-forward validated forecasting and prediction intervals per index.
        7. Assembles multi-period historical comparisons and scientific disclaimers.
        """
        # 1. Parse and sanitize requested indices and horizon
        if selected_indices is None or len(selected_indices) == 0:
            active_indices = ["NDVI", "NDMI", "NDRE", "EVI"]
        else:
            active_indices = [idx.upper() for idx in selected_indices if idx.upper() in SUPPORTED_INDICES]
            if len(active_indices) == 0:
                active_indices = ["NDVI", "NDMI", "NDRE", "EVI"]

        horizon_key = forecast_horizon.lower() if forecast_horizon.lower() in FORECAST_HORIZONS else "medium"
        horizon_info = FORECAST_HORIZONS[horizon_key]
        n_steps = horizon_info["steps"]

        warnings: List[str] = []
        limitations: List[str] = []

        # 2. Retrieve multi-temporal historical observations
        try:
            hist_res = await self.historical_svc.get_historical_time_series(
                geometry=geometry,
                start_date=start_date,
                end_date=end_date,
                max_cloud_coverage=max_cloud_coverage,
                aoi_area_km2=aoi_area_km2,
            )
            raw_observations = hist_res.get("observations", [])
            is_demo = hist_res.get("is_demo", self.is_demo_mode)
        except Exception as e:
            raise GrowthPredictionError(f"Failed to retrieve historical observations: {str(e)}")

        # 3. Clean, deduplicate, and chronologically sort observations
        cleaned_observations = self._clean_and_sort_observations(raw_observations)

        if len(cleaned_observations) < 2:
            return self._build_insufficient_data_response(
                cleaned_observations=cleaned_observations,
                start_date=start_date,
                end_date=end_date,
                is_demo=is_demo,
                reason="At least 2 valid historical observations are required to determine a trajectory.",
            )

        # 4. Feature extraction with strict no-leakage sequential processing
        enriched_observations = self._extract_time_series_features(cleaned_observations)

        # 5. Trend analysis for each requested index
        trend_statistics = self._compute_trend_statistics(enriched_observations, active_indices)

        # 6. Determine future forecast dates based on observed satellite cadence
        forecast_dates, median_cadence_days = self._determine_forecast_schedule(
            enriched_observations, n_steps=n_steps
        )

        if median_cadence_days > 20:
            warnings.append(
                f"Historical observation cadence is sparse (median {median_cadence_days} days). "
                "Forecast uncertainty increases with irregular sampling intervals."
            )

        # 7. Execute walk-forward validation and forecasting per index
        forecast_values: Dict[str, List[float]] = {}
        prediction_intervals: Dict[str, List[Dict[str, float]]] = {}
        validation_metrics: Dict[str, Dict[str, Any]] = {}
        active_models: Dict[str, str] = {}

        for idx_name in active_indices:
            idx_series = self._extract_index_series(enriched_observations, idx_name)
            valid_vals = idx_series["values"][~np.isnan(idx_series["values"])]

            # Check for SAR insufficiency
            if idx_name in SAR_INDICES:
                if len(valid_vals) < 3:
                    warnings.append(
                        f"Insufficient Sentinel-1 SAR observations for {idx_name} (only {len(valid_vals)} valid passes). "
                        f"SAR forecasting for {idx_name} is withheld without value fabrication."
                    )
                    continue

            # Check optical minimum observations
            if len(valid_vals) < 2:
                warnings.append(
                    f"Insufficient observations to forecast {idx_name} (only {len(valid_vals)} valid measurements)."
                )
                continue

            # Evaluate baseline models via walk-forward validation
            model_evals = self._evaluate_forecasters(
                dates=idx_series["dates"],
                values=idx_series["values"],
                times_days=idx_series["times_days"],
                index_name=idx_name,
            )

            # Choose requested or best model
            chosen_forecaster_cls, chosen_model_name = self._select_forecaster(
                requested_model_type=model_type,
                model_evals=model_evals,
                index_name=idx_name,
                n_valid=len(valid_vals),
            )
            active_models[idx_name] = chosen_model_name

            # Fit chosen model on entire historical series up to date T
            forecaster = chosen_forecaster_cls(index_name=idx_name)
            forecaster.fit(
                times_days=idx_series["times_days"],
                values=idx_series["values"],
                dates=idx_series["dates"],
            )

            # Project into future dates
            future_times_days = np.array(
                [idx_series["times_days"][-1] + (h + 1) * median_cadence_days for h in range(n_steps)],
                dtype=np.float64,
            )
            preds = forecaster.predict(future_times_days, future_dates=forecast_dates)
            f_vals = [round(float(v), 4) for v in preds]
            forecast_values[idx_name] = f_vals

            # Calculate calibrated prediction intervals using walk-forward residual error
            eval_record = model_evals.get(chosen_model_name, {})
            residual_std = eval_record.get("residual_std")
            
            # If walk-forward couldn't compute residual_std (e.g. series too short), fallback to series std dev
            if residual_std is None or math.isnan(residual_std):
                if len(valid_vals) >= 3:
                    residual_std = float(np.std(valid_vals, ddof=1)) * 0.75
                else:
                    residual_std = None

            intervals = compute_prediction_intervals(
                forecast_values=f_vals,
                residual_std=residual_std,
                index_name=idx_name,
                confidence_level=0.80,
            )
            if intervals:
                prediction_intervals[idx_name] = intervals

            validation_metrics[idx_name] = {
                "selected_model": chosen_model_name,
                "all_models_evaluated": model_evals,
                "residual_std": residual_std,
                "prediction_intervals_available": intervals is not None,
            }

        # 8. Multi-period historical comparison (Prior vs Current vs Forecast)
        historical_comparison = self._build_historical_comparison(
            enriched_observations, forecast_values, active_indices
        )

        # 9. Assemble data quality overview
        data_quality = self._summarize_data_quality(enriched_observations)

        # 10. Disclaimers & Limitations
        if is_demo:
            warnings.insert(
                0,
                "DEMO DATA — SYNTHETIC FORECAST. Projections reflect synthetic canopy curves and are not calibrated against in-situ ground-truth data."
            )

        limitations.extend([
            "Vegetation indicator forecasts project empirical satellite radiometric and active microwave trajectories; they do not guarantee future harvest yield or biological disease resistance.",
            "Weather anomalies, unpredicted precipitation, or sudden pest infestations will deviate actual trajectories from statistical baselines.",
            "SAR backscatter forecasts reflect surface structure and roughness; dielectric shifts caused by recent precipitation or irrigation can alter backscatter independently of biomass changes."
        ])

        disclaimer = (
            "Crop growth forecasts represent statistical time-series projections of satellite-derived vegetation "
            "and microwave reflectance indices (NDVI, NDMI, NDRE, EVI, VV, VH). They do not constitute guaranteed agricultural "
            "outcomes, yield promises, or proof of pest/disease immunity. Always corroborate projections with in-field agronomist inspection."
        )

        response_data = {
            "status": "success",
            "assessment_status": "ASSESSMENT_AVAILABLE",
            "is_demo": is_demo,
            "model_name": f"MultimodalGrowthPredictor_{model_type}",
            "model_version": "v1.0.0",
            "forecast_horizon": horizon_key,
            "forecast_horizon_label": horizon_info["label"],
            "forecast_dates": forecast_dates,
            "forecast_values": forecast_values,
            "prediction_intervals": prediction_intervals,
            "trend_statistics": trend_statistics,
            "historical_observations": enriched_observations,
            "validation_metrics": validation_metrics,
            "data_quality": data_quality,
            "observation_count": len(enriched_observations),
            "historical_comparison": historical_comparison,
            "warnings": warnings,
            "limitations": limitations,
            "scientific_disclaimer": disclaimer,
        }

        return self._sanitize_for_json(response_data)

    # ---------------------------------------------------------
    # Internal Pipeline Steps
    # ---------------------------------------------------------

    def _clean_and_sort_observations(
        self,
        observations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Deduplicates observations on identical dates (selecting highest valid pixel percentage)
        and sorts chronologically.
        """
        if not observations:
            return []

        dedup_map: Dict[str, Dict[str, Any]] = {}
        for obs in observations:
            date_str = (obs.get("date") or obs.get("datetime") or "")[:10]
            if not date_str:
                continue

            valid_pct = float(obs.get("valid_pixel_percentage") or obs.get("valid_percentage") or 100.0)

            if date_str not in dedup_map:
                dedup_map[date_str] = obs
            else:
                existing_valid = float(
                    dedup_map[date_str].get("valid_pixel_percentage")
                    or dedup_map[date_str].get("valid_percentage")
                    or 0.0
                )
                if valid_pct > existing_valid:
                    dedup_map[date_str] = obs

        sorted_list = sorted(dedup_map.values(), key=lambda x: (x.get("date") or x.get("datetime") or "")[:10])
        return sorted_list

    def _extract_time_series_features(
        self,
        observations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extracts index values, quality metrics, and sequential lagged features.
        Strictly prevents temporal leakage: step t uses information solely from steps <= t.
        """
        enriched: List[Dict[str, Any]] = []

        # Helper to get float mean
        def _get_idx_val(obs: Dict[str, Any], key: str) -> Optional[float]:
            k_lower = key.lower()
            if k_lower in obs:
                val = obs[k_lower]
                if isinstance(val, dict):
                    v = val.get("mean")
                    return float(v) if v is not None and not math.isnan(float(v)) else None
                elif isinstance(val, (int, float)):
                    return float(val) if not math.isnan(float(val)) else None

            # Check SAR structure
            sar = obs.get("sar")
            if isinstance(sar, dict) and k_lower in sar:
                val = sar[k_lower]
                if isinstance(val, dict):
                    v = val.get("mean")
                    return float(v) if v is not None and not math.isnan(float(v)) else None
                elif isinstance(val, (int, float)):
                    return float(val) if not math.isnan(float(val)) else None

            return None

        # Track rolling historical windows for each index
        index_histories: Dict[str, List[float]] = {idx: [] for idx in SUPPORTED_INDICES}

        for i, obs in enumerate(observations):
            date_str = (obs.get("date") or obs.get("datetime") or "")[:10]
            try:
                obs_dt = datetime.date.fromisoformat(date_str)
                doy = obs_dt.timetuple().tm_yday
            except Exception:
                obs_dt = None
                doy = None

            cloud_cov = float(obs.get("cloud_coverage") or 0.0)
            valid_pct = float(obs.get("valid_pixel_percentage") or obs.get("valid_percentage") or 100.0)
            nodata_pct = max(0.0, round(100.0 - valid_pct, 1))

            sar_obj = obs.get("sar")
            sar_gap = sar_obj.get("date_difference_days") if isinstance(sar_obj, dict) else None

            # Calculate interval from previous observation
            if i > 0 and obs_dt and enriched[i - 1].get("datetime_obj"):
                interval_days = max(1, (obs_dt - enriched[i - 1]["datetime_obj"]).days)
            else:
                interval_days = None

            indices_dict: Dict[str, Optional[float]] = {}
            sequential_features: Dict[str, Dict[str, Any]] = {}

            for idx in SUPPORTED_INDICES:
                val = _get_idx_val(obs, idx)
                indices_dict[idx] = val

                prev_val = index_histories[idx][-1] if len(index_histories[idx]) > 0 else None
                if val is not None:
                    index_histories[idx].append(val)

                # Sequential metrics
                abs_change = (val - prev_val) if (val is not None and prev_val is not None) else None
                rate_per_day = (abs_change / interval_days) if (abs_change is not None and interval_days) else None

                # Rolling stats (window k=3)
                window = index_histories[idx][-3:]
                roll_mean = float(np.mean(window)) if len(window) > 0 else None
                roll_std = float(np.std(window)) if len(window) > 1 else 0.0

                sequential_features[idx] = {
                    "previous_valid": prev_val,
                    "absolute_change": round(abs_change, 4) if abs_change is not None else None,
                    "rate_per_day": round(rate_per_day, 5) if rate_per_day is not None else None,
                    "rolling_mean_3": round(roll_mean, 4) if roll_mean is not None else None,
                    "rolling_std_3": round(roll_std, 4) if roll_std is not None else None,
                }

            rec = {
                "observation_id": obs.get("observation_id", f"obs_{i}"),
                "date": date_str,
                "datetime_obj": obs_dt,
                "day_of_year": doy,
                "indices": indices_dict,
                "cloud_coverage": cloud_cov,
                "valid_pixel_percentage": valid_pct,
                "nodata_percentage": nodata_pct,
                "temporal_matching_gap_days": sar_gap,
                "observation_interval_days": interval_days,
                "quality_status": obs.get("quality", "Good" if valid_pct >= 75.0 else "Fair"),
                "sequential_features": sequential_features,
            }
            enriched.append(rec)

        return enriched

    def _compute_trend_statistics(
        self,
        observations: List[Dict[str, Any]],
        indices: List[str],
    ) -> Dict[str, Any]:
        """
        Computes trend slope, net change, and classification for each index.
        """
        stats_dict: Dict[str, Any] = {}

        for idx in indices:
            pairs: List[Tuple[int, float]] = []  # (elapsed_days, value)
            first_dt = None

            for obs in observations:
                val = obs["indices"].get(idx)
                dt = obs.get("datetime_obj")
                if val is not None and dt is not None:
                    if first_dt is None:
                        first_dt = dt
                    elapsed = (dt - first_dt).days
                    pairs.append((elapsed, val))

            if len(pairs) < 2:
                stats_dict[idx] = {
                    "index_name": idx,
                    "current_value": pairs[-1][1] if pairs else None,
                    "first_value": pairs[0][1] if pairs else None,
                    "absolute_change": None,
                    "relative_change_pct": None,
                    "trend_slope_per_day": None,
                    "trend_direction": TREND_INSUFFICIENT,
                    "observation_count": len(pairs),
                    "time_span_days": 0,
                    "description": "Insufficient observations to establish a mathematical trend.",
                }
                continue

            first_val = pairs[0][1]
            latest_val = pairs[-1][1]
            abs_change = latest_val - first_val
            total_days = max(1, pairs[-1][0] - pairs[0][0])

            # Relative change (for optical indices where first_val > 0.05)
            if abs(first_val) > 0.05 and idx in OPTICAL_INDICES:
                rel_change_pct = round((abs_change / abs(first_val)) * 100.0, 2)
            else:
                rel_change_pct = None

            # OLS slope
            t_arr = np.array([p[0] for p in pairs], dtype=np.float64)
            v_arr = np.array([p[1] for p in pairs], dtype=np.float64)
            cov = np.cov(t_arr, v_arr)[0, 1] if len(pairs) > 1 else 0.0
            var_t = np.var(t_arr)
            slope = float(cov / var_t) if var_t > 1e-4 else (abs_change / total_days)

            # Categorization based on documented thresholds
            thresholds = DEFAULT_TREND_THRESHOLDS["sar" if idx in SAR_INDICES else "optical"]
            min_delta = thresholds["min_delta"]
            min_slope = thresholds["min_slope_per_day"]
            min_obs = int(thresholds["min_observations"])

            if len(pairs) < min_obs:
                direction = TREND_INSUFFICIENT
                desc = f"Fewer than {min_obs} valid observations; trajectory confidence is low."
            elif abs_change > min_delta and slope > min_slope:
                direction = TREND_INCREASING
                desc = f"{idx} demonstrates upward momentum ({abs_change:+.3f} net, {slope:+.4f}/day)."
            elif abs_change < -min_delta and slope < -min_slope:
                direction = TREND_DECREASING
                desc = f"{idx} demonstrates downward trajectory ({abs_change:+.3f} net, {slope:+.4f}/day)."
            else:
                direction = TREND_STABLE
                desc = f"{idx} displays relative stability within ±{min_delta} variance."

            stats_dict[idx] = {
                "index_name": idx,
                "current_value": round(latest_val, 4),
                "first_value": round(first_val, 4),
                "absolute_change": round(abs_change, 4),
                "relative_change_pct": rel_change_pct,
                "trend_slope_per_day": round(slope, 5),
                "trend_direction": direction,
                "observation_count": len(pairs),
                "time_span_days": total_days,
                "description": desc,
            }

        return stats_dict

    def _determine_forecast_schedule(
        self,
        observations: List[Dict[str, Any]],
        n_steps: int,
    ) -> Tuple[List[str], int]:
        """
        Calculates median satellite revisit cadence from historical observations
        and determines future forecast dates.
        """
        intervals: List[int] = []
        last_dt = None

        for obs in observations:
            dt = obs.get("datetime_obj")
            if dt:
                if last_dt is not None:
                    diff = (dt - last_dt).days
                    if diff > 0:
                        intervals.append(diff)
                last_dt = dt

        if not intervals:
            median_cadence = 6  # Nominal Sentinel-2 5-6 day revisit
        else:
            median_cadence = int(round(float(np.median(intervals))))
            median_cadence = max(4, min(14, median_cadence))

        if last_dt is None:
            last_dt = datetime.date.today()

        forecast_dates: List[str] = []
        for step in range(1, n_steps + 1):
            future_dt = last_dt + datetime.timedelta(days=step * median_cadence)
            forecast_dates.append(future_dt.isoformat())

        return forecast_dates, median_cadence

    def _extract_index_series(
        self,
        observations: List[Dict[str, Any]],
        index_name: str,
    ) -> Dict[str, Any]:
        """
        Extracts synchronized date, times (elapsed days), and value arrays for a specific index.
        """
        dates: List[str] = []
        times_days: List[float] = []
        values: List[float] = []

        first_dt = None
        for obs in observations:
            dt = obs.get("datetime_obj")
            if dt is None:
                continue
            if first_dt is None:
                first_dt = dt

            dates.append(obs["date"])
            times_days.append(float((dt - first_dt).days))
            val = obs["indices"].get(index_name)
            values.append(val if val is not None else np.nan)

        return {
            "dates": dates,
            "times_days": np.array(times_days, dtype=np.float64),
            "values": np.array(values, dtype=np.float64),
        }

    def _evaluate_forecasters(
        self,
        dates: List[str],
        values: np.ndarray,
        times_days: np.ndarray,
        index_name: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Executes walk-forward validation across all baseline forecasters for a given index.
        """
        evals: Dict[str, Dict[str, Any]] = {}

        candidate_forecasters = [
            LastObservationForecaster,
            MovingAverageForecaster,
            LinearTrendForecaster,
            SeasonalNaiveForecaster,
        ]

        for forecaster_cls in candidate_forecasters:
            res = walk_forward_validate(
                dates=dates,
                values=values,
                times_days=times_days,
                forecaster_cls=forecaster_cls,
                index_name=index_name,
                min_train_size=3,
                horizon_steps=1,
            )
            evals[res["model_name"]] = res

        return evals

    def _select_forecaster(
        self,
        requested_model_type: str,
        model_evals: Dict[str, Dict[str, Any]],
        index_name: str,
        n_valid: int,
    ) -> Tuple[Type[BaseForecaster], str]:
        """
        Maps user model preference or best walk-forward validated model to an active class.
        """
        mapping = {
            "last_observation": (LastObservationForecaster, "LastObservation_Naive"),
            "moving_average": (MovingAverageForecaster, "MovingAverage"),
            "linear_trend": (LinearTrendForecaster, "LinearTrend"),
            "seasonal_naive": (SeasonalNaiveForecaster, "SeasonalNaive"),
        }

        req_clean = requested_model_type.lower()

        if req_clean == "best_validated" and model_evals:
            # Pick valid candidate with lowest walk-forward RMSE
            best_model_name = "LinearTrend"
            best_rmse = float("inf")

            for m_name, metrics in model_evals.items():
                rmse = metrics.get("rmse")
                if rmse is not None and rmse < best_rmse:
                    best_rmse = rmse
                    best_model_name = m_name

            for key, (cls_obj, name_str) in mapping.items():
                if name_str == best_model_name:
                    return cls_obj, name_str
            return LinearTrendForecaster, "LinearTrend"

        if req_clean in mapping:
            cls_obj, name_str = mapping[req_clean]
            # Verify seasonal naive viability
            if req_clean == "seasonal_naive" and n_valid < 10:
                return LinearTrendForecaster, "LinearTrend"
            return cls_obj, name_str

        # Default baseline
        return LinearTrendForecaster, "LinearTrend"

    def _build_historical_comparison(
        self,
        observations: List[Dict[str, Any]],
        forecast_values: Dict[str, List[float]],
        indices: List[str],
    ) -> Dict[str, Any]:
        """
        Compares earlier historical window vs recent window vs future projection.
        """
        n = len(observations)
        mid = n // 2

        prior_obs = observations[:mid]
        current_obs = observations[mid:]

        prior_dates = [prior_obs[0]["date"], prior_obs[-1]["date"]] if prior_obs else ["N/A", "N/A"]
        current_dates = [current_obs[0]["date"], current_obs[-1]["date"]] if current_obs else ["N/A", "N/A"]

        def _calc_period_means(obs_list: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
            res = {}
            for idx in indices:
                vals = [o["indices"][idx] for o in obs_list if o["indices"].get(idx) is not None]
                res[idx] = round(float(np.mean(vals)), 4) if vals else None
            return res

        prior_means = _calc_period_means(prior_obs)
        current_means = _calc_period_means(current_obs)

        comparison_deltas: Dict[str, Optional[float]] = {}
        for idx in indices:
            if prior_means.get(idx) is not None and current_means.get(idx) is not None:
                comparison_deltas[idx] = round(current_means[idx] - prior_means[idx], 4)
            else:
                comparison_deltas[idx] = None

        return {
            "prior_period": {
                "period_label": "Prior Observation Window",
                "date_range": prior_dates,
                "observation_count": len(prior_obs),
                "mean_values": prior_means,
            },
            "current_period": {
                "period_label": "Recent Observation Window",
                "date_range": current_dates,
                "observation_count": len(current_obs),
                "mean_values": current_means,
            },
            "period_to_period_change": comparison_deltas,
            "forecast_continuation": {
                idx: {
                    "first_forecast": forecast_values[idx][0] if forecast_values.get(idx) else None,
                    "terminal_forecast": forecast_values[idx][-1] if forecast_values.get(idx) else None,
                }
                for idx in indices if idx in forecast_values
            },
        }

    def _summarize_data_quality(
        self,
        observations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregates data-quality statistics across the historical timeline.
        """
        if not observations:
            return {"status": "EMPTY"}

        valid_pcts = [o["valid_pixel_percentage"] for o in observations]
        clouds = [o["cloud_coverage"] for o in observations]
        sar_gaps = [o["temporal_matching_gap_days"] for o in observations if o["temporal_matching_gap_days"] is not None]

        mean_valid = float(np.mean(valid_pcts))
        mean_cloud = float(np.mean(clouds))
        mean_gap = float(np.mean(sar_gaps)) if sar_gaps else None

        if mean_valid >= 85.0 and mean_cloud <= 15.0:
            verdict = "HIGH_QUALITY"
        elif mean_valid >= 65.0:
            verdict = "ACCEPTABLE"
        else:
            verdict = "POOR"

        return {
            "mean_valid_pixel_percentage": round(mean_valid, 1),
            "mean_cloud_coverage": round(mean_cloud, 1),
            "mean_sar_temporal_gap_days": round(mean_gap, 1) if mean_gap is not None else None,
            "sar_matched_observations_count": len(sar_gaps),
            "total_observations_count": len(observations),
            "quality_verdict": verdict,
        }

    def _build_insufficient_data_response(
        self,
        cleaned_observations: List[Dict[str, Any]],
        start_date: str,
        end_date: str,
        is_demo: bool,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Constructs standardized graceful response when observations are wholly inadequate.
        """
        res = {
            "status": "partial",
            "assessment_status": "INSUFFICIENT_OBSERVATIONS",
            "is_demo": is_demo,
            "model_name": "GrowthPredictor_DataQualityGate",
            "model_version": "v1.0.0",
            "forecast_horizon": "none",
            "forecast_horizon_label": "None",
            "forecast_dates": [],
            "forecast_values": {},
            "prediction_intervals": {},
            "trend_statistics": {},
            "historical_observations": cleaned_observations,
            "validation_metrics": {},
            "data_quality": {"quality_verdict": "INSUFFICIENT_DATA"},
            "observation_count": len(cleaned_observations),
            "historical_comparison": {},
            "warnings": [reason],
            "limitations": [
                "Time-series forecasting cannot extrapolate with fewer than 2 valid historical acquisitions.",
                "Expand the start/end date range or increase cloud coverage tolerance to retrieve more observations."
            ],
            "scientific_disclaimer": (
                "Forecasting withheld due to insufficient historical observations. "
                "Reliable time-series trend analysis requires multi-temporal acquisitions."
            ),
        }
        return self._sanitize_for_json(res)

    def _sanitize_for_json(self, data: Any) -> Any:
        """
        Recursively replaces NaN, Inf, and datetime objects with JSON-compliant primitives.
        """
        if data is None:
            return None
        if isinstance(data, (bool, np.bool_)):
            return bool(data)
        if isinstance(data, (float, np.floating)):
            f = float(data)
            return None if math.isnan(f) or math.isinf(f) else f
        if isinstance(data, (int, np.integer)):
            return int(data)
        if isinstance(data, str):
            return data
        if isinstance(data, (datetime.date, datetime.datetime)):
            return data.isoformat()
        if isinstance(data, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in data.items() if k != "datetime_obj"}
        if isinstance(data, (list, tuple)):
            return [self._sanitize_for_json(x) for x in data]
        if isinstance(data, np.ndarray):
            return [self._sanitize_for_json(x) for x in data.tolist()]
        return str(data)

growth_prediction_service = GrowthPredictionService()
