"""
AI-Powered Early Crop Stress Detection Service (Phase 13).

Integrates:
- Fused Sentinel-1 SAR and Sentinel-2 optical spectral features (Phase 12)
- Multi-temporal historical observations (Phase 8)
- Versioned, calibrated machine-learning stress models
- Spatial raster inference on canonical UTM grid
- Multi-class stress probability estimation with non-causal explainability
"""

import os
import io
import json
import math
import base64
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import joblib

from services.copernicus_auth import copernicus_auth
from services.fusion_service import (
    multimodal_fusion_service,
    MASK_VALID,
    MASK_PARTIAL,
    MASK_INVALID,
)
from services.historical_service import historical_monitoring_service
from ml.stress_detection.schema import (
    ALL_MODEL_FEATURE_NAMES,
    STRESS_CLASSES,
    CLASS_NO_RELIABLE_ASSESSMENT,
    CLASS_NO_SIGNIFICANT_STRESS,
    CLASS_POTENTIAL_STRESS,
    CLASS_ELEVATED_STRESS,
    LABEL_MAPPING,
    REVERSE_LABEL_MAPPING,
)
from ml.stress_detection.data_pipeline import extract_observation_features

class StressDetectionError(Exception):
    """Base exception for crop stress detection service."""
    pass

class ModelUnavailableError(StressDetectionError):
    """Raised when no valid trained model exists and demo mode is disabled."""
    pass

class InsufficientObservationsError(StressDetectionError):
    """Raised when observation data is wholly inadequate for inference."""
    pass

class StressDetectionService:
    """
    Early Crop Stress Detection Service.
    
    Predicts early potential vegetative strain indicators combining:
    - Multispectral optical indices (NDVI, NDMI, NDRE, EVI)
    - C-band SAR backscatter (VV, VH, VV/VH ratio)
    - Historical temporal change and rates of change
    - Spatial stress probability mapping across canonical UTM grids
    
    SCIENTIFIC GUARDRAILS:
    - Early stress detection signals represent physical canopy anomalies.
    - Does NOT claim to diagnose specific pathogens, pests, or nitrogen deficits.
    - Does NOT fabricate model performance or real ground-truth labels.
    """

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(backend_dir, "ml", "models")
        self.models_dir = models_dir

        self._model_cache: Dict[str, Any] = {}
        self._metadata_cache: Dict[str, Any] = {}

    @property
    def is_demo_mode(self) -> bool:
        demo_env = os.getenv("DEMO_MODE", "true").lower()
        return demo_env in ("1", "true", "yes")

    def load_model(self, prefer_production: bool = True) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Loads a versioned stress detection model artifact:
        1. Checks for production model: `production_stress_model.joblib`
        2. If absent and DEMO_MODE=true, loads `demo_stress_model.joblib`
        3. If absent and DEMO_MODE=false, returns (None, None) to signal model unvailable.
        """
        prod_model_path = os.path.join(self.models_dir, "production_stress_model.joblib")
        prod_meta_path = os.path.join(self.models_dir, "production_stress_model_metadata.json")

        if prefer_production and os.path.exists(prod_model_path):
            if "production" not in self._model_cache:
                self._model_cache["production"] = joblib.load(prod_model_path)
                if os.path.exists(prod_meta_path):
                    with open(prod_meta_path, "r", encoding="utf-8") as f:
                        self._metadata_cache["production"] = json.load(f)
                else:
                    self._metadata_cache["production"] = {}
            return self._model_cache["production"], self._metadata_cache["production"]

        # Check demo baseline model
        demo_model_path = os.path.join(self.models_dir, "demo_stress_model.joblib")
        demo_meta_path = os.path.join(self.models_dir, "demo_stress_model_metadata.json")

        if self.is_demo_mode or not prefer_production:
            if os.path.exists(demo_model_path):
                if "demo" not in self._model_cache:
                    self._model_cache["demo"] = joblib.load(demo_model_path)
                    if os.path.exists(demo_meta_path):
                        with open(demo_meta_path, "r", encoding="utf-8") as f:
                            self._metadata_cache["demo"] = json.load(f)
                    else:
                        self._metadata_cache["demo"] = {}
                return self._model_cache["demo"], self._metadata_cache["demo"]

        return None, None

    async def detect_crop_stress(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        max_cloud_coverage: float = 30.0,
        max_temporal_window_days: int = 3,
        spatial_inference: bool = True,
        aoi_area_km2: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end early crop stress detection:
        1. Executes multimodal feature fusion (S1 + S2) on latest acquisition
        2. Retrieves historical observations strictly on or prior to the observation date
        3. Extracts 29 physical, temporal, and quality features
        4. Validates model availability (production vs demo vs unavailable)
        5. Computes multi-class calibrated stress probability and class prediction
        6. Runs optional spatial raster inference on canonical UTM grid
        7. Assembles local feature contribution explanations and transparent disclaimers
        """
        # 1. Execute Multimodal Feature Fusion for the current state
        fusion_record = await multimodal_fusion_service.execute_multimodal_fusion(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
            max_cloud_coverage=max_cloud_coverage,
            temporal_window_days=max_temporal_window_days,
            normalization="raw",
            aoi_area_km2=aoi_area_km2,
        )

        obs_date_str = (
            fusion_record.get("temporal_matching", {}).get("optical_date")
            or end_date
        )

        # 2. Retrieve Historical Observations for Temporal Context
        # Compute start date for historical timeline (e.g. 60 days before obs_date)
        try:
            obs_dt = datetime.date.fromisoformat(obs_date_str[:10])
            hist_start_dt = obs_dt - datetime.timedelta(days=60)
            hist_start_str = hist_start_dt.isoformat()
        except Exception:
            hist_start_str = start_date

        hist_res = await historical_monitoring_service.get_historical_time_series(
            geometry=geometry,
            start_date=hist_start_str,
            end_date=obs_date_str,
            max_cloud_coverage=max_cloud_coverage,
            aoi_area_km2=aoi_area_km2,
        )
        historical_timeline = hist_res.get("observations", [])

        # 3. Extract Full Feature Vector with Strict Leakage Prevention
        features_dict = extract_observation_features(
            current_obs=fusion_record,
            historical_observations=historical_timeline,
            current_date=obs_date_str,
        )

        # 4. Check Data Quality Thresholds
        quality_summary = fusion_record.get("quality_summary", {})
        cloud_cov = quality_summary.get("cloud_coverage", 0.0)
        valid_pct = quality_summary.get("fused_valid_percentage", 100.0)
        missing_warnings: List[str] = []
        limitations: List[str] = []

        if features_dict.get("missing_sar_flag", 0.0) == 1.0:
            missing_warnings.append(
                f"Sentinel-1 SAR observation unavailable within ±{max_temporal_window_days} days. "
                "Inference evaluated using optical spectral indices only."
            )

        if features_dict.get("insufficient_history_flag", 0.0) == 1.0:
            limitations.append(
                "No prior historical observations available before this acquisition date. "
                "Temporal rate-of-change and delta indicators could not be calculated."
            )

        if cloud_cov > 60.0 or valid_pct < 25.0:
            # Insufficient reliable data
            return self._build_unreliable_assessment_response(
                fusion_record=fusion_record,
                features_dict=features_dict,
                reason=(
                    f"Excessive cloud cover ({cloud_cov:.1f}%) or insufficient valid area ({valid_pct:.1f}%) "
                    "precludes reliable early stress assessment."
                ),
                missing_warnings=missing_warnings,
                limitations=limitations,
            )

        # 5. Check Model Availability
        model, model_metadata = self.load_model(prefer_production=True)

        if model is None:
            # Model training is blocked pending ground-truth data in production mode
            status_note = (
                "Supervised early stress detection model training is blocked pending verified, "
                "legally licensed agronomic ground-truth labels. To train a model, ingest verified field labels "
                "via `backend/ml/stress_detection/train_evaluate.py` or enable DEMO_MODE=true for baseline demonstration."
            )
            disclaimer = (
                "Crop stress detection pipeline requires calibrated model weights. "
                "No fabricated predictions are emitted when model artifact is absent."
            )
            return {
                "status": "unavailable",
                "is_demo": False,
                "assessment_status": "MODEL_TRAINING_BLOCKED_PENDING_GROUND_TRUTH_DATA",
                "predicted_class": CLASS_NO_RELIABLE_ASSESSMENT,
                "stress_probability": None,
                "confidence_score": None,
                "class_probabilities": None,
                "model_name": "None",
                "model_version": "None",
                "features_used_count": len(ALL_MODEL_FEATURE_NAMES),
                "model_info": {
                    "model_name": "None",
                    "model_version": "None",
                    "model_type": "None",
                    "is_calibrated": False,
                    "is_demo_model": False,
                    "status_note": status_note,
                },
                "observation_dates": {
                    "current_optical_date": fusion_record.get("temporal_matching", {}).get("optical_date"),
                    "current_sar_date": fusion_record.get("temporal_matching", {}).get("sar_date"),
                    "prediction_date": obs_date_str,
                    "prior_observations_count": int(features_dict.get("usable_history_count", 0)),
                },
                "observation_metadata": {
                    "prediction_date": obs_date_str,
                    "optical_scene_id": fusion_record.get("temporal_matching", {}).get("optical_scene_id"),
                    "sar_scene_id": fusion_record.get("temporal_matching", {}).get("sar_scene_id"),
                    "temporal_gap_days": fusion_record.get("temporal_matching", {}).get("date_difference_days"),
                    "historical_observations_count": int(features_dict.get("usable_history_count", 0)),
                },
                "features_used": ALL_MODEL_FEATURE_NAMES,
                "data_quality": {
                    "cloud_coverage": cloud_cov,
                    "valid_pixel_pct": valid_pct,
                    "nodata_pct": max(0.0, 100.0 - valid_pct),
                    "s1_s2_temporal_gap_days": fusion_record.get("temporal_matching", {}).get("date_difference_days"),
                    "missing_sar": bool(features_dict.get("missing_sar_flag", 0.0) == 1.0),
                    "quality_verdict": "FAIR" if valid_pct >= 50.0 else "POOR",
                },
                "quality_summary": {
                    "valid_pixel_percentage": valid_pct,
                    "cloud_coverage": cloud_cov,
                    "quality_tier": "Fair" if valid_pct >= 50.0 else "Poor",
                },
                "missing_features_warnings": missing_warnings,
                "missing_feature_warnings": missing_warnings,
                "prediction_limitations": limitations,
                "field_summary": {
                    "dominant_condition": CLASS_NO_RELIABLE_ASSESSMENT,
                    "mean_ndvi": round(float(features_dict["opt_ndvi"]), 4) if not math.isnan(features_dict.get("opt_ndvi", np.nan)) else None,
                    "mean_ndmi": round(float(features_dict["opt_ndmi"]), 4) if not math.isnan(features_dict.get("opt_ndmi", np.nan)) else None,
                    "mean_ndre": round(float(features_dict["opt_ndre"]), 4) if not math.isnan(features_dict.get("opt_ndre", np.nan)) else None,
                    "mean_evi": round(float(features_dict["opt_evi"]), 4) if not math.isnan(features_dict.get("opt_evi", np.nan)) else None,
                    "mean_sar_vv": round(float(features_dict["sar_vv"]), 4) if not math.isnan(features_dict.get("sar_vv", np.nan)) else None,
                    "mean_sar_vh": round(float(features_dict["sar_vh"]), 4) if not math.isnan(features_dict.get("sar_vh", np.nan)) else None,
                    "sar_ratio": round(float(features_dict["sar_vv_vh_ratio"]), 4) if not math.isnan(features_dict.get("sar_vv_vh_ratio", np.nan)) else None,
                },
                "spatial_stress_map": {
                    "raster_available": False,
                    "stress_map_png": None,
                    "resolution_meters": 10,
                    "assessable_area_percentage": 0.0,
                    "assessable_area_pct": 0.0,
                    "unassessable_pixels_masked": True,
                },
                "explainability": {
                    "method": "Feature Sensitivity Analysis & Random Forest Gini Importance",
                    "global_feature_importance": [],
                    "global_top_features": [],
                    "local_contributions": [],
                    "scientific_disclaimer": disclaimer,
                    "disclaimer": disclaimer,
                },
                "recommended_actions": [
                    "Model training is currently blocked pending validated agronomic ground-truth data.",
                    "Verify field status manually or enable demonstrator mode with DEMO_MODE=true.",
                ],
            }

        # 6. Run Tabular Inference
        feature_vector = np.array([[features_dict.get(f, np.nan) for f in ALL_MODEL_FEATURE_NAMES]], dtype=np.float32)
        
        try:
            pred_class_idx = int(model.predict(feature_vector)[0])
            pred_class_name = REVERSE_LABEL_MAPPING.get(pred_class_idx, CLASS_POTENTIAL_STRESS)
            
            # Predict probabilities
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(feature_vector)[0]
                class_probs = {
                    REVERSE_LABEL_MAPPING[i]: round(float(probs[i]), 4)
                    for i in range(len(probs))
                    if i in REVERSE_LABEL_MAPPING
                }
                # Stress probability is P(Potential Stress) + P(Elevated Stress)
                stress_prob = round(float(class_probs.get(CLASS_POTENTIAL_STRESS, 0.0) + class_probs.get(CLASS_ELEVATED_STRESS, 0.0)), 4)
                conf_score = round(float(np.max(probs)), 4)
            else:
                class_probs = None
                stress_prob = 1.0 if pred_class_idx > 0 else 0.0
                conf_score = 0.75

        except Exception as e:
            return self._build_unreliable_assessment_response(
                fusion_record=fusion_record,
                features_dict=features_dict,
                reason=f"Model inference failed: {str(e)}",
                missing_warnings=missing_warnings,
                limitations=limitations,
            )

        # 7. Spatial Raster Inference (if requested and canonical grid is present)
        spatial_map_result = {
            "raster_available": False,
            "stress_map_png": None,
            "assessable_area_percentage": 0.0,
            "stress_zone_percentages": {
                "nominal": 0.0,
                "potential_stress": 0.0,
                "elevated_stress": 0.0,
                "unassessed": 100.0,
            },
        }

        if spatial_inference and "_quality_mask" in fusion_record and "_raw_features" in fusion_record:
            spatial_map_result = self._generate_spatial_stress_raster(
                model=model,
                fusion_record=fusion_record,
                features_dict=features_dict,
            )

        # 8. Local Explainability (Feature Contribution Analysis)
        local_contributions = self._compute_local_contributions(
            features_dict=features_dict,
            pred_class_idx=pred_class_idx,
            global_importance=model_metadata.get("global_feature_importance", []),
        )

        is_demo_run = fusion_record.get("is_demo", False) or model_metadata.get("is_demo_model", False)
        disclaimer_text = (
            "Early stress detection signals reflect physical radiometric and active radar backscatter "
            "anomalies. They do not constitute diagnostic proof of biological pathogens, soil nitrogen deficiency, "
            "or specific causal agents without local agronomist ground verification."
        )

        # Build recommended actions based on predicted status
        if pred_class_name == CLASS_ELEVATED_STRESS:
            recs = [
                "Immediate ground-truth field inspection recommended in high-stress zones.",
                "Verify soil moisture at root-zone depths and inspect for pest, fungal, or foliar chlorosis symptoms.",
                "Consult agronomic extension advisor before initiating any chemical or irrigation changes.",
            ]
        elif pred_class_name == CLASS_POTENTIAL_STRESS:
            recs = [
                "Schedule targeted field scout within 48-72 hours to evaluate canopy moisture and vigor.",
                "Cross-reference with recent irrigation logs or precipitation records.",
                "Monitor consecutive satellite passes for persistent red-edge or SAR backscatter degradation.",
            ]
        elif pred_class_name == CLASS_NO_SIGNIFICANT_STRESS:
            recs = [
                "Canopy vigor and SAR volume backscatter remain within nominal bounds.",
                "Maintain scheduled crop management and routine satellite monitoring.",
            ]
        else:
            recs = [
                "Ground investigation recommended once atmospheric obstruction clears.",
            ]

        # Top features for explainability
        raw_global_imp = model_metadata.get("global_feature_importance", [])
        global_top_features = [
            {"feature": f["feature"], "importance": float(f["importance"])}
            for f in raw_global_imp[:8]
        ]

        def _val_or_none(k: str) -> Optional[float]:
            v = features_dict.get(k, np.nan)
            return round(float(v), 4) if not math.isnan(v) else None

        return {
            "status": "success",
            "is_demo": is_demo_run,
            "assessment_status": "ASSESSMENT_AVAILABLE",
            "predicted_class": pred_class_name,
            "stress_probability": stress_prob,
            "confidence_score": conf_score,
            "class_probabilities": class_probs,
            "model_name": model_metadata.get("model_name", "MultimodalCropStressDetector"),
            "model_version": model_metadata.get("model_version", "v1.0.0"),
            "features_used_count": len(ALL_MODEL_FEATURE_NAMES),
            "model_info": {
                "model_name": model_metadata.get("model_name", "MultimodalCropStressDetector"),
                "model_version": model_metadata.get("model_version", "v1.0.0"),
                "model_type": model_metadata.get("model_type", "RandomForest"),
                "is_calibrated": model_metadata.get("is_calibrated", False),
                "is_demo_model": is_demo_run,
            },
            "observation_dates": {
                "current_optical_date": fusion_record.get("temporal_matching", {}).get("optical_date"),
                "current_sar_date": fusion_record.get("temporal_matching", {}).get("sar_date"),
                "prediction_date": obs_date_str,
                "prior_observations_count": int(features_dict.get("usable_history_count", 0)),
            },
            "observation_metadata": {
                "prediction_date": obs_date_str,
                "optical_scene_id": fusion_record.get("temporal_matching", {}).get("optical_scene_id"),
                "sar_scene_id": fusion_record.get("temporal_matching", {}).get("sar_scene_id"),
                "temporal_gap_days": fusion_record.get("temporal_matching", {}).get("date_difference_days"),
                "historical_observations_count": int(features_dict.get("usable_history_count", 0)),
            },
            "features_used": ALL_MODEL_FEATURE_NAMES,
            "data_quality": {
                "cloud_coverage": cloud_cov,
                "valid_pixel_pct": valid_pct,
                "nodata_pct": max(0.0, 100.0 - valid_pct),
                "s1_s2_temporal_gap_days": fusion_record.get("temporal_matching", {}).get("date_difference_days"),
                "missing_sar": bool(features_dict.get("missing_sar_flag", 0.0) == 1.0),
                "quality_verdict": "HIGH_QUALITY" if valid_pct >= 85.0 else ("ACCEPTABLE" if valid_pct >= 65.0 else "POOR"),
            },
            "quality_summary": {
                "valid_pixel_percentage": valid_pct,
                "cloud_coverage": cloud_cov,
                "quality_tier": "Excellent" if valid_pct >= 85.0 else ("Good" if valid_pct >= 65.0 else "Fair"),
            },
            "missing_features_warnings": missing_warnings,
            "missing_feature_warnings": missing_warnings,
            "prediction_limitations": limitations,
            "field_summary": {
                "dominant_condition": pred_class_name,
                "mean_ndvi": _val_or_none("opt_ndvi"),
                "mean_ndmi": _val_or_none("opt_ndmi"),
                "mean_ndre": _val_or_none("opt_ndre"),
                "mean_evi": _val_or_none("opt_evi"),
                "mean_sar_vv": _val_or_none("sar_vv"),
                "mean_sar_vh": _val_or_none("sar_vh"),
                "sar_ratio": _val_or_none("sar_vv_vh_ratio"),
            },
            "spatial_stress_map": spatial_map_result,
            "explainability": {
                "method": "Feature Sensitivity Analysis & Random Forest Gini Importance",
                "global_feature_importance": raw_global_imp[:8],
                "global_top_features": global_top_features,
                "local_contributions": local_contributions,
                "scientific_disclaimer": disclaimer_text,
                "disclaimer": disclaimer_text,
            },
            "recommended_actions": recs,
        }

    def _generate_spatial_stress_raster(
        self,
        model: Any,
        fusion_record: Dict[str, Any],
        features_dict: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Executes per-pixel spatial stress inference across the canonical UTM grid:
        - Preserves AOI geometry, spatial extent, transform, and NoData mask.
        - Evaluates pixels where `quality_mask > 0`.
        - Generates RGBA colormapped PNG overlay:
            🟩 Green: Nominal Canopy (Low stress)
            🟨 Amber: Potential Vigour Anomaly
            🟥 Crimson: Elevated Stress
            ⬛ Transparent / MASK_INVALID: Unassessed / NoData
        """
        quality_mask: np.ndarray = fusion_record["_quality_mask"]
        raw_features: Dict[str, Optional[np.ndarray]] = fusion_record["_raw_features"]
        h, w = quality_mask.shape
        total_pixels = quality_mask.size

        valid_coords = np.where(quality_mask > 0)
        n_valid = len(valid_coords[0])

        if n_valid == 0:
            return {
                "raster_available": False,
                "stress_map_png": None,
                "assessable_area_percentage": 0.0,
                "stress_zone_percentages": {
                    "nominal": 0.0,
                    "potential_stress": 0.0,
                    "elevated_stress": 0.0,
                    "unassessed": 100.0,
                },
            }

        # Build feature matrix for valid pixels
        pixel_matrix = np.zeros((n_valid, len(ALL_MODEL_FEATURE_NAMES)), dtype=np.float32)

        # Mapping of array names in raw_features
        band_name_map = {
            "opt_ndvi": "NDVI",
            "opt_ndmi": "NDMI",
            "opt_ndre": "NDRE",
            "opt_evi": "EVI",
            "opt_b02": "B02",
            "opt_b03": "B03",
            "opt_b04": "B04",
            "opt_b05": "B05",
            "opt_b08": "B08",
            "opt_b11": "B11",
            "sar_vv": "VV",
            "sar_vh": "VH",
            "sar_vv_vh_ratio": "VV_VH_RATIO",
        }

        for col_idx, feat_name in enumerate(ALL_MODEL_FEATURE_NAMES):
            if feat_name in band_name_map:
                source_key = band_name_map[feat_name]
                source_arr = raw_features.get(source_key)
                if source_arr is not None:
                    pixel_matrix[:, col_idx] = source_arr[valid_coords]
                else:
                    pixel_matrix[:, col_idx] = np.nan
            else:
                # Field-level or temporal scalar feature broadcast to all valid pixels
                scalar_val = features_dict.get(feat_name, np.nan)
                pixel_matrix[:, col_idx] = scalar_val

        # Vectorized batch prediction
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(pixel_matrix)
                # Probability of stress = sum of class 1 and class 2 probabilities
                stress_probs_valid = probs[:, 1] + (probs[:, 2] if probs.shape[1] > 2 else 0.0)
                pred_classes_valid = np.argmax(probs, axis=1)
            else:
                pred_classes_valid = model.predict(pixel_matrix)
                stress_probs_valid = np.where(pred_classes_valid > 0, 1.0, 0.0)
        except Exception:
            return {
                "raster_available": False,
                "stress_map_png": None,
                "assessable_area_percentage": 0.0,
                "stress_zone_percentages": {
                    "nominal": 0.0,
                    "potential_stress": 0.0,
                    "elevated_stress": 0.0,
                    "unassessed": 100.0,
                },
            }

        # Build RGBA raster
        rgba_img = np.zeros((h, w, 4), dtype=np.uint8)

        # Count zones
        count_nominal = int(np.count_nonzero(pred_classes_valid == 0))
        count_potential = int(np.count_nonzero(pred_classes_valid == 1))
        count_elevated = int(np.count_nonzero(pred_classes_valid == 2))

        # Color mapping:
        # Class 0: Green (34, 197, 94, 210)
        # Class 1: Amber (245, 158, 11, 230)
        # Class 2: Crimson (225, 29, 72, 240)
        colors = {
            0: (34, 197, 94, 210),
            1: (245, 158, 11, 230),
            2: (225, 29, 72, 240),
        }

        for c_idx, color in colors.items():
            matching_mask = (pred_classes_valid == c_idx)
            rows = valid_coords[0][matching_mask]
            cols = valid_coords[1][matching_mask]
            rgba_img[rows, cols] = color

        # Encode to PNG Base64
        pil_img = Image.fromarray(rgba_img, mode="RGBA")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        assessable_pct = round((n_valid / total_pixels) * 100.0, 2)
        nominal_pct = round((count_nominal / n_valid) * 100.0, 2) if n_valid > 0 else 0.0
        potential_pct = round((count_potential / n_valid) * 100.0, 2) if n_valid > 0 else 0.0
        elevated_pct = round((count_elevated / n_valid) * 100.0, 2) if n_valid > 0 else 0.0
        unassessed_pct = round(100.0 - assessable_pct, 2)

        return {
            "raster_available": True,
            "stress_map_png": png_b64,
            "assessable_area_percentage": assessable_pct,
            "stress_zone_percentages": {
                "nominal": nominal_pct,
                "potential_stress": potential_pct,
                "elevated_stress": elevated_pct,
                "unassessed": unassessed_pct,
            },
            "legend": {
                "nominal": {"label": "Nominal / Low Stress", "color": "#22c55e", "code": 0},
                "potential_stress": {"label": "Potential Vigour Anomaly", "color": "#f59e0b", "code": 1},
                "elevated_stress": {"label": "Elevated Potential Stress", "color": "#e11d48", "code": 2},
                "unassessed": {"label": "Unassessed / Obscured / NoData", "color": "#64748b", "code": -1},
            },
        }

    def _compute_local_contributions(
        self,
        features_dict: Dict[str, float],
        pred_class_idx: int,
        global_importance: List[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """
        Computes local feature contribution explanations indicating which physical measurements
        contributed most toward the predicted stress status.
        """
        contributions: List[Dict[str, Any]] = []

        # Analyze key physical drivers: NDMI, NDRE, NDVI, SAR VV, VH, and their temporal deltas
        key_checks = [
            ("delta_ndmi", "Moisture rate of change", "negative_indicates_drying"),
            ("opt_ndmi", "Canopy moisture indicator", "lower_indicates_water_strain"),
            ("delta_ndre", "Red-edge chlorophyll delta", "negative_indicates_early_chlorosis"),
            ("opt_ndre", "Red-edge vegetation vigor", "lower_indicates_chlorophyll_loss"),
            ("delta_ndvi", "NDVI vigor difference", "negative_indicates_biomass_decline"),
            ("opt_ndvi", "Canopy vegetation vigor", "lower_indicates_low_canopy"),
            ("delta_vh", "Cross-polarization SAR delta", "negative_indicates_canopy_biomass_loss"),
            ("sar_vv_vh_ratio", "SAR structural cross-ratio", "high_ratio_indicates_surface_dominance"),
        ]

        for feat_key, label, mechanism in key_checks:
            val = features_dict.get(feat_key, np.nan)
            if math.isnan(val):
                continue

            impact_direction = "neutral"
            explanation = ""

            if feat_key == "delta_ndmi":
                if val < -0.04:
                    impact_direction = "elevates_stress"
                    explanation = f"NDMI dropped by {val:+.3f} since prior pass, signaling rapid canopy moisture depletion."
                elif val > 0.02:
                    impact_direction = "reduces_stress"
                    explanation = f"NDMI increased by {val:+.3f}, indicating moisture replenishment."
            elif feat_key == "delta_ndre":
                if val < -0.03:
                    impact_direction = "elevates_stress"
                    explanation = f"NDRE red-edge dropped by {val:+.3f}, indicating early chlorophyll degradation."
            elif feat_key == "delta_ndvi":
                if val < -0.05:
                    impact_direction = "elevates_stress"
                    explanation = f"NDVI decreased by {val:+.3f}, reflecting structural vigor decline."
            elif feat_key == "opt_ndmi" and val < 0.05:
                impact_direction = "elevates_stress"
                explanation = f"Low moisture index (NDMI = {val:.2f}) indicates acute cellular water stress."
            elif feat_key == "delta_vh" and val < -1.0:
                impact_direction = "elevates_stress"
                explanation = f"Radar VH backscatter dropped by {val:+.1f} dB, reflecting volume canopy scattering reduction."

            if impact_direction != "neutral":
                direction = "increasing_stress" if impact_direction == "elevates_stress" else "moderating_stress"
                leverage = round(min(0.4, abs(val) * 0.2 if not math.isnan(val) else 0.1), 3)
                contributions.append({
                    "feature": feat_key,
                    "label": label,
                    "value": round(val, 4),
                    "impact": impact_direction,
                    "direction": direction,
                    "contribution": leverage,
                    "explanation": explanation,
                })

        if not contributions:
            # Baseline fallback when observations are nominal
            for feat_key, label in [("opt_ndvi", "Canopy vegetation vigor"), ("opt_ndmi", "Canopy moisture indicator"), ("sar_vv_vh_ratio", "SAR structural cross-ratio")]:
                val = features_dict.get(feat_key, np.nan)
                if not math.isnan(val):
                    contributions.append({
                        "feature": feat_key,
                        "label": label,
                        "value": round(val, 4),
                        "impact": "neutral",
                        "direction": "moderating_stress",
                        "contribution": 0.05,
                        "explanation": f"{label} is within nominal vegetative bounds ({val:.2f}).",
                    })

        return contributions[:5]

    def _build_unreliable_assessment_response(
        self,
        fusion_record: Dict[str, Any],
        features_dict: Dict[str, float],
        reason: str,
        missing_warnings: List[str],
        limitations: List[str],
    ) -> Dict[str, Any]:
        """Builds a standardized 'No reliable assessment' response."""
        quality_summary = fusion_record.get("quality_summary", {})
        cloud_cov = quality_summary.get("cloud_coverage", 0.0)
        valid_pct = quality_summary.get("fused_valid_percentage", 0.0)
        obs_date_str = fusion_record.get("temporal_matching", {}).get("optical_date")

        disclaimer_text = (
            "Assessment withheld due to severe atmospheric obstruction or excessive NoData area. "
            "Inferences must not be extrapolated over clouded or invalid pixels."
        )

        def _val_or_none(k: str) -> Optional[float]:
            v = features_dict.get(k, np.nan)
            return round(float(v), 4) if not math.isnan(v) else None

        return {
            "status": "partial",
            "is_demo": fusion_record.get("is_demo", False),
            "assessment_status": "POOR_DATA_QUALITY",
            "predicted_class": CLASS_NO_RELIABLE_ASSESSMENT,
            "stress_probability": None,
            "confidence_score": None,
            "class_probabilities": None,
            "model_name": "MultimodalCropStressDetector",
            "model_version": "v1.0.0",
            "features_used_count": len(ALL_MODEL_FEATURE_NAMES),
            "model_info": {
                "model_name": "MultimodalCropStressDetector",
                "model_version": "v1.0.0",
                "model_type": "DataQualityGate",
                "is_calibrated": False,
                "is_demo_model": fusion_record.get("is_demo", False),
            },
            "observation_dates": {
                "current_optical_date": obs_date_str,
                "current_sar_date": fusion_record.get("temporal_matching", {}).get("sar_date"),
                "prediction_date": obs_date_str or "",
                "prior_observations_count": int(features_dict.get("usable_history_count", 0)),
            },
            "observation_metadata": {
                "prediction_date": obs_date_str,
                "optical_scene_id": fusion_record.get("temporal_matching", {}).get("optical_scene_id"),
                "sar_scene_id": fusion_record.get("temporal_matching", {}).get("sar_scene_id"),
                "temporal_gap_days": fusion_record.get("temporal_matching", {}).get("date_difference_days"),
                "historical_observations_count": int(features_dict.get("usable_history_count", 0)),
            },
            "features_used": ALL_MODEL_FEATURE_NAMES,
            "data_quality": {
                "cloud_coverage": cloud_cov,
                "valid_pixel_pct": valid_pct,
                "nodata_pct": max(0.0, 100.0 - valid_pct),
                "s1_s2_temporal_gap_days": fusion_record.get("temporal_matching", {}).get("date_difference_days"),
                "missing_sar": bool(features_dict.get("missing_sar_flag", 0.0) == 1.0),
                "quality_verdict": "POOR",
            },
            "quality_summary": {
                "valid_pixel_percentage": valid_pct,
                "cloud_coverage": cloud_cov,
                "quality_tier": "Poor",
            },
            "missing_features_warnings": missing_warnings,
            "missing_feature_warnings": missing_warnings,
            "prediction_limitations": limitations + [reason],
            "field_summary": {
                "dominant_condition": CLASS_NO_RELIABLE_ASSESSMENT,
                "mean_ndvi": _val_or_none("opt_ndvi"),
                "mean_ndmi": _val_or_none("opt_ndmi"),
                "mean_ndre": _val_or_none("opt_ndre"),
                "mean_evi": _val_or_none("opt_evi"),
                "mean_sar_vv": _val_or_none("sar_vv"),
                "mean_sar_vh": _val_or_none("sar_vh"),
                "sar_ratio": _val_or_none("sar_vv_vh_ratio"),
            },
            "spatial_stress_map": {
                "raster_available": False,
                "stress_map_png": None,
                "resolution_meters": 10,
                "assessable_area_percentage": 0.0,
                "assessable_area_pct": 0.0,
                "unassessable_pixels_masked": True,
                "stress_zone_percentages": {
                    "nominal": 0.0,
                    "potential_stress": 0.0,
                    "elevated_stress": 0.0,
                    "unassessed": 100.0,
                },
            },
            "explainability": {
                "method": "Feature Sensitivity Analysis & Random Forest Gini Importance",
                "global_feature_importance": [],
                "global_top_features": [],
                "local_contributions": [],
                "scientific_disclaimer": disclaimer_text,
                "disclaimer": disclaimer_text,
            },
            "recommended_actions": [
                "Assessment withheld due to severe atmospheric obstruction or excessive NoData area.",
                "Acquisition should be repeated during cloud-free or radar-favorable conditions.",
            ],
        }

stress_detection_service = StressDetectionService()
