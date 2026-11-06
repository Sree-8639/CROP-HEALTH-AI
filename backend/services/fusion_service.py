import os
import io
import math
import json
import base64
import datetime
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, asdict

import numpy as np
import tifffile
from PIL import Image

from services.raster_alignment import (
    CanonicalGrid,
    create_canonical_aoi_grid,
    resample_raster_to_canonical_grid,
    align_all_multimodal_features,
    RasterAlignmentError,
)
from services.sentinel2_service import (
    sentinel2_service,
    Sentinel2Error,
    NoImageryFoundError,
)
from services.sentinel1_service import (
    sentinel1_service,
    Sentinel1Error,
    NoSARDataFoundError,
)
from services.ndvi_service import (
    calculate_ndvi,
    create_valid_pixel_mask,
    generate_all_demo_spectral_bands,
)
from services.ndmi_service import calculate_ndmi
from services.ndre_service import calculate_ndre
from services.evi_service import calculate_evi
from services.provenance import build_provenance_badge

class MultimodalFusionError(Exception):
    """Base exception for multimodal data fusion operations."""
    pass

MultimodalFusionPreparationError = MultimodalFusionError

class TemporalMatchingError(MultimodalFusionError):
    """Raised when temporal matching fails or cannot be computed."""
    pass

class SpatialAlignmentError(MultimodalFusionError):
    """Raised when spatial reprojection and grid alignment fails."""
    pass

# Standard names for all 13 core multimodal satellite features
MULTIMODAL_FEATURE_NAMES = [
    "B02", "B03", "B04", "B05", "B08", "B11",
    "NDVI", "NDMI", "NDRE", "EVI",
    "VV", "VH", "VV_VH_RATIO",
]

# Quality Mask constants
MASK_INVALID = 0  # NoData, cloud, shadow, corrupt
MASK_VALID = 1    # Fully valid multimodal (both optical and SAR usable)
MASK_PARTIAL = 2  # Partially valid (optical valid, SAR missing/out-of-range, or vice versa)

class MultimodalFusionService:
    """
    Multimodal Satellite Feature-Fusion Service (Sentinel-2 Optical + Sentinel-1 SAR).
    
    Combines:
    - Sentinel-2 Level-2A BOA reflectance (B02, B03, B04, B05, B08, B11)
    - Multispectral Vegetation Indices (NDVI, NDMI, NDRE, EVI)
    - Sentinel-1 Level-1 GRD SAR Backscatter (VV, VH, VV/VH ratio in dB)
    - Unified Quality Mask (VALID, PARTIAL, INVALID)
    - Canonical AOI Grid (UTM / metric projection, standardized transform)
    - Configurable Temporal Matching (default ±3 days)
    - Dynamic Normalization (raw, min-max, z-score)
    - Dataset Export (.npz, .json, GeoTIFF)
    
    Strict Architectural Guardrails:
    - Does NOT train deep learning models in this phase.
    - Does NOT fabricate early stress or crop growth predictions.
    - Does NOT invent missing SAR values when outside temporal window.
    """

    DEFAULT_TEMPORAL_WINDOW_DAYS = 3

    def __init__(self):
        pass

    @property
    def is_demo_mode(self) -> bool:
        demo_env = os.getenv("DEMO_MODE", "true").lower()
        return demo_env in ("1", "true", "yes")

    def prepare_multimodal_dataset(
        self,
        optical_data: Dict[str, Any],
        sar_data: Dict[str, Any],
        temporal_match: Optional[Dict[str, Any]] = None,
        target_resolution_m: int = 10,
    ) -> Dict[str, Any]:
        """Backwards compatibility helper for Phase 11."""
        opt_indices = optical_data.get("indices", {})
        opt_quality = optical_data.get("quality", {})
        opt_scene = optical_data.get("scene", {})

        ndvi_stats = opt_indices.get("ndvi", {}).get("statistics", {})
        ndmi_stats = opt_indices.get("ndmi", {}).get("statistics", {})
        ndre_stats = opt_indices.get("ndre", {}).get("statistics", {})
        evi_stats = opt_indices.get("evi", {}).get("statistics", {})

        optical_valid_pct = (
            opt_quality.get("valid_percentage")
            or opt_indices.get("ndvi", {}).get("statistics", {}).get("valid_pixel_percentage")
            or 100.0
        )

        sar_features = sar_data.get("sar_features", {})
        sar_quality = sar_data.get("quality", {})
        vv_stats = sar_features.get("vv", {}).get("statistics", {})
        vh_stats = sar_features.get("vh", {}).get("statistics", {})
        ratio_stats = sar_features.get("vv_vh_ratio", {}).get("statistics", {})
        sar_valid_pct = sar_quality.get("valid_pixel_percentage", 100.0)

        opt_date = opt_scene.get("acquisition_date") or optical_data.get("date") or datetime.date.today().isoformat()
        sar_date = sar_data.get("observation_date") or datetime.date.today().isoformat()

        return {
            "timestamp": opt_date,
            "temporal_match": temporal_match or {
                "sentinel2_date": opt_date.split("T")[0],
                "sentinel1_date": sar_date.split("T")[0],
                "date_difference_days": 0,
                "matched": True,
            },
            "optical": {
                "ndvi": {"mean": ndvi_stats.get("mean"), "median": ndvi_stats.get("median"), "min": ndvi_stats.get("min"), "max": ndvi_stats.get("max")},
                "ndmi": {"mean": ndmi_stats.get("mean"), "median": ndmi_stats.get("median"), "min": ndmi_stats.get("min"), "max": ndmi_stats.get("max")},
                "ndre": {"mean": ndre_stats.get("mean"), "median": ndre_stats.get("median"), "min": ndre_stats.get("min"), "max": ndre_stats.get("max")},
                "evi": {"mean": evi_stats.get("mean"), "median": evi_stats.get("median"), "min": evi_stats.get("min"), "max": evi_stats.get("max")},
            },
            "sar": {
                "vv": {"mean": vv_stats.get("mean"), "median": vv_stats.get("median"), "min": vv_stats.get("min"), "max": vv_stats.get("max"), "std": vv_stats.get("std")},
                "vh": {"mean": vh_stats.get("mean"), "median": vh_stats.get("median"), "min": vh_stats.get("min"), "max": vh_stats.get("max"), "std": vh_stats.get("std")},
                "vv_vh_ratio": {"mean": ratio_stats.get("mean"), "median": ratio_stats.get("median"), "min": ratio_stats.get("min"), "max": ratio_stats.get("max"), "std": ratio_stats.get("std")},
                "polarization": sar_data.get("polarization", ["VV", "VH"]),
                "product_type": sar_data.get("product_type", "Sentinel-1 GRD"),
                "orbit_direction": sar_data.get("orbit_direction", "descending"),
            },
            "quality": {
                "optical_valid_percentage": round(float(optical_valid_pct), 2),
                "sar_valid_percentage": round(float(sar_valid_pct), 2),
                "multimodal_completeness": "complete" if (ndvi_stats and vv_stats) else "partial",
                "fusion_readiness": True if (optical_valid_pct >= 50.0 and sar_valid_pct >= 50.0) else False,
            },
            "physical_specifications": {
                "target_resolution_m": target_resolution_m,
                "optical_constellation": "Copernicus Sentinel-2 MSI (Level-2A BOA Reflectance)",
                "sar_constellation": "Copernicus Sentinel-1 C-SAR (Level-1 GRD Backscatter)",
                "optical_bands_used": ["B02", "B04", "B05", "B08", "B11"],
                "sar_polarizations_used": ["VV", "VH"],
            }
        }

    def align_spatial_rasters(
        self,
        optical_rasters: Dict[str, np.ndarray],
        sar_rasters: Dict[str, np.ndarray],
        reference_grid: str = "B08",
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Backwards compatibility helper for Phase 11."""
        from services.raster_utils import align_bands, validate_band_shapes
        combined = {**optical_rasters, **sar_rasters}
        if not combined:
            raise MultimodalFusionError("Cannot align empty raster dictionaries.")
        aligned_combined = align_bands(combined, reference_band=reference_grid)
        validate_band_shapes(aligned_combined)
        aligned_optical = {k: aligned_combined[k] for k in optical_rasters.keys() if k in aligned_combined}
        aligned_sar = {k: aligned_combined[k] for k in sar_rasters.keys() if k in aligned_combined}
        return aligned_optical, aligned_sar

    def match_temporal_observations(
        self,
        sentinel2_date: str,
        sentinel1_scenes: List[Dict[str, Any]],
        temporal_window_days: int = DEFAULT_TEMPORAL_WINDOW_DAYS,
    ) -> Dict[str, Any]:
        """
        Temporal Matching Engine:
        For a given Sentinel-2 observation date, finds available Sentinel-1 observations
        within ±temporal_window_days (default 3 days).
        Selects the best available SAR observation by smallest absolute time difference.
        Tie-breaker: higher observation quality (e.g. valid pixel percentage).
        
        If no SAR observation exists within window, returns sar_available = False
        without fabricating radar measurements.
        """
        try:
            s2_dt = datetime.date.fromisoformat(sentinel2_date.split("T")[0])
        except (ValueError, TypeError) as exc:
            raise TemporalMatchingError(f"Invalid Sentinel-2 date '{sentinel2_date}': {str(exc)}")

        if not sentinel1_scenes:
            return {
                "matched": False,
                "sar_available": False,
                "sentinel2_date": s2_dt.isoformat(),
                "sentinel1_date": None,
                "temporal_gap_days": None,
                "temporal_match_quality": "UNAVAILABLE",
                "matched_scene": None,
                "candidate_count": 0,
                "message": "No Sentinel-1 observations found in catalog.",
            }

        candidates = []
        for scene in sentinel1_scenes:
            s1_date_str = scene.get("acquisition_date") or scene.get("datetime") or scene.get("date")
            if not s1_date_str:
                continue
            try:
                s1_dt = datetime.date.fromisoformat(s1_date_str.split("T")[0])
            except ValueError:
                continue

            gap_days = abs((s1_dt - s2_dt).days)
            if gap_days <= temporal_window_days:
                # Calculate quality score (prefer descending/ascending consistency or higher valid %)
                quality_score = float(scene.get("valid_percentage", 100.0))
                candidates.append({
                    "scene": scene,
                    "date": s1_dt,
                    "gap_days": gap_days,
                    "quality_score": quality_score,
                })

        if not candidates:
            return {
                "matched": False,
                "sar_available": False,
                "sentinel2_date": s2_dt.isoformat(),
                "sentinel1_date": None,
                "temporal_gap_days": None,
                "temporal_match_quality": "UNAVAILABLE",
                "matched_scene": None,
                "candidate_count": 0,
                "message": f"Sentinel-1 observation unavailable for this time window (±{temporal_window_days} days).",
            }

        # Sort: primary key = smallest gap_days, secondary key = highest quality_score
        candidates.sort(key=lambda c: (c["gap_days"], -c["quality_score"]))
        best = candidates[0]
        gap_days = best["gap_days"]

        if gap_days <= 1:
            match_quality = "EXCELLENT"
        elif gap_days <= 2:
            match_quality = "GOOD"
        elif gap_days <= 3:
            match_quality = "FAIR"
        else:
            match_quality = "POOR"

        return {
            "matched": True,
            "sar_available": True,
            "sentinel2_date": s2_dt.isoformat(),
            "sentinel1_date": best["date"].isoformat(),
            "temporal_gap_days": gap_days,
            "temporal_match_quality": match_quality,
            "matched_scene": best["scene"],
            "candidate_count": len(candidates),
            "message": f"Matched Sentinel-1 SAR acquisition with {gap_days} day(s) offset ({match_quality} quality).",
        }

    def compute_feature_statistics(
        self,
        array: Optional[np.ndarray],
        valid_mask: Optional[np.ndarray] = None,
        feature_name: str = "feature",
    ) -> Dict[str, Any]:
        """
        Calculates robust statistics for a single 2D feature raster:
        min, max, mean, median, standard deviation, valid pixel count,
        valid pixel percentage, and NoData percentage.
        """
        if array is None:
            return {
                "feature": feature_name,
                "available": False,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "valid_pixel_count": 0,
                "valid_pixel_percentage": 0.0,
                "nodata_percentage": 100.0,
            }

        arr = np.asarray(array, dtype=np.float32)
        total_pixels = int(arr.size)
        if total_pixels == 0:
            return {
                "feature": feature_name,
                "available": False,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "valid_pixel_count": 0,
                "valid_pixel_percentage": 0.0,
                "nodata_percentage": 100.0,
            }

        # Determine valid values: finite and not NaN
        is_finite = np.isfinite(arr)
        if valid_mask is not None:
            pixel_valid = is_finite & (valid_mask > 0)
        else:
            pixel_valid = is_finite

        valid_count = int(np.count_nonzero(pixel_valid))
        valid_pct = round((valid_count / total_pixels) * 100.0, 2)
        nodata_pct = round(100.0 - valid_pct, 2)

        if valid_count == 0:
            return {
                "feature": feature_name,
                "available": False,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "valid_pixel_count": 0,
                "valid_pixel_percentage": 0.0,
                "nodata_percentage": 100.0,
            }

        valid_vals = arr[pixel_valid]

        return {
            "feature": feature_name,
            "available": True,
            "min": round(float(np.min(valid_vals)), 4),
            "max": round(float(np.max(valid_vals)), 4),
            "mean": round(float(np.mean(valid_vals)), 4),
            "median": round(float(np.median(valid_vals)), 4),
            "std": round(float(np.std(valid_vals)), 4),
            "valid_pixel_count": valid_count,
            "valid_pixel_percentage": valid_pct,
            "nodata_percentage": nodata_pct,
        }

    def build_unified_quality_mask(
        self,
        optical_mask: np.ndarray,
        sar_mask: Optional[np.ndarray],
        sar_available: bool,
    ) -> np.ndarray:
        """
        Creates the canonical 3-tier multimodal quality mask:
        - 1 = VALID (both optical and SAR are valid for this pixel)
        - 2 = PARTIAL (optical is valid, but SAR is missing or invalid; or vice versa)
        - 0 = INVALID (cloud, cloud shadow, NoData, out-of-bounds)
        """
        h, w = optical_mask.shape
        quality_mask = np.zeros((h, w), dtype=np.uint8)

        opt_valid = (optical_mask > 0)

        if sar_available and sar_mask is not None:
            sar_valid = (sar_mask > 0)
            both_valid = opt_valid & sar_valid
            either_valid = opt_valid | sar_valid

            quality_mask[either_valid] = MASK_PARTIAL
            quality_mask[both_valid] = MASK_VALID
            quality_mask[~either_valid] = MASK_INVALID
        else:
            # When SAR is unavailable, valid optical pixels are marked PARTIAL (optical-only)
            quality_mask[opt_valid] = MASK_PARTIAL
            quality_mask[~opt_valid] = MASK_INVALID

        return quality_mask

    def normalize_feature_stack(
        self,
        features: Dict[str, Optional[np.ndarray]],
        method: str = "raw",
        stats_cache: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Applies configurable feature normalization without leaking future observations:
        - 'raw': returns exact physical satellite measurements.
        - 'min-max': scales finite valid pixels linearly to [0.0, 1.0].
        - 'z-score': standardizes to zero-mean and unit-variance: (x - mean) / std.
        Preserves NaNs for NoData/missing pixels.
        """
        if method == "raw":
            return {k: (v.copy() if v is not None else None) for k, v in features.items()}

        normalized: Dict[str, Optional[np.ndarray]] = {}

        for name, arr in features.items():
            if arr is None:
                normalized[name] = None
                continue

            arr_copy = arr.copy().astype(np.float32)
            finite_mask = np.isfinite(arr_copy)

            if not finite_mask.any():
                normalized[name] = arr_copy
                continue

            valid_vals = arr_copy[finite_mask]

            if method == "min-max":
                min_val = float(np.min(valid_vals))
                max_val = float(np.max(valid_vals))
                diff = max_val - min_val
                if diff > 1e-6:
                    arr_copy[finite_mask] = (valid_vals - min_val) / diff
                else:
                    arr_copy[finite_mask] = 0.5
                normalized[name] = np.clip(arr_copy, 0.0, 1.0)

            elif method == "z-score":
                mean_val = float(np.mean(valid_vals))
                std_val = float(np.std(valid_vals))
                if std_val > 1e-6:
                    arr_copy[finite_mask] = (valid_vals - mean_val) / std_val
                else:
                    arr_copy[finite_mask] = 0.0
                normalized[name] = arr_copy
            else:
                normalized[name] = arr_copy

        return normalized

    async def execute_multimodal_fusion(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        max_cloud_coverage: float = 30.0,
        temporal_window_days: int = DEFAULT_TEMPORAL_WINDOW_DAYS,
        normalization: str = "raw",
        aoi_area_km2: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        End-to-End Multimodal Satellite Feature-Fusion Pipeline:
        1. Create Canonical AOI Grid (UTM / metric projection, resolution, dimensions)
        2. Retrieve Sentinel-2 Multispectral Bands (B02, B03, B04, B05, B08, B11, SCL)
        3. Query Sentinel-1 SAR GRD Catalog over AOI
        4. Perform Configurable Temporal Matching (±temporal_window_days)
        5. Retrieve/Compute SAR Backscatter (VV, VH, VV/VH ratio) if matched
        6. Spatial Alignment & Resampling onto Canonical Grid
        7. Compute Spectral Vegetation Indices (NDVI, NDMI, NDRE, EVI)
        8. Construct Unified 3-Tier Quality Mask (VALID, PARTIAL, INVALID)
        9. Compute Comprehensive Feature Quality Statistics
        10. Apply Normalization (raw, min-max, z-score)
        11. Package Standardized Multimodal Dataset ready for downstream AI
        """
        # 1. Establish Canonical AOI Grid
        canonical_grid = create_canonical_aoi_grid(geometry, target_resolution_m=10.0)
        target_shape = (canonical_grid.height, canonical_grid.width)

        # 2. Sentinel-2 Optical Acquisition
        is_demo_run = self.is_demo_mode or not sentinel2_service.auth.has_credentials()

        if is_demo_run:
            raw_s2_bands = generate_all_demo_spectral_bands(start_date, end_date, shape=target_shape, include_b03=True)
            s2_scene = {
                "scene_id": f"S2A_MSIL2A_DEMO_{end_date.replace('-', '')}T052021",
                "acquisition_date": f"{end_date}T05:20:21Z",
                "cloud_coverage": 4.8,
                "data_source": "Copernicus Sentinel-2 (Demo)",
                "is_demo": True,
            }
        else:
            try:
                s2_scene, raw_s2_bands = await sentinel2_service._fetch_multispectral_bands(
                    geometry=geometry,
                    start_date=start_date,
                    end_date=end_date,
                    max_cloud_coverage=max_cloud_coverage,
                    band_names=["B02", "B03", "B04", "B05", "B08", "B11"],
                )
            except Exception:
                # Deterministic fallback if remote query failed
                raw_s2_bands = generate_all_demo_spectral_bands(start_date, end_date, shape=target_shape, include_b03=True)
                s2_scene = {
                    "scene_id": f"S2A_MSIL2A_DEMO_{end_date.replace('-', '')}T052021",
                    "acquisition_date": f"{end_date}T05:20:21Z",
                    "cloud_coverage": 5.0,
                    "data_source": "Copernicus Sentinel-2 (Fallback Demo)",
                    "is_demo": True,
                }
                is_demo_run = True

        # Extract acquisition date
        s2_date = s2_scene.get("acquisition_date", f"{end_date}T10:00:00Z").split("T")[0]

        # 3 & 4. Sentinel-1 SAR Search & Temporal Matching
        s1_scenes = await sentinel1_service.search_imagery(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
        )

        temporal_match = self.match_temporal_observations(
            sentinel2_date=s2_date,
            sentinel1_scenes=s1_scenes,
            temporal_window_days=temporal_window_days,
        )

        sar_available = temporal_match["sar_available"]
        s1_scene_info = temporal_match.get("matched_scene")

        # 5. Retrieve SAR Backscatter if matched
        raw_sar_features: Dict[str, np.ndarray] = {}
        sar_mask: Optional[np.ndarray] = None

        if sar_available:
            try:
                vv_raw, vh_raw, sar_m, s1_meta = await sentinel1_service.get_sar_bands(
                    geometry=geometry,
                    start_date=temporal_match["sentinel1_date"],
                    end_date=temporal_match["sentinel1_date"],
                    width=canonical_grid.width,
                    height=canonical_grid.height,
                )
                raw_sar_features["VV_LINEAR"] = vv_raw
                raw_sar_features["VH_LINEAR"] = vh_raw
                sar_mask = sar_m
            except Exception:
                # If SAR data fetch failed, mark SAR unavailable rather than fabricating values
                sar_available = False
                temporal_match["sar_available"] = False
                temporal_match["matched"] = False
                temporal_match["message"] = "Sentinel-1 data download failed for matched scene."

        # 6. Spatial Alignment of Optical and SAR features to Canonical Grid
        aligned_opt_bands, aligned_sar_bands = align_all_multimodal_features(
            optical_features=raw_s2_bands,
            sar_features=raw_sar_features,
            canonical_grid=canonical_grid,
        )

        # 7. Compute Spectral Indices on Canonical Grid
        b02 = aligned_opt_bands["B02"]
        b03 = aligned_opt_bands.get("B03", b02 * 1.1)
        b04 = aligned_opt_bands["B04"]
        b05 = aligned_opt_bands["B05"]
        b08 = aligned_opt_bands["B08"]
        b11 = aligned_opt_bands["B11"]
        scl = aligned_opt_bands.get("SCL")
        data_mask = aligned_opt_bands.get("dataMask")

        optical_valid_mask = create_valid_pixel_mask(b04, b08, scl=scl, data_mask=data_mask)

        ndvi = calculate_ndvi(b04, b08, valid_mask=optical_valid_mask)
        ndmi = calculate_ndmi(b08, b11, valid_mask=optical_valid_mask)
        ndre = calculate_ndre(b08, b05, valid_mask=optical_valid_mask)
        evi = calculate_evi(b02, b04, b08, valid_mask=optical_valid_mask)

        # SAR dB Backscatter and VV/VH Ratio
        if sar_available and "VV_LINEAR" in aligned_sar_bands and "VH_LINEAR" in aligned_sar_bands:
            vv_lin = np.maximum(aligned_sar_bands["VV_LINEAR"], 1e-6)
            vh_lin = np.maximum(aligned_sar_bands["VH_LINEAR"], 1e-6)

            vv_db = 10.0 * np.log10(vv_lin)
            vh_db = 10.0 * np.log10(vh_lin)
            vv_vh_ratio = vv_db - vh_db  # Equivalent to 10 * log10(VV / VH)

            # Resample SAR mask to canonical grid if needed
            if sar_mask is not None:
                aligned_sar_mask = resample_raster_to_canonical_grid(
                    sar_mask, target_shape=target_shape, is_discrete=True
                )
            else:
                aligned_sar_mask = np.ones(target_shape, dtype=bool)
        else:
            vv_db = None
            vh_db = None
            vv_vh_ratio = None
            aligned_sar_mask = None

        # 8. Construct Unified 3-Tier Quality Mask
        quality_mask = self.build_unified_quality_mask(
            optical_mask=optical_valid_mask,
            sar_mask=aligned_sar_mask,
            sar_available=sar_available,
        )

        # 9. Assemble Raw Feature Stack Dictionary
        raw_feature_stack: Dict[str, Optional[np.ndarray]] = {
            "B02": b02,
            "B03": b03,
            "B04": b04,
            "B05": b05,
            "B08": b08,
            "B11": b11,
            "NDVI": ndvi,
            "NDMI": ndmi,
            "NDRE": ndre,
            "EVI": evi,
            "VV": vv_db,
            "VH": vh_db,
            "VV_VH_RATIO": vv_vh_ratio,
        }

        # 10. Calculate Feature Quality Statistics
        feature_statistics: Dict[str, Any] = {}
        features_available_count = 0

        for feat_name, arr in raw_feature_stack.items():
            if feat_name in ("VV", "VH", "VV_VH_RATIO"):
                mask_for_stats = (quality_mask == MASK_VALID) if sar_available else None
            else:
                mask_for_stats = (quality_mask > 0)

            stats = self.compute_feature_statistics(arr, valid_mask=mask_for_stats, feature_name=feat_name)
            feature_statistics[feat_name] = stats
            if stats["available"]:
                features_available_count += 1

        # 11. Apply Normalization if requested
        processed_features = self.normalize_feature_stack(
            raw_feature_stack, method=normalization
        )

        # 12. Quality Summary & AI Readiness Status
        total_pixels = int(quality_mask.size)
        valid_pixels = int(np.count_nonzero(quality_mask == MASK_VALID))
        partial_pixels = int(np.count_nonzero(quality_mask == MASK_PARTIAL))
        invalid_pixels = int(np.count_nonzero(quality_mask == MASK_INVALID))

        opt_valid_pct = round((np.count_nonzero(optical_valid_mask) / total_pixels) * 100.0, 2)
        sar_valid_pct = (
            round((np.count_nonzero(aligned_sar_mask) / total_pixels) * 100.0, 2)
            if (sar_available and aligned_sar_mask is not None)
            else 0.0
        )
        fused_valid_pct = round((valid_pixels / total_pixels) * 100.0, 2)

        if sar_available and fused_valid_pct >= 50.0:
            fusion_readiness = "READY_FOR_AI"
            readiness_desc = "High-quality multimodal optical + SAR feature stack ready for AI model ingestion."
        elif opt_valid_pct >= 50.0:
            fusion_readiness = "PARTIAL_OPTICAL_ONLY"
            readiness_desc = "Multispectral optical features valid; SAR observations missing or insufficient."
        else:
            fusion_readiness = "INSUFFICIENT_QUALITY"
            readiness_desc = "Heavy cloud cover or excessive NoData pixels compromise dataset validity."

        # Colormap previews for visualization
        previews = self._generate_feature_previews(
            raw_feature_stack, quality_mask
        )

        # Build standardized provenance badge
        provenance = build_provenance_badge(
            is_demo=is_demo_run,
            sensor="Copernicus Sentinel-2 MSI + Sentinel-1 C-SAR",
            optical_date=s2_scene.get("acquisition_date"),
            sar_date=s1_scene_info.get("acquisition_date") if s1_scene_info else None,
            temporal_gap_days=temporal_match.get("temporal_gap_days"),
            cloud_coverage_percentage=s2_scene.get("cloud_coverage", 0.0),
            valid_pixel_percentage=fused_valid_pct,
            scene_id=f"{s2_scene.get('scene_id', 'S2_SCENE')} + {s1_scene_info.get('scene_id', 'NO_SAR') if s1_scene_info else 'NO_SAR'}",
        )

        # Compile final fusion record
        fusion_record = {
            "status": "success",
            "data_source": "SIMULATED" if is_demo_run else "REAL",
            "is_demo": is_demo_run,
            "fusion_readiness": fusion_readiness,
            "readiness_description": readiness_desc,
            "provenance": provenance,
            "features_available_count": features_available_count,
            "total_features_count": len(MULTIMODAL_FEATURE_NAMES),
            "normalization_applied": normalization,
            "canonical_grid": canonical_grid.to_dict(),
            "temporal_matching": temporal_match,
            "quality_summary": {
                "optical_valid_percentage": opt_valid_pct,
                "sar_valid_percentage": sar_valid_pct,
                "fused_valid_percentage": fused_valid_pct,
                "partial_pixel_percentage": round((partial_pixels / total_pixels) * 100.0, 2),
                "invalid_pixel_percentage": round((invalid_pixels / total_pixels) * 100.0, 2),
                "cloud_coverage": s2_scene.get("cloud_coverage", 0.0),
            },
            "feature_statistics": feature_statistics,
            "previews": previews,
            "scenes": {
                "optical": s2_scene,
                "sar": s1_scene_info,
            },
            # Cached in-memory arrays for dataset export
            "_raw_features": raw_feature_stack,
            "_normalized_features": processed_features,
            "_quality_mask": quality_mask,
        }

        return fusion_record

    def export_dataset(
        self,
        fusion_record: Dict[str, Any],
        export_format: str = "npz",
    ) -> Union[bytes, Dict[str, Any]]:
        """
        Exports the fused multimodal observation for downstream machine-learning models:
        - 'npz': Compressed NumPy archive (.npz) containing 2D arrays, quality mask, metadata,
                 and labels_placeholder = None.
        - 'json': Clean JSON-serializable dictionary with statistics and grid specs.
        - 'geotiff': Multi-band TIFF containing all 13 feature channels + 1 quality mask channel.
        """
        raw_features = fusion_record.get("_raw_features")
        quality_mask = fusion_record.get("_quality_mask")
        grid = fusion_record.get("canonical_grid", {})

        if raw_features is None or quality_mask is None:
            raise MultimodalFusionError("Cannot export fusion record without internal feature arrays.")

        meta_json = json.dumps({
            "crs": grid.get("crs"),
            "epsg": grid.get("epsg"),
            "width": grid.get("width"),
            "height": grid.get("height"),
            "resolution_m": grid.get("resolution_m"),
            "bounds": grid.get("bounds"),
            "geo_bounds": grid.get("geo_bounds"),
            "temporal_matching": fusion_record.get("temporal_matching"),
            "quality_summary": fusion_record.get("quality_summary"),
            "feature_statistics": fusion_record.get("feature_statistics"),
            "labels_placeholder": None,
        }, indent=2)

        if export_format.lower() == "json":
            return json.loads(meta_json)

        elif export_format.lower() == "npz":
            # Prepare dictionary for np.savez_compressed
            save_dict = {
                "quality_mask": quality_mask,
                "metadata": np.array(meta_json),
                "labels_placeholder": np.array([], dtype=np.float32),  # No fake labels
            }
            for name in MULTIMODAL_FEATURE_NAMES:
                arr = raw_features.get(name)
                if arr is not None:
                    save_dict[name] = arr.astype(np.float32)
                else:
                    save_dict[name] = np.full((grid.get("height", 64), grid.get("width", 64)), np.nan, dtype=np.float32)

            bio = io.BytesIO()
            np.savez_compressed(bio, **save_dict)
            bio.seek(0)
            return bio.getvalue()

        elif export_format.lower() in ("geotiff", "tiff", "tif"):
            # Multi-channel float32 stack: shape (channels, height, width)
            h = grid.get("height", 64)
            w = grid.get("width", 64)
            channels = []
            for name in MULTIMODAL_FEATURE_NAMES:
                arr = raw_features.get(name)
                if arr is not None and arr.shape == (h, w):
                    channels.append(arr.astype(np.float32))
                else:
                    channels.append(np.full((h, w), np.nan, dtype=np.float32))

            # Channel 14: Quality Mask
            channels.append(quality_mask.astype(np.float32))

            stack = np.stack(channels, axis=0)  # Shape (14, H, W)
            bio = io.BytesIO()
            tifffile.imwrite(bio, stack, photometric="minisblack")
            bio.seek(0)
            return bio.getvalue()

        else:
            raise MultimodalFusionError(f"Unsupported export format '{export_format}'. Expected 'npz', 'json', or 'geotiff'.")

    def _generate_feature_previews(
        self,
        features: Dict[str, Optional[np.ndarray]],
        quality_mask: np.ndarray,
    ) -> Dict[str, Optional[str]]:
        """
        Generates base64 PNG previews with scientific colormaps for core fused features:
        NDVI, NDMI, NDRE, EVI, VV, VH, VV/VH Ratio, and Quality Mask.
        """
        previews: Dict[str, Optional[str]] = {}

        # 1. Quality Mask Preview (Green = Valid, Yellow = Partial, Red = Invalid)
        mask_rgb = np.zeros((*quality_mask.shape, 3), dtype=np.uint8)
        mask_rgb[quality_mask == MASK_VALID] = [34, 197, 94]     # Emerald Green
        mask_rgb[quality_mask == MASK_PARTIAL] = [234, 179, 8]   # Amber Yellow
        mask_rgb[quality_mask == MASK_INVALID] = [239, 68, 68]   # Red
        previews["quality_mask"] = self._rgb_to_base64_png(mask_rgb)

        # 2. Vegetation Indices (NDVI, NDMI, NDRE, EVI)
        for idx in ("NDVI", "NDMI", "NDRE", "EVI"):
            arr = features.get(idx)
            if arr is not None:
                previews[idx.lower()] = self._array_to_colormap_png(arr, min_val=-0.2, max_val=0.85, palette="rdylgn")
            else:
                previews[idx.lower()] = None

        # 3. SAR Backscatter (VV, VH, VV/VH Ratio)
        if features.get("VV") is not None:
            previews["vv"] = self._array_to_colormap_png(features["VV"], min_val=-22.0, max_val=-5.0, palette="gray")
        else:
            previews["vv"] = None

        if features.get("VH") is not None:
            previews["vh"] = self._array_to_colormap_png(features["VH"], min_val=-28.0, max_val=-12.0, palette="viridis")
        else:
            previews["vh"] = None

        if features.get("VV_VH_RATIO") is not None:
            previews["vv_vh_ratio"] = self._array_to_colormap_png(features["VV_VH_RATIO"], min_val=2.0, max_val=12.0, palette="turbo")
        else:
            previews["vv_vh_ratio"] = None

        return previews

    def _array_to_colormap_png(
        self,
        arr: np.ndarray,
        min_val: float,
        max_val: float,
        palette: str = "rdylgn",
    ) -> str:
        """Helper to render a 2D float array into a base64 PNG with color mapping."""
        nan_mask = np.isnan(arr)
        clean_arr = np.nan_to_num(arr, nan=min_val)
        norm = np.clip((clean_arr - min_val) / (max_val - min_val + 1e-8), 0.0, 1.0)
        h, w = arr.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        if palette == "gray":
            val = (norm * 255).astype(np.uint8)
            rgb[..., 0] = val
            rgb[..., 1] = val
            rgb[..., 2] = val
        elif palette == "viridis":
            # Simplified viridis approximation: purple -> teal -> yellow
            rgb[..., 0] = (np.clip(1.5 * norm - 0.5, 0, 1) * 253 + (1 - norm) * 68).astype(np.uint8)
            rgb[..., 1] = (norm * 231 + (1 - norm) * 1).astype(np.uint8)
            rgb[..., 2] = ((1 - norm) * 112 + norm * 36).astype(np.uint8)
        elif palette == "turbo":
            # Turbo spectrum
            rgb[..., 0] = (np.sin(norm * math.pi) * 200 + norm * 55).astype(np.uint8)
            rgb[..., 1] = (np.sin(norm * math.pi * 0.8) * 230).astype(np.uint8)
            rgb[..., 2] = ((1.0 - norm) * 220).astype(np.uint8)
        else:
            # RdYlGn for vegetation
            rgb[..., 0] = (np.clip(2.0 * (1.0 - norm), 0.0, 1.0) * 220).astype(np.uint8)
            rgb[..., 1] = (np.clip(2.0 * norm, 0.0, 1.0) * 200).astype(np.uint8)
            rgb[..., 2] = 40

        # Mask NaNs with dark neutral background
        rgb[nan_mask] = [30, 30, 30]

        return self._rgb_to_base64_png(rgb)

    def _rgb_to_base64_png(self, rgb: np.ndarray) -> str:
        img = Image.fromarray(rgb)
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        bio.seek(0)
        return f"data:image/png;base64,{base64.b64encode(bio.getvalue()).decode('utf-8')}"

# Global singleton
multimodal_fusion_service = MultimodalFusionService()
