import io
import base64
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw

from services.raster_utils import resample_band

# Configurable heuristic vegetation condition configuration
# Note: These are heuristic reference thresholds and must NOT be presented as universal calibrated agricultural limits.
DEFAULT_CONDITION_CONFIG: Dict[str, Any] = {
    "name": "Heuristic Vegetation Condition Configuration",
    "description": (
        "Configurable analytical baseline ranges for Sentinel-2 spectral indices. "
        "These thresholds serve as empirical reference points and do NOT represent universal "
        "agricultural ground truth across all crop varieties, phenological stages, and soil conditions."
    ),
    "ndvi": {
        "min_reference": 0.0,
        "low": 0.25,
        "moderate": 0.45,
        "good": 0.65,
        "max_reference": 0.90,
    },
    "ndmi": {
        "min_reference": -0.30,
        "low": -0.05,
        "moderate": 0.15,
        "good": 0.35,
        "max_reference": 0.60,
    },
    "ndre": {
        "min_reference": 0.05,
        "low": 0.18,
        "moderate": 0.28,
        "good": 0.42,
        "max_reference": 0.65,
    },
    "evi": {
        "min_reference": 0.05,
        "low": 0.18,
        "moderate": 0.32,
        "good": 0.50,
        "max_reference": 0.80,
    },
    "weights": {
        "ndvi": 0.35,
        "ndmi": 0.25,
        "ndre": 0.20,
        "evi": 0.20,
    },
}

ZONE_COLORS = {
    0: (0, 0, 0, 0),         # 0 = No Data (Transparent)
    1: (215, 48, 39, 255),   # 1 = Very Low (Crimson Red)
    2: (244, 109, 67, 255),  # 2 = Low (Coral Amber)
    3: (254, 224, 139, 255), # 3 = Moderate (Golden Yellow)
    4: (166, 217, 106, 255), # 4 = Good (Lime Green)
    5: (26, 152, 80, 255),   # 5 = Very Good (Emerald Forest Green)
}


def normalize_index_array(
    arr: np.ndarray,
    min_ref: float,
    low_ref: float,
    mod_ref: float,
    good_ref: float,
    max_ref: float,
) -> np.ndarray:
    """
    Normalizes a spectral index array to [0.0, 1.0] using piece-wise linear interpolation
    across the configured heuristic reference levels:
    - arr <= min_ref -> 0.0
    - arr == low_ref -> 0.30
    - arr == mod_ref -> 0.55
    - arr == good_ref -> 0.80
    - arr >= max_ref -> 1.0
    NaN values are preserved safely.
    """
    res = np.full(arr.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return res

    v = arr[valid]
    norm = np.zeros_like(v, dtype=np.float64)

    # Segment 1: < min_ref
    m1 = v < min_ref
    norm[m1] = 0.0

    # Segment 2: min_ref to low_ref (maps 0.0 to 0.30)
    m2 = (v >= min_ref) & (v < low_ref)
    if np.any(m2):
        denom = max(1e-6, low_ref - min_ref)
        norm[m2] = 0.0 + ((v[m2] - min_ref) / denom) * 0.30

    # Segment 3: low_ref to mod_ref (maps 0.30 to 0.55)
    m3 = (v >= low_ref) & (v < mod_ref)
    if np.any(m3):
        denom = max(1e-6, mod_ref - low_ref)
        norm[m3] = 0.30 + ((v[m3] - low_ref) / denom) * 0.25

    # Segment 4: mod_ref to good_ref (maps 0.55 to 0.80)
    m4 = (v >= mod_ref) & (v < good_ref)
    if np.any(m4):
        denom = max(1e-6, good_ref - mod_ref)
        norm[m4] = 0.55 + ((v[m4] - mod_ref) / denom) * 0.25

    # Segment 5: good_ref to max_ref (maps 0.80 to 1.00)
    m5 = (v >= good_ref) & (v < max_ref)
    if np.any(m5):
        denom = max(1e-6, max_ref - good_ref)
        norm[m5] = 0.80 + ((v[m5] - good_ref) / denom) * 0.20

    # Segment 6: >= max_ref
    m6 = v >= max_ref
    norm[m6] = 1.0

    res[valid] = np.clip(norm, 0.0, 1.0)
    return res


def compute_crop_condition(
    ndvi: Optional[Union[np.ndarray, list]] = None,
    ndmi: Optional[Union[np.ndarray, list]] = None,
    ndre: Optional[Union[np.ndarray, list]] = None,
    evi: Optional[Union[np.ndarray, list]] = None,
    valid_mask: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    pixel_size_m: float = 10.0,
    total_aoi_area_km2: Optional[float] = None,
    cloud_coverage: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculates heuristic vegetation condition score (0 to 100), spatial condition zones (1 to 5),
    pattern recognition, and data quality information from available multispectral indices.
    """
    cfg = config or DEFAULT_CONDITION_CONFIG
    weights_cfg = cfg.get("weights", DEFAULT_CONDITION_CONFIG["weights"])

    # 1. Determine common target shape from available indices
    available_indices: Dict[str, np.ndarray] = {}
    target_shape: Optional[Tuple[int, int]] = None

    for name, arr in [("ndvi", ndvi), ("ndmi", ndmi), ("ndre", ndre), ("evi", evi)]:
        if arr is not None:
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2:
                available_indices[name] = a
                if target_shape is None:
                    target_shape = a.shape

    if not available_indices or target_shape is None:
        raise ValueError("At least one 2D index array must be provided for crop condition analysis.")

    h, w = target_shape
    total_pixels = h * w

    # Align all available indices to target shape
    for name, arr in list(available_indices.items()):
        if arr.shape != target_shape:
            available_indices[name] = resample_band(arr, target_shape=target_shape, method="bilinear")

    # Combine valid mask with NaN exclusion across all supplied indices
    if valid_mask is not None:
        combined_mask = np.asarray(valid_mask, dtype=bool)
        if combined_mask.shape != target_shape:
            combined_mask = resample_band(combined_mask, target_shape=target_shape, is_discrete=True) > 0
    else:
        combined_mask = np.ones(target_shape, dtype=bool)

    # Also invalidate any pixel that is NaN in all provided indices
    any_valid_pixel = np.zeros(target_shape, dtype=bool)
    for arr in available_indices.values():
        any_valid_pixel |= ~np.isnan(arr)
    combined_mask &= any_valid_pixel

    valid_pixel_count = int(np.sum(combined_mask))
    invalid_pixel_count = total_pixels - valid_pixel_count
    valid_pixel_percentage = round((valid_pixel_count / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0

    # Area calculations
    pixel_area_m2 = pixel_size_m * pixel_size_m
    calculated_area_km2 = round((valid_pixel_count * pixel_area_m2) / 1_000_000.0, 4)
    if total_aoi_area_km2 is not None and total_aoi_area_km2 > 0:
        area_represented_km2 = round(total_aoi_area_km2 * (valid_pixel_percentage / 100.0), 2)
    else:
        area_represented_km2 = calculated_area_km2

    # Handle edge case: 0 valid pixels
    if valid_pixel_count == 0:
        empty_zones = {"very_low": 0.0, "low": 0.0, "moderate": 0.0, "good": 0.0, "very_good": 0.0}
        zone_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        return {
            "overall_condition": "No Data",
            "condition_score": 0.0,
            "condition_score_mean": None,
            "condition_score_median": None,
            "condition_score_std": None,
            "dominant_zone": "No Data",
            "valid_pixel_percentage": 0.0,
            "area_represented_km2": 0.0,
            "zones": empty_zones,
            "zone_counts": zone_counts,
            "classified_raster": np.zeros(target_shape, dtype=np.uint8),
            "condition_score_grid": np.full(target_shape, np.nan, dtype=np.float64),
            "patterns": [],
            "quality": {
                "valid_pixels": 0,
                "invalid_pixels": total_pixels,
                "valid_percentage": 0.0,
                "quality_level": "Poor",
            },
            "warnings": [
                "No valid vegetation observations found in the selected area. All pixels were masked by quality/cloud filtering."
            ],
            "analysis_metadata": {
                "method": "heuristic_spectral_condition_scoring",
                "indices_used": list(available_indices.keys()),
                "weights_applied": {},
                "disclaimer": "Heuristic vegetation condition analysis; not a disease diagnosis or nutrient assessment.",
            },
        }

    # 2. Normalize and combine available indices
    # Renormalize weights among present indices
    active_weights: Dict[str, float] = {}
    for name in available_indices.keys():
        active_weights[name] = float(weights_cfg.get(name, 0.25))

    total_weight = sum(active_weights.values())
    if total_weight > 0:
        for name in active_weights:
            active_weights[name] /= total_weight
    else:
        eq = 1.0 / len(active_weights)
        for name in active_weights:
            active_weights[name] = eq

    # Accumulate weighted normalized scores
    score_accumulator = np.zeros(target_shape, dtype=np.float64)
    weight_accumulator = np.zeros(target_shape, dtype=np.float64)

    normalized_grids: Dict[str, np.ndarray] = {}
    for name, arr in available_indices.items():
        sub_cfg = cfg.get(name, DEFAULT_CONDITION_CONFIG[name])
        norm = normalize_index_array(
            arr,
            min_ref=sub_cfg["min_reference"],
            low_ref=sub_cfg["low"],
            mod_ref=sub_cfg["moderate"],
            good_ref=sub_cfg["good"],
            max_ref=sub_cfg["max_reference"],
        )
        normalized_grids[name] = norm

        w_val = active_weights[name]
        valid_idx = ~np.isnan(norm) & combined_mask
        score_accumulator[valid_idx] += norm[valid_idx] * w_val
        weight_accumulator[valid_idx] += w_val

    # Final per-pixel condition score (0.0 to 100.0)
    condition_score_grid = np.full(target_shape, np.nan, dtype=np.float64)
    valid_score_mask = (weight_accumulator > 0.0) & combined_mask
    condition_score_grid[valid_score_mask] = (
        (score_accumulator[valid_score_mask] / weight_accumulator[valid_score_mask]) * 100.0
    )

    valid_scores = condition_score_grid[valid_score_mask]
    if valid_scores.size == 0:
        mean_score = 0.0
        median_score = 0.0
        std_score = 0.0
    else:
        mean_score = round(float(np.mean(valid_scores)), 1)
        median_score = round(float(np.median(valid_scores)), 1)
        std_score = round(float(np.std(valid_scores)), 1)

    # Categorize overall condition score
    if mean_score >= 80.0:
        overall_condition = "Very Good"
    elif mean_score >= 60.0:
        overall_condition = "Good"
    elif mean_score >= 40.0:
        overall_condition = "Moderate"
    elif mean_score >= 20.0:
        overall_condition = "Low"
    else:
        overall_condition = "Very Low"

    # 3. Spatial Condition Zones Classification Raster
    # 0 = No Data, 1 = Very Low, 2 = Low, 3 = Moderate, 4 = Good, 5 = Very Good
    classified_raster = np.zeros(target_shape, dtype=np.uint8)

    c1 = valid_score_mask & (condition_score_grid < 20.0)
    c2 = valid_score_mask & (condition_score_grid >= 20.0) & (condition_score_grid < 40.0)
    c3 = valid_score_mask & (condition_score_grid >= 40.0) & (condition_score_grid < 60.0)
    c4 = valid_score_mask & (condition_score_grid >= 60.0) & (condition_score_grid < 80.0)
    c5 = valid_score_mask & (condition_score_grid >= 80.0)

    classified_raster[c1] = 1
    classified_raster[c2] = 2
    classified_raster[c3] = 3
    classified_raster[c4] = 4
    classified_raster[c5] = 5

    count_c1 = int(np.sum(c1))
    count_c2 = int(np.sum(c2))
    count_c3 = int(np.sum(c3))
    count_c4 = int(np.sum(c4))
    count_c5 = int(np.sum(c5))

    zone_counts = {1: count_c1, 2: count_c2, 3: count_c3, 4: count_c4, 5: count_c5}

    zones_percentage = {
        "very_low": round((count_c1 / valid_pixel_count) * 100.0, 1),
        "low": round((count_c2 / valid_pixel_count) * 100.0, 1),
        "moderate": round((count_c3 / valid_pixel_count) * 100.0, 1),
        "good": round((count_c4 / valid_pixel_count) * 100.0, 1),
        "very_good": round((count_c5 / valid_pixel_count) * 100.0, 1),
    }

    # Dominant zone
    dominant_key = max(zones_percentage, key=lambda k: zones_percentage[k])
    dominant_zone_labels = {
        "very_low": "Very Low Condition",
        "low": "Low Condition",
        "moderate": "Moderate Condition",
        "good": "Good Condition",
        "very_good": "Very Good Condition",
    }
    dominant_zone = dominant_zone_labels.get(dominant_key, "Moderate Condition")

    # 4. Stress Patterns Recognition
    patterns: List[Dict[str, Any]] = []

    ndvi_arr = available_indices.get("ndvi")
    ndmi_arr = available_indices.get("ndmi")
    ndre_arr = available_indices.get("ndre")
    evi_arr = available_indices.get("evi")

    low_ndvi_th = cfg.get("ndvi", DEFAULT_CONDITION_CONFIG["ndvi"])["low"]
    mod_ndvi_th = cfg.get("ndvi", DEFAULT_CONDITION_CONFIG["ndvi"])["moderate"]
    low_ndmi_th = cfg.get("ndmi", DEFAULT_CONDITION_CONFIG["ndmi"])["low"]
    mod_ndmi_th = cfg.get("ndmi", DEFAULT_CONDITION_CONFIG["ndmi"])["moderate"]
    low_ndre_th = cfg.get("ndre", DEFAULT_CONDITION_CONFIG["ndre"])["low"]
    low_evi_th = cfg.get("evi", DEFAULT_CONDITION_CONFIG["evi"])["low"]

    # Pattern A: Low NDVI + Low NDMI
    if ndvi_arr is not None and ndmi_arr is not None:
        p_a = valid_score_mask & (ndvi_arr < low_ndvi_th) & (ndmi_arr < low_ndmi_th)
        pct_a = round((float(np.sum(p_a)) / valid_pixel_count) * 100.0, 1)
        if pct_a >= 0.5:
            severity = "high" if pct_a >= 35.0 else ("moderate" if pct_a >= 15.0 else "low")
            patterns.append({
                "type": "possible_moisture_stress",
                "label": "Low Vigor & Reduced Moisture Signal",
                "severity": severity,
                "percentage": pct_a,
                "explanation": (
                    "Reduced vegetation vigor is present together with a lower moisture-related spectral signal. "
                    "This pattern may be consistent with vegetation stress, but field verification is required."
                ),
            })

    # Pattern B: Low NDVI + relatively normal NDMI
    if ndvi_arr is not None and ndmi_arr is not None:
        p_b = valid_score_mask & (ndvi_arr < low_ndvi_th) & (ndmi_arr >= mod_ndmi_th)
        pct_b = round((float(np.sum(p_b)) / valid_pixel_count) * 100.0, 1)
        if pct_b >= 0.5:
            severity = "high" if pct_b >= 35.0 else ("moderate" if pct_b >= 15.0 else "low")
            patterns.append({
                "type": "possible_canopy_vigor_reduction",
                "label": "Low Vigor with Preserved Moisture Signal",
                "severity": severity,
                "percentage": pct_b,
                "explanation": (
                    "Vegetation vigor is lower than the configured reference range without a similarly low "
                    "moisture-related signal. Other environmental, soil, or crop management factors may contribute."
                ),
            })

    # Pattern C: Normal NDVI + Low NDRE
    if ndvi_arr is not None and ndre_arr is not None:
        p_c = valid_score_mask & (ndvi_arr >= mod_ndvi_th) & (ndre_arr < low_ndre_th)
        pct_c = round((float(np.sum(p_c)) / valid_pixel_count) * 100.0, 1)
        if pct_c >= 0.5:
            severity = "high" if pct_c >= 35.0 else ("moderate" if pct_c >= 15.0 else "low")
            patterns.append({
                "type": "possible_chlorophyll_transition",
                "label": "Red-Edge Transition Discrepancy",
                "severity": severity,
                "percentage": pct_c,
                "explanation": (
                    "Red-edge response is relatively lower than the configured reference range while general canopy "
                    "greenness is maintained. This may indicate a change in vegetation/chlorophyll-related spectral response, "
                    "but it does not by itself establish nutrient deficiency."
                ),
            })

    # Pattern D: Low EVI + Low NDVI
    if evi_arr is not None and ndvi_arr is not None:
        p_d = valid_score_mask & (evi_arr < low_evi_th) & (ndvi_arr < low_ndvi_th)
        pct_d = round((float(np.sum(p_d)) / valid_pixel_count) * 100.0, 1)
        if pct_d >= 0.5:
            severity = "high" if pct_d >= 35.0 else ("moderate" if pct_d >= 15.0 else "low")
            patterns.append({
                "type": "concurring_canopy_reduction",
                "label": "Dual Canopy Signal Reduction",
                "severity": severity,
                "percentage": pct_d,
                "explanation": (
                    "Both vegetation indices indicate reduced vegetation signal in the analyzed area."
                ),
            })

    # 5. Data Quality Assessment
    warnings: List[str] = []

    if valid_pixel_percentage >= 85.0:
        quality_level = "Excellent"
    elif valid_pixel_percentage >= 70.0:
        quality_level = "Good"
    elif valid_pixel_percentage >= 45.0:
        quality_level = "Fair"
    else:
        quality_level = "Poor"

    if valid_pixel_percentage < 70.0:
        warnings.append(
            f"Large portions of the AOI ({round(100.0 - valid_pixel_percentage, 1)}%) were excluded by the quality mask (cloud, shadow, or detector margins)."
        )

    if valid_pixel_percentage < 45.0:
        warnings.append(
            "Interpretation confidence is reduced because valid observations cover only a small portion of the AOI."
        )

    if cloud_coverage > 20.0:
        warnings.append(
            f"Elevated scene cloud coverage ({cloud_coverage}%). Spectral reflectance values may be subject to diffuse atmospheric haze."
        )

    if len(available_indices) < 4:
        missing = [k for k in ["ndvi", "ndmi", "ndre", "evi"] if k not in available_indices]
        warnings.append(
            f"Analysis performed with subset of indices. Missing: {', '.join(missing)}. Weights were dynamically renormalized."
        )

    return {
        "overall_condition": overall_condition,
        "condition_score": mean_score,
        "condition_score_mean": mean_score,
        "condition_score_median": median_score,
        "condition_score_std": std_score,
        "dominant_zone": dominant_zone,
        "valid_pixel_percentage": valid_pixel_percentage,
        "area_represented_km2": area_represented_km2,
        "zones": zones_percentage,
        "zone_counts": zone_counts,
        "classified_raster": classified_raster,
        "condition_score_grid": condition_score_grid,
        "patterns": patterns,
        "quality": {
            "valid_pixels": valid_pixel_count,
            "invalid_pixels": invalid_pixel_count,
            "valid_percentage": valid_pixel_percentage,
            "quality_level": quality_level,
        },
        "warnings": warnings,
        "analysis_metadata": {
            "method": "heuristic_spectral_condition_scoring",
            "indices_used": list(available_indices.keys()),
            "weights_applied": active_weights,
            "disclaimer": (
                "Heuristic vegetation condition analysis; not a medical or disease diagnosis. "
                "Patterns represent physical spectral signals and require agronomist ground verification."
            ),
        },
    }


def generate_condition_colormap(
    classified_raster: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    is_demo: bool = False,
    output_size: Tuple[int, int] = (512, 512),
) -> str:
    """
    Renders discrete spatial condition classification raster into a crisp RGBA PNG Data URL:
    0 = No Data (Transparent)
    1 = Very Low (Crimson Red)
    2 = Low (Coral Amber)
    3 = Moderate (Golden Yellow)
    4 = Good (Lime Green)
    5 = Very Good (Emerald Forest Green)
    """
    raster_arr = np.asarray(classified_raster, dtype=np.uint8)
    h, w = raster_arr.shape

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        effective = np.where(mask, raster_arr, 0)
    else:
        effective = raster_arr

    rgba_grid = np.zeros((h, w, 4), dtype=np.uint8)
    for class_id, color in ZONE_COLORS.items():
        match_pixels = effective == class_id
        if np.any(match_pixels):
            rgba_grid[match_pixels] = color

    img = Image.fromarray(rgba_grid, mode="RGBA")
    if (w, h) != output_size:
        # Use Nearest Neighbor to preserve crisp discrete zone boundaries
        img = img.resize(output_size, resample=Image.Resampling.NEAREST)

    if is_demo:
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, output_size[0], 42], fill=(20, 20, 25, 230))
        draw.rectangle([0, 42, output_size[0], 44], fill=(234, 179, 8, 255))
        draw.text((36, 12), "DEMO DATA — NOT LIVE SATELLITE ANALYSIS", fill=(255, 255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
