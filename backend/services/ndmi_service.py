import io
import base64
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw

from services.raster_utils import resample_band

class NDMIComputationError(Exception):
    """Raised when NDMI calculation or processing fails."""
    pass

def calculate_ndmi(
    nir: Union[np.ndarray, list],
    swir: Union[np.ndarray, list],
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculates Normalized Difference Moisture Index (NDMI) from Sentinel-2
    NIR (B08, 10m) and SWIR (B11, 20m) surface reflectance:

        NDMI = (B08 - B11) / (B08 + B11)

    NDMI is a vegetation moisture/water-related spectral indicator.
    It indicates canopy equivalent water thickness and moisture-related signals.
    It is NOT a direct proof of drought.
    """
    nir_arr = np.asarray(nir, dtype=np.float64)
    swir_arr = np.asarray(swir, dtype=np.float64)

    # Automatically resample SWIR (20m) to match NIR (10m) if shapes differ
    if swir_arr.shape != nir_arr.shape:
        swir_arr = resample_band(swir_arr, target_shape=nir_arr.shape, method="bilinear").astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = nir_arr - swir_arr
        denominator = nir_arr + swir_arr

        invalid = (
            np.isnan(nir_arr)
            | np.isnan(swir_arr)
            | np.isinf(nir_arr)
            | np.isinf(swir_arr)
            | (nir_arr < 0.0)
            | (swir_arr < 0.0)
            | np.isclose(denominator, 0.0, atol=1e-7)
        )

        ndmi = np.where(invalid, np.nan, numerator / denominator)

        if valid_mask is not None:
            mask_arr = np.asarray(valid_mask, dtype=bool)
            if mask_arr.shape != ndmi.shape:
                mask_arr = resample_band(mask_arr, target_shape=ndmi.shape, is_discrete=True) > 0
            ndmi = np.where(mask_arr, ndmi, np.nan)

    return ndmi

def calculate_ndmi_statistics(
    ndmi: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    pixel_size_m: float = 10.0,
    total_aoi_area_km2: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculates statistical metrics for NDMI moisture-related spectral measurements.
    """
    ndmi_arr = np.asarray(ndmi, dtype=np.float64)
    total_pixels = int(ndmi_arr.size)

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool) & ~np.isnan(ndmi_arr)
    else:
        mask = ~np.isnan(ndmi_arr)

    valid_vals = ndmi_arr[mask]
    valid_count = int(valid_vals.size)
    valid_percentage = round((valid_count / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0

    pixel_area_m2 = pixel_size_m * pixel_size_m
    calculated_area_km2 = round((valid_count * pixel_area_m2) / 1_000_000.0, 4)
    if total_aoi_area_km2 is not None and total_aoi_area_km2 > 0:
        area_represented_km2 = round(total_aoi_area_km2 * (valid_percentage / 100.0), 2)
    else:
        area_represented_km2 = calculated_area_km2

    if valid_count == 0:
        return {
            "valid_pixels": 0,
            "total_pixels": total_pixels,
            "valid_pixel_percentage": 0.0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "min_ndmi": None,
            "max_ndmi": None,
            "mean_ndmi": None,
            "median_ndmi": None,
            "std_dev": None,
            "area_represented_km2": 0.0,
            "moisture_signal_breakdown": {
                "very_low": {"pixels": 0, "percentage": 0.0},
                "low": {"pixels": 0, "percentage": 0.0},
                "moderate": {"pixels": 0, "percentage": 0.0},
                "high": {"pixels": 0, "percentage": 0.0},
                "very_high": {"pixels": 0, "percentage": 0.0},
            },
            "moisture_signal_distribution": {
                "very_low": {"pixels": 0, "percentage": 0.0},
                "low": {"pixels": 0, "percentage": 0.0},
                "moderate": {"pixels": 0, "percentage": 0.0},
                "high": {"pixels": 0, "percentage": 0.0},
                "very_high": {"pixels": 0, "percentage": 0.0},
            },
        }

    min_val = round(float(np.min(valid_vals)), 4)
    max_val = round(float(np.max(valid_vals)), 4)
    mean_val = round(float(np.mean(valid_vals)), 4)
    median_val = round(float(np.median(valid_vals)), 4)
    std_val = round(float(np.std(valid_vals)), 4)

    # Scientific moisture-related signal categories (not drought labels)
    vl_count = int(np.sum(valid_vals < -0.2))
    l_count = int(np.sum((valid_vals >= -0.2) & (valid_vals < 0.0)))
    m_count = int(np.sum((valid_vals >= 0.0) & (valid_vals < 0.2)))
    h_count = int(np.sum((valid_vals >= 0.2) & (valid_vals < 0.4)))
    vh_count = int(np.sum(valid_vals >= 0.4))

    breakdown = {
        "very_low": {
            "label": "Very Low Moisture Signal (NDMI < -0.2)",
            "pixels": vl_count,
            "percentage": round((vl_count / valid_count) * 100.0, 1),
        },
        "low": {
            "label": "Low Moisture Signal (-0.2 ≤ NDMI < 0.0)",
            "pixels": l_count,
            "percentage": round((l_count / valid_count) * 100.0, 1),
        },
        "moderate": {
            "label": "Moderate Moisture Signal (0.0 ≤ NDMI < 0.2)",
            "pixels": m_count,
            "percentage": round((m_count / valid_count) * 100.0, 1),
        },
        "high": {
            "label": "High Moisture Signal (0.2 ≤ NDMI < 0.4)",
            "pixels": h_count,
            "percentage": round((h_count / valid_count) * 100.0, 1),
        },
        "very_high": {
            "label": "Very High Moisture Signal (NDMI ≥ 0.4)",
            "pixels": vh_count,
            "percentage": round((vh_count / valid_count) * 100.0, 1),
        },
    }

    return {
        "valid_pixels": valid_count,
        "total_pixels": total_pixels,
        "valid_pixel_percentage": valid_percentage,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "median": median_val,
        "min_ndmi": min_val,
        "max_ndmi": max_val,
        "mean_ndmi": mean_val,
        "median_ndmi": median_val,
        "std_dev": std_val,
        "area_represented_km2": area_represented_km2,
        "moisture_signal_breakdown": breakdown,
        "moisture_signal_distribution": breakdown,
    }

def interpolate_ndmi_color(val: float) -> Tuple[int, int, int, int]:
    """
    Interpolates a continuous moisture-related colormap for NDMI (-0.6 to +0.6):
    - Very Low (Dry/Barren): Ochre / Rust-Brown [166, 97, 26, 255]
    - Low: Pale Tan / Sand [223, 194, 125, 255]
    - Moderate: Light Teal / Pale Cyan [128, 205, 193, 255]
    - High: Vivid Cyan / Aqua [53, 151, 143, 255]
    - Very High: Deep Marine / Royal Blue [1, 102, 94, 255]
    """
    if np.isnan(val):
        return (15, 23, 42, 0) # Transparent

    val = max(-0.6, min(0.6, val))
    # Normalize [-0.6, 0.6] to [0.0, 1.0]
    t = (val + 0.6) / 1.2

    if t < 0.25:
        # Ochre to Pale Sand
        sub_t = t / 0.25
        r = int(166 + sub_t * (223 - 166))
        g = int(97 + sub_t * (194 - 97))
        b = int(26 + sub_t * (125 - 26))
    elif t < 0.5:
        # Pale Sand to Pale Cyan
        sub_t = (t - 0.25) / 0.25
        r = int(223 + sub_t * (128 - 223))
        g = int(194 + sub_t * (205 - 194))
        b = int(125 + sub_t * (193 - 125))
    elif t < 0.75:
        # Pale Cyan to Vivid Teal
        sub_t = (t - 0.5) / 0.25
        r = int(128 + sub_t * (53 - 128))
        g = int(205 + sub_t * (151 - 205))
        b = int(193 + sub_t * (143 - 193))
    else:
        # Vivid Teal to Deep Marine Blue
        sub_t = (t - 0.75) / 0.25
        r = int(53 + sub_t * (1 - 53))
        g = int(151 + sub_t * (102 - 151))
        b = int(143 + sub_t * (94 - 143))

    return (r, g, b, 255)

def generate_ndmi_colormap(
    ndmi: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    is_demo: bool = False,
    output_size: Tuple[int, int] = (512, 512),
) -> str:
    """Renders NDMI array into a continuous RGBA visual map and returns base64 PNG."""
    ndmi_arr = np.asarray(ndmi, dtype=np.float64)
    h, w = ndmi_arr.shape

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        effective = np.where(mask, ndmi_arr, np.nan)
    else:
        effective = ndmi_arr

    flat_vals = effective.flatten()
    flat_colors = np.array([interpolate_ndmi_color(v) for v in flat_vals], dtype=np.uint8)
    rgba_grid = flat_colors.reshape((h, w, 4))

    img = Image.fromarray(rgba_grid, mode="RGBA")
    if (w, h) != output_size:
        img = img.resize(output_size, resample=Image.Resampling.BILINEAR)

    if is_demo:
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, output_size[0], 42], fill=(20, 20, 25, 230))
        draw.rectangle([0, 42, output_size[0], 44], fill=(234, 179, 8, 255))
        draw.text((36, 12), "DEMO DATA — NOT LIVE SATELLITE ANALYSIS", fill=(255, 255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
