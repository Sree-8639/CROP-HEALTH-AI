import io
import base64
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw

from services.raster_utils import resample_band

class EVIComputationError(Exception):
    """Raised when EVI calculation or processing fails."""
    pass

def calculate_evi(
    blue: Union[np.ndarray, list],
    red: Union[np.ndarray, list],
    nir: Union[np.ndarray, list],
    valid_mask: Optional[np.ndarray] = None,
    g: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    l: float = 1.0,
) -> np.ndarray:
    """
    Calculates Enhanced Vegetation Index (EVI) from Sentinel-2
    Blue (B02), Red (B04), and NIR (B08) surface reflectance:

        EVI = G * (NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L)

    Default coefficients: G=2.5, C1=6.0, C2=7.5, L=1.0

    EVI optimizes the vegetation signal with improved sensitivity in
    high biomass regions and decouples background canopy soil signals and
    atmospheric aerosol scattering influences.
    """
    b_arr = np.asarray(blue, dtype=np.float64)
    r_arr = np.asarray(red, dtype=np.float64)
    n_arr = np.asarray(nir, dtype=np.float64)

    # Validate shapes match
    if not (b_arr.shape == r_arr.shape == n_arr.shape):
        target_shape = n_arr.shape
        if b_arr.shape != target_shape:
            b_arr = resample_band(b_arr, target_shape=target_shape, method="bilinear").astype(np.float64)
        if r_arr.shape != target_shape:
            r_arr = resample_band(r_arr, target_shape=target_shape, method="bilinear").astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = g * (n_arr - r_arr)
        denominator = n_arr + (c1 * r_arr) - (c2 * b_arr) + l

        invalid = (
            np.isnan(b_arr)
            | np.isnan(r_arr)
            | np.isnan(n_arr)
            | np.isinf(b_arr)
            | np.isinf(r_arr)
            | np.isinf(n_arr)
            | (b_arr < 0.0)
            | (r_arr < 0.0)
            | (denominator <= 1e-6)
            | np.isclose(denominator, 0.0, atol=1e-7)
        )

        evi = np.where(invalid, np.nan, numerator / denominator)

        if valid_mask is not None:
            mask_arr = np.asarray(valid_mask, dtype=bool)
            if mask_arr.shape != evi.shape:
                mask_arr = resample_band(mask_arr, target_shape=evi.shape, is_discrete=True) > 0
            evi = np.where(mask_arr, evi, np.nan)

    return evi

def calculate_evi_statistics(
    evi: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    pixel_size_m: float = 10.0,
    total_aoi_area_km2: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculates statistical metrics for EVI enhanced vegetation measurements.
    """
    evi_arr = np.asarray(evi, dtype=np.float64)
    total_pixels = int(evi_arr.size)

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool) & ~np.isnan(evi_arr)
    else:
        mask = ~np.isnan(evi_arr)

    valid_vals = evi_arr[mask]
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
            "min_evi": None,
            "max_evi": None,
            "mean_evi": None,
            "median_evi": None,
            "std_dev": None,
            "area_represented_km2": 0.0,
            "enhanced_vegetation_breakdown": {
                "very_low": {"pixels": 0, "percentage": 0.0},
                "low": {"pixels": 0, "percentage": 0.0},
                "moderate": {"pixels": 0, "percentage": 0.0},
                "high": {"pixels": 0, "percentage": 0.0},
                "very_high": {"pixels": 0, "percentage": 0.0},
            },
            "canopy_signal_distribution": {
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

    # Scientific Enhanced Vegetation distribution breakdown (EVI typically ~ 0 to 0.8)
    vl_count = int(np.sum(valid_vals < 0.1))
    l_count = int(np.sum((valid_vals >= 0.1) & (valid_vals < 0.2)))
    m_count = int(np.sum((valid_vals >= 0.2) & (valid_vals < 0.4)))
    h_count = int(np.sum((valid_vals >= 0.4) & (valid_vals < 0.6)))
    vh_count = int(np.sum(valid_vals >= 0.6))

    breakdown = {
        "very_low": {
            "label": "Very Low Vegetation Signal (EVI < 0.1)",
            "pixels": vl_count,
            "percentage": round((vl_count / valid_count) * 100.0, 1),
        },
        "low": {
            "label": "Low Vegetation Signal (0.1 ≤ EVI < 0.2)",
            "pixels": l_count,
            "percentage": round((l_count / valid_count) * 100.0, 1),
        },
        "moderate": {
            "label": "Moderate Vegetation Signal (0.2 ≤ EVI < 0.4)",
            "pixels": m_count,
            "percentage": round((m_count / valid_count) * 100.0, 1),
        },
        "high": {
            "label": "High Vegetation Signal (0.4 ≤ EVI < 0.6)",
            "pixels": h_count,
            "percentage": round((h_count / valid_count) * 100.0, 1),
        },
        "very_high": {
            "label": "Very High Vegetation Signal (EVI ≥ 0.6)",
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
        "min_evi": min_val,
        "max_evi": max_val,
        "mean_evi": mean_val,
        "median_evi": median_val,
        "std_dev": std_val,
        "area_represented_km2": area_represented_km2,
        "enhanced_vegetation_breakdown": breakdown,
        "canopy_signal_distribution": breakdown,
    }

def interpolate_evi_color(val: float) -> Tuple[int, int, int, int]:
    """
    Interpolates continuous Enhanced Vegetation Index colormap (0.0 to 0.8):
    - < 0.05: Earth Brown / Sandy Tan [190, 150, 100, 255]
    - 0.05 - 0.20: Pale Gold / Wheat [235, 205, 110, 255]
    - 0.20 - 0.40: Yellow-Green / Celadon [165, 215, 95, 255]
    - 0.40 - 0.60: Vibrant Leaf Green [65, 185, 75, 255]
    - > 0.60: Deep Canopy Emerald [15, 115, 45, 255]
    """
    if np.isnan(val):
        return (15, 23, 42, 0)

    val = max(0.0, min(0.8, val))
    t = val / 0.8

    if t < 0.25:
        # Tan to Pale Wheat
        sub_t = t / 0.25
        r = int(190 + sub_t * (235 - 190))
        g = int(150 + sub_t * (205 - 150))
        b = int(100 + sub_t * (110 - 100))
    elif t < 0.5:
        # Pale Wheat to Yellow-Green
        sub_t = (t - 0.25) / 0.25
        r = int(235 + sub_t * (165 - 235))
        g = int(205 + sub_t * (215 - 205))
        b = int(110 + sub_t * (95 - 110))
    elif t < 0.75:
        # Yellow-Green to Leaf Green
        sub_t = (t - 0.5) / 0.25
        r = int(165 + sub_t * (65 - 165))
        g = int(215 + sub_t * (185 - 215))
        b = int(95 + sub_t * (75 - 95))
    else:
        # Leaf Green to Deep Canopy Emerald
        sub_t = (t - 0.75) / 0.25
        r = int(65 + sub_t * (15 - 65))
        g = int(185 + sub_t * (115 - 185))
        b = int(75 + sub_t * (45 - 75))

    return (r, g, b, 255)

def generate_evi_colormap(
    evi: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    is_demo: bool = False,
    output_size: Tuple[int, int] = (512, 512),
) -> str:
    """Renders EVI array into a continuous RGBA visual map and returns base64 PNG."""
    evi_arr = np.asarray(evi, dtype=np.float64)
    h, w = evi_arr.shape

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        effective = np.where(mask, evi_arr, np.nan)
    else:
        effective = evi_arr

    flat_vals = effective.flatten()
    flat_colors = np.array([interpolate_evi_color(v) for v in flat_vals], dtype=np.uint8)
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
