import io
import base64
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw

from services.raster_utils import resample_band

class NDREComputationError(Exception):
    """Raised when NDRE calculation or processing fails."""
    pass

def calculate_ndre(
    nir: Union[np.ndarray, list],
    red_edge: Union[np.ndarray, list],
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculates Normalized Difference Red Edge Index (NDRE) from Sentinel-2
    NIR (B08, 10m native) and Red-Edge (B05, 20m native) surface reflectance:

        NDRE = (B08 - B05) / (B08 + B05)

    The Red-Edge band (705 nm) penetrates deeper into dense crop canopies than
    red light (665 nm), avoiding early saturation in high-biomass crops.
    It is a red-edge vegetation indicator (NOT a definitive nitrogen proof).

    Spatial Resampling:
    B05 is natively 20 m resolution. It is explicitly resampled via bilinear
    interpolation to match the 10 m B08 analysis grid prior to calculation.
    """
    nir_arr = np.asarray(nir, dtype=np.float64)
    re_arr = np.asarray(red_edge, dtype=np.float64)

    # Resample B05 (20m) to match B08 (10m) if dimensions differ
    if re_arr.shape != nir_arr.shape:
        re_arr = resample_band(re_arr, target_shape=nir_arr.shape, method="bilinear").astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = nir_arr - re_arr
        denominator = nir_arr + re_arr

        invalid = (
            np.isnan(nir_arr)
            | np.isnan(re_arr)
            | np.isinf(nir_arr)
            | np.isinf(re_arr)
            | (nir_arr < 0.0)
            | (re_arr < 0.0)
            | np.isclose(denominator, 0.0, atol=1e-7)
        )

        ndre = np.where(invalid, np.nan, numerator / denominator)

        if valid_mask is not None:
            mask_arr = np.asarray(valid_mask, dtype=bool)
            if mask_arr.shape != ndre.shape:
                mask_arr = resample_band(mask_arr, target_shape=ndre.shape, is_discrete=True) > 0
            ndre = np.where(mask_arr, ndre, np.nan)

    return ndre

def calculate_ndre_statistics(
    ndre: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    pixel_size_m: float = 10.0,
    total_aoi_area_km2: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculates statistical metrics for NDRE red-edge measurements.
    """
    ndre_arr = np.asarray(ndre, dtype=np.float64)
    total_pixels = int(ndre_arr.size)

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool) & ~np.isnan(ndre_arr)
    else:
        mask = ~np.isnan(ndre_arr)

    valid_vals = ndre_arr[mask]
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
            "min_ndre": None,
            "max_ndre": None,
            "mean_ndre": None,
            "median_ndre": None,
            "std_dev": None,
            "area_represented_km2": 0.0,
            "red_edge_signal_breakdown": {
                "very_low": {"pixels": 0, "percentage": 0.0},
                "low": {"pixels": 0, "percentage": 0.0},
                "moderate": {"pixels": 0, "percentage": 0.0},
                "high": {"pixels": 0, "percentage": 0.0},
                "very_high": {"pixels": 0, "percentage": 0.0},
            },
            "red_edge_distribution": {
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

    # Scientific Red-Edge signal breakdown (typically ranges from ~ -0.1 to 0.6)
    vl_count = int(np.sum(valid_vals < 0.1))
    l_count = int(np.sum((valid_vals >= 0.1) & (valid_vals < 0.2)))
    m_count = int(np.sum((valid_vals >= 0.2) & (valid_vals < 0.35)))
    h_count = int(np.sum((valid_vals >= 0.35) & (valid_vals < 0.5)))
    vh_count = int(np.sum(valid_vals >= 0.5))

    breakdown = {
        "very_low": {
            "label": "Very Low Red-Edge Signal (NDRE < 0.1)",
            "pixels": vl_count,
            "percentage": round((vl_count / valid_count) * 100.0, 1),
        },
        "low": {
            "label": "Low Red-Edge Signal (0.1 ≤ NDRE < 0.2)",
            "pixels": l_count,
            "percentage": round((l_count / valid_count) * 100.0, 1),
        },
        "moderate": {
            "label": "Moderate Red-Edge Signal (0.2 ≤ NDRE < 0.35)",
            "pixels": m_count,
            "percentage": round((m_count / valid_count) * 100.0, 1),
        },
        "high": {
            "label": "High Red-Edge Signal (0.35 ≤ NDRE < 0.5)",
            "pixels": h_count,
            "percentage": round((h_count / valid_count) * 100.0, 1),
        },
        "very_high": {
            "label": "Very High Red-Edge Signal (NDRE ≥ 0.5)",
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
        "min_ndre": min_val,
        "max_ndre": max_val,
        "mean_ndre": mean_val,
        "median_ndre": median_val,
        "std_dev": std_val,
        "area_represented_km2": area_represented_km2,
        "red_edge_signal_breakdown": breakdown,
        "red_edge_distribution": breakdown,
    }

def interpolate_ndre_color(val: float) -> Tuple[int, int, int, int]:
    """
    Interpolates continuous Red-Edge chlorophyll signal colormap (0.0 to 0.6):
    - < 0.05: Russet/Copper [180, 50, 40, 255]
    - 0.05 - 0.20: Amber-Gold [230, 140, 40, 255] -> Yellow [245, 215, 60, 255]
    - 0.20 - 0.35: Chartreuse/Lime [160, 215, 70, 255]
    - 0.35 - 0.50: Emerald [50, 180, 95, 255]
    - > 0.50: Deep Forest Emerald [15, 110, 50, 255]
    """
    if np.isnan(val):
        return (15, 23, 42, 0)

    val = max(0.0, min(0.6, val))
    t = val / 0.6

    if t < 0.25:
        # Copper to Amber
        sub_t = t / 0.25
        r = int(180 + sub_t * (230 - 180))
        g = int(50 + sub_t * (140 - 50))
        b = int(40 + sub_t * (40 - 40))
    elif t < 0.5:
        # Amber to Yellow-Chartreuse
        sub_t = (t - 0.25) / 0.25
        r = int(230 + sub_t * (160 - 230))
        g = int(140 + sub_t * (215 - 140))
        b = int(40 + sub_t * (70 - 40))
    elif t < 0.75:
        # Chartreuse to Emerald
        sub_t = (t - 0.5) / 0.25
        r = int(160 + sub_t * (50 - 160))
        g = int(215 + sub_t * (180 - 215))
        b = int(70 + sub_t * (95 - 70))
    else:
        # Emerald to Deep Forest
        sub_t = (t - 0.75) / 0.25
        r = int(50 + sub_t * (15 - 50))
        g = int(180 + sub_t * (110 - 180))
        b = int(95 + sub_t * (50 - 95))

    return (r, g, b, 255)

def generate_ndre_colormap(
    ndre: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    is_demo: bool = False,
    output_size: Tuple[int, int] = (512, 512),
) -> str:
    """Renders NDRE array into a continuous RGBA visual map and returns base64 PNG."""
    ndre_arr = np.asarray(ndre, dtype=np.float64)
    h, w = ndre_arr.shape

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        effective = np.where(mask, ndre_arr, np.nan)
    else:
        effective = ndre_arr

    flat_vals = effective.flatten()
    flat_colors = np.array([interpolate_ndre_color(v) for v in flat_vals], dtype=np.uint8)
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
