import io
import base64
import math
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class NDVIComputationError(Exception):
    """Raised when NDVI calculation or spectral processing fails."""
    pass

def calculate_ndvi(
    red: Union[np.ndarray, list],
    nir: Union[np.ndarray, list],
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculates Normalized Difference Vegetation Index (NDVI) from Sentinel-2 Red (B04)
    and Near-Infrared (B08) reflectance arrays:

        NDVI = (B08 - B04) / (B08 + B04)

    Rules:
    1. Converts inputs to float64 for scientific numerical precision.
    2. Handles division by zero (sets resulting pixels to np.nan).
    3. Preserves nodata, negative reflectance, and invalid pixels as np.nan.
    4. Does not silently replace invalid values with zero.
    """
    red_arr = np.asarray(red, dtype=np.float64)
    nir_arr = np.asarray(nir, dtype=np.float64)

    if red_arr.shape != nir_arr.shape:
        raise ValueError(
            f"B04 (Red) and B08 (NIR) array shapes must match. Got {red_arr.shape} vs {nir_arr.shape}."
        )

    # Calculate numerator and denominator with NumPy warning suppression
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = nir_arr - red_arr
        denominator = nir_arr + red_arr

        # Identify invalid reflectance or zero division conditions
        # Real optical reflectance is non-negative.
        invalid_inputs = (
            np.isnan(red_arr)
            | np.isnan(nir_arr)
            | np.isinf(red_arr)
            | np.isinf(nir_arr)
            | (red_arr < 0.0)
            | (nir_arr < 0.0)
            | np.isclose(denominator, 0.0, atol=1e-7)
        )

        ndvi = np.where(invalid_inputs, np.nan, numerator / denominator)

        # Apply external valid pixel mask if provided
        if valid_mask is not None:
            mask_arr = np.asarray(valid_mask, dtype=bool)
            if mask_arr.shape != ndvi.shape:
                raise ValueError("valid_mask shape must match spectral arrays.")
            ndvi = np.where(mask_arr, ndvi, np.nan)

    return ndvi

def create_valid_pixel_mask(
    red: np.ndarray,
    nir: np.ndarray,
    scl: Optional[np.ndarray] = None,
    data_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Creates a boolean mask indicating strictly valid pixels for NDVI statistics.
    Excludes:
    - Nodata pixels (dataMask == 0 or NaNs)
    - Division by zero / negative reflectance
    - Clouds, cloud shadows, cirrus, snow (from Sentinel-2 Scene Classification Layer SCL)

    Sentinel-2 SCL Classes:
    0: NO_DATA, 1: SATURATED/DEFECTIVE, 2: DARK_AREA, 3: CLOUD_SHADOW,
    4: VEGETATION, 5: NOT_VEGETATED, 6: WATER, 7: UNCLASSIFIED,
    8: CLOUD_MEDIUM_PROB, 9: CLOUD_HIGH_PROB, 10: THIN_CIRRUS, 11: SNOW
    """
    red_arr = np.asarray(red, dtype=np.float64)
    nir_arr = np.asarray(nir, dtype=np.float64)

    # Basic spectral validity
    valid = (
        np.isfinite(red_arr)
        & np.isfinite(nir_arr)
        & (red_arr >= 0.0)
        & (nir_arr >= 0.0)
        & ((nir_arr + red_arr) > 1e-7)
    )

    # Exclude swath nodata
    if data_mask is not None:
        dm_arr = np.asarray(data_mask)
        valid = valid & (dm_arr > 0)

    # Exclude clouds, shadows, defective pixels via SCL
    if scl is not None:
        scl_arr = np.asarray(scl)
        # Invalid classes: 0 (No data), 1 (Defective), 3 (Shadow), 8 (Cloud med), 9 (Cloud high), 10 (Cirrus), 11 (Snow)
        cloud_or_invalid = np.isin(scl_arr, [0, 1, 3, 8, 9, 10, 11])
        valid = valid & (~cloud_or_invalid)

    return valid

def calculate_ndvi_statistics(
    ndvi: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    pixel_size_m: float = 10.0,
    total_aoi_area_km2: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculates comprehensive statistics for NDVI measurements.
    Statistics strictly exclude nodata, invalid, and cloud-masked pixels.
    """
    ndvi_arr = np.asarray(ndvi, dtype=np.float64)
    total_pixels = int(ndvi_arr.size)

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool) & ~np.isnan(ndvi_arr)
    else:
        mask = ~np.isnan(ndvi_arr)

    valid_vals = ndvi_arr[mask]
    valid_count = int(valid_vals.size)
    valid_percentage = round((valid_count / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0

    # Calculate approximate area in km²
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
            "min_ndvi": None,
            "max_ndvi": None,
            "mean_ndvi": None,
            "median_ndvi": None,
            "std_dev": None,
            "area_represented_km2": 0.0,
            "vegetation_vigor_breakdown": {
                "very_low": {"pixels": 0, "percentage": 0.0},
                "low": {"pixels": 0, "percentage": 0.0},
                "moderate": {"pixels": 0, "percentage": 0.0},
                "high": {"pixels": 0, "percentage": 0.0},
                "very_high": {"pixels": 0, "percentage": 0.0},
            },
        }

    # Statistical metrics
    min_val = round(float(np.min(valid_vals)), 4)
    max_val = round(float(np.max(valid_vals)), 4)
    mean_val = round(float(np.mean(valid_vals)), 4)
    median_val = round(float(np.median(valid_vals)), 4)
    std_val = round(float(np.std(valid_vals)), 4)

    # Scientific vegetation vigor breakdown based on NDVI measurements
    # (Described strictly as NDVI-based vegetation vigor, not crop health)
    vl_count = int(np.sum(valid_vals < 0.1))
    l_count = int(np.sum((valid_vals >= 0.1) & (valid_vals < 0.3)))
    m_count = int(np.sum((valid_vals >= 0.3) & (valid_vals < 0.5)))
    h_count = int(np.sum((valid_vals >= 0.5) & (valid_vals < 0.7)))
    vh_count = int(np.sum(valid_vals >= 0.7))

    breakdown = {
        "very_low": {
            "label": "Very Low Vegetation (NDVI < 0.1)",
            "pixels": vl_count,
            "percentage": round((vl_count / valid_count) * 100.0, 1),
        },
        "low": {
            "label": "Low Vegetation (0.1 ≤ NDVI < 0.3)",
            "pixels": l_count,
            "percentage": round((l_count / valid_count) * 100.0, 1),
        },
        "moderate": {
            "label": "Moderate Vegetation (0.3 ≤ NDVI < 0.5)",
            "pixels": m_count,
            "percentage": round((m_count / valid_count) * 100.0, 1),
        },
        "high": {
            "label": "High Vegetation (0.5 ≤ NDVI < 0.7)",
            "pixels": h_count,
            "percentage": round((h_count / valid_count) * 100.0, 1),
        },
        "very_high": {
            "label": "Very High Vegetation (NDVI ≥ 0.7)",
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
        "min_ndvi": min_val,
        "max_ndvi": max_val,
        "mean_ndvi": mean_val,
        "median_ndvi": median_val,
        "std_dev": std_val,
        "area_represented_km2": area_represented_km2,
        "vegetation_vigor_breakdown": breakdown,
    }

def interpolate_color(val: float) -> Tuple[int, int, int, int]:
    """
    Interpolates a continuous NDVI scientific colormap (-1 to +1):
    - < 0.0: Blue / Brown Water or Non-Vegetated [68, 1, 84, 255]
    - 0.0 - 0.2: Red / Ochre [215, 48, 39, 255] (Very Low)
    - 0.2 - 0.4: Amber / Yellow [253, 174, 97, 255] -> [255, 230, 110, 255] (Low)
    - 0.4 - 0.6: Yellow-Green to Light Green [166, 217, 106, 255] (Moderate)
    - 0.6 - 0.8: Emerald Green [102, 189, 99, 255] (High)
    - 0.8 - 1.0: Deep Lush Pine Green [26, 120, 50, 255] (Very High)
    """
    if np.isnan(val):
        return (15, 23, 42, 0) # Transparent

    # Clamp NDVI within valid range [-1.0, 1.0]
    val = max(-1.0, min(1.0, val))

    if val < 0.0:
        # Water / Non-vegetated dark blue/red
        t = (val + 1.0) # 0 to 1
        r = int(50 + t * 100)
        g = int(70 + t * 50)
        b = int(140 - t * 40)
        return (r, g, b, 240)
    elif val < 0.2:
        # Very Low: Brown/Red to Ochre
        t = val / 0.2
        r = int(215 + t * (244 - 215))
        g = int(48 + t * (109 - 48))
        b = int(39 + t * (67 - 39))
        return (r, g, b, 255)
    elif val < 0.4:
        # Low: Ochre to Bright Yellow
        t = (val - 0.2) / 0.2
        r = int(244 + t * (255 - 244))
        g = int(109 + t * (225 - 109))
        b = int(67 + t * (80 - 67))
        return (r, g, b, 255)
    elif val < 0.6:
        # Moderate: Yellow to Lime Green
        t = (val - 0.4) / 0.2
        r = int(255 + t * (140 - 255))
        g = int(225 + t * (210 - 225))
        b = int(80 + t * (85 - 80))
        return (r, g, b, 255)
    elif val < 0.8:
        # High: Lime Green to Emerald
        t = (val - 0.6) / 0.2
        r = int(140 + t * (35 - 140))
        g = int(210 + t * (165 - 210))
        b = int(85 + t * (60 - 85))
        return (r, g, b, 255)
    else:
        # Very High: Emerald to Deep Lush Green
        t = (val - 0.8) / 0.2
        r = int(35 + t * (10 - 35))
        g = int(165 + t * (110 - 165))
        b = int(60 + t * (30 - 60))
        return (r, g, b, 255)

def generate_ndvi_colormap(
    ndvi: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    is_demo: bool = False,
    output_size: Tuple[int, int] = (512, 512),
) -> str:
    """
    Renders an NDVI array into a continuous RGBA visual map and returns
    a base64-encoded PNG Data URL.
    """
    ndvi_arr = np.asarray(ndvi, dtype=np.float64)
    h, w = ndvi_arr.shape

    # Apply mask if given
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        effective_ndvi = np.where(mask, ndvi_arr, np.nan)
    else:
        effective_ndvi = ndvi_arr

    # Vectorized color mapping
    rgba_grid = np.zeros((h, w, 4), dtype=np.uint8)

    # For speed and precision on 2D arrays:
    flat_vals = effective_ndvi.flatten()
    flat_colors = np.array([interpolate_color(v) for v in flat_vals], dtype=np.uint8)
    rgba_grid = flat_colors.reshape((h, w, 4))

    img = Image.fromarray(rgba_grid, mode="RGBA")

    # Resize to smooth high-resolution display if needed
    if (w, h) != output_size:
        img = img.resize(output_size, resample=Image.Resampling.BILINEAR)

    # Stamp DEMO banner if in demo mode
    if is_demo:
        draw = ImageDraw.Draw(img)
        # Top banner
        draw.rectangle([0, 0, output_size[0], 42], fill=(20, 20, 25, 230))
        draw.rectangle([0, 42, output_size[0], 44], fill=(234, 179, 8, 255))
        draw.text((36, 12), "DEMO DATA — NOT LIVE SATELLITE ANALYSIS", fill=(255, 255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def generate_demo_spectral_bands(
    start_date: str,
    end_date: str,
    shape: Tuple[int, int] = (128, 128),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates realistic synthetic B04 (Red) and B08 (NIR) arrays and SCL classes
    so that DEMO_MODE actually computes NDVI through the real calculation pipeline.
    """
    h, w = shape
    seed_val = sum(ord(c) for c in f"{start_date}_{end_date}") % (2**31 - 1)
    import random
    rng = random.Random(seed_val)
    np_rng = np.random.RandomState(seed_val)

    b04 = np.zeros((h, w), dtype=np.float32)
    b08 = np.zeros((h, w), dtype=np.float32)
    scl = np.full((h, w), 4, dtype=np.uint8) # 4 = vegetation
    data_mask = np.ones((h, w), dtype=np.uint8)

    grid_h = h // 8
    grid_w = w // 8

    # Field archetypes: (Red, NIR, SCL)
    archetypes = [
        (0.04, 0.58, 4),  # Dense lush green crop (NDVI ~ 0.87)
        (0.06, 0.52, 4),  # Healthy crop (NDVI ~ 0.79)
        (0.08, 0.42, 4),  # Moderate vigor vegetation (NDVI ~ 0.68)
        (0.12, 0.32, 4),  # Growing / lower density vegetation (NDVI ~ 0.45)
        (0.16, 0.24, 5),  # Emerging crop / dry vegetation (NDVI ~ 0.20, SCL 5 Not Veg)
        (0.24, 0.28, 5),  # Fallow / Bare tilled soil (NDVI ~ 0.08, SCL 5 Not Veg)
    ]

    for r in range(8):
        for c in range(8):
            base_red, base_nir, base_scl = rng.choice(archetypes)
            y1, y2 = r * grid_h, (r + 1) * grid_h
            x1, x2 = c * grid_w, (c + 1) * grid_w

            # Add subtle natural texture variation per field
            noise_red = np_rng.uniform(-0.01, 0.01, size=(y2 - y1, x2 - x1))
            noise_nir = np_rng.uniform(-0.02, 0.02, size=(y2 - y1, x2 - x1))

            b04[y1:y2, x1:x2] = np.clip(base_red + noise_red, 0.02, 0.40)
            b08[y1:y2, x1:x2] = np.clip(base_nir + noise_nir, 0.15, 0.80)
            scl[y1:y2, x1:x2] = base_scl

    # Add an irrigation canal (water: high Red relative to NIR -> negative NDVI, SCL 6)
    for x in range(w):
        y = int(h * 0.35 + math.sin(x / 12.0) * 6)
        if 0 <= y < h:
            y_min, y_max = max(0, y - 2), min(h, y + 3)
            b04[y_min:y_max, x] = 0.06
            b08[y_min:y_max, x] = 0.02
            scl[y_min:y_max, x] = 6 # Water

    # Add a small cloud cluster in top corner to test cloud exclusion
    b04[0:14, 0:16] = 0.52
    b08[0:14, 0:16] = 0.55
    scl[0:14, 0:16] = 9 # High probability cloud

    return b04, b08, scl, data_mask

def generate_all_demo_spectral_bands(
    start_date: str = "2026-06-01",
    end_date: str = "2026-06-15",
    shape: Tuple[int, int] = (128, 128),
    width: Optional[int] = None,
    height: Optional[int] = None,
    include_b03: bool = False,
) -> Dict[str, np.ndarray]:
    if width is not None and height is not None:
        shape = (height, width)
    """
    Generates realistic synthetic Sentinel-2 Level-2A surface reflectance arrays
    for all key agricultural bands:
    - B02 (Blue, 10m)
    - B04 (Red, 10m)
    - B05 (RedEdge, 20m - returned at native 20m grid)
    - B08 (NIR, 10m)
    - B11 (SWIR, 20m - returned at native 20m grid)
    - SCL (Scene Classification Layer, 20m)
    - data_mask (Swath mask, 10m)
    """
    h, w = shape
    # Use a seeded RandomState for deterministic values across repeated calls
    seed_val = sum(ord(c) for c in f"{start_date}_{end_date}") % (2**31 - 1)
    np_rng = np.random.RandomState(seed_val)

    b04, b08, scl, data_mask = generate_demo_spectral_bands(start_date, end_date, shape=shape)

    # Generate B02 (Blue 10m, correlated with Red but lower in soil/vegetation)
    b02 = np.clip(b04 * 0.65 + np_rng.uniform(-0.01, 0.01, (h, w)), 0.01, 0.35).astype(np.float32)

    # Generate B03 (Green 10m, prominent reflectance in healthy photosynthetic canopy)
    b03 = np.clip(b04 * 0.70 + b08 * 0.18 + np_rng.uniform(-0.01, 0.01, (h, w)), 0.02, 0.45).astype(np.float32)

    # Subsample reference grids for realistic correlation at half resolution (20m)
    b04_sub = b04[::2, ::2]
    b08_sub = b08[::2, ::2]
    h20, w20 = b04_sub.shape

    # B05 is situated on the steep red-edge slope between B04 and B08: ~ B04 + 0.35 * (B08 - B04)
    b05_20m = np.clip(b04_sub + 0.35 * (b08_sub - b04_sub) + np_rng.uniform(-0.01, 0.01, (h20, w20)), 0.03, 0.50)

    # B11 (SWIR): absorbed by leaf water; high in dry soil, lower in hydrated vegetation
    # For vegetation: B11 is significantly lower than B08. For soil: B11 >= B08
    b11_20m = np.clip(b04_sub * 1.1 + b08_sub * 0.25 + np_rng.uniform(-0.02, 0.02, (h20, w20)), 0.02, 0.45)

    res = {
        "B02": b02,
        "B04": b04,
        "B05": b05_20m.astype(np.float32),
        "B08": b08,
        "B11": b11_20m.astype(np.float32),
        "SCL": scl,
        "dataMask": data_mask,
    }
    if include_b03:
        res["B03"] = b03
    return res

