import math
from typing import Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
from scipy import ndimage

class RasterAlignmentError(ValueError):
    """Raised when geospatial raster alignment or reprojection fails."""
    pass

@dataclass
class CanonicalGrid:
    """
    Canonical spatial grid defining the standard coordinate reference system,
    resolution, dimensions, and affine transform for fused multimodal satellite features.
    """
    crs: str
    epsg: int
    width: int
    height: int
    resolution_m: float
    bounds: List[float]  # [min_x, min_y, max_x, max_y] in CRS units
    geo_bounds: List[float]  # [min_lon, min_lat, max_lon, max_lat] in WGS84
    transform: List[float]  # [a, b, c, d, e, f] where x = a*col + b*row + c, y = d*col + e*row + f
    geometry: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def calculate_utm_epsg(lon: float, lat: float) -> int:
    """
    Calculates the standard WGS84 UTM zone EPSG code for a given longitude and latitude.
    """
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone

def wgs84_to_utm_approx(lon: float, lat: float, utm_zone: int) -> Tuple[float, float]:
    """
    High-accuracy mathematical projection from WGS84 (lon, lat) in degrees
    to Universal Transverse Mercator (UTM) coordinates (easting, northing) in meters.
    Implements standard Karney / USGS ellipsoidal transverse mercator formulas.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f
    e_prime2 = e2 / (1.0 - e2)

    k0 = 0.9996
    lon0 = math.radians((utm_zone - 1) * 6.0 - 180.0 + 3.0)

    phi = math.radians(lat)
    lam = math.radians(lon)

    N = a / math.sqrt(1.0 - e2 * math.sin(phi) ** 2)
    T = math.tan(phi) ** 2
    C = e_prime2 * math.cos(phi) ** 2
    A = (lam - lon0) * math.cos(phi)

    # Meridional arc
    M = a * (
        (1.0 - e2 / 4.0 - 3.0 * e2 ** 2 / 64.0 - 5.0 * e2 ** 3 / 256.0) * phi
        - (3.0 * e2 / 8.0 + 3.0 * e2 ** 2 / 32.0 + 45.0 * e2 ** 3 / 1024.0) * math.sin(2.0 * phi)
        + (15.0 * e2 ** 2 / 256.0 + 45.0 * e2 ** 3 / 1024.0) * math.sin(4.0 * phi)
        - (35.0 * e2 ** 3 / 3072.0) * math.sin(6.0 * phi)
    )

    x = k0 * N * (
        A
        + (1.0 - T + C) * A ** 3 / 6.0
        + (5.0 - 18.0 * T + T ** 2 + 72.0 * C - 58.0 * e_prime2) * A ** 5 / 120.0
    ) + 500000.0  # False Easting

    y = k0 * (
        M
        + N
        * math.tan(phi)
        * (
            A ** 2 / 2.0
            + (5.0 - T + 9.0 * C + 4.0 * C ** 2) * A ** 4 / 24.0
            + (61.0 - 58.0 * T + T ** 2 + 600.0 * C - 330.0 * e_prime2) * A ** 6 / 720.0
        )
    )
    if lat < 0:
        y += 10000000.0  # False Northing for Southern hemisphere

    return x, y

def extract_bounds_from_geometry(geometry: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Extracts [min_lon, min_lat, max_lon, max_lat] from a GeoJSON Polygon or MultiPolygon.
    """
    coords: List[List[float]] = []
    geom_type = geometry.get("type", "")

    if geom_type == "Polygon":
        for ring in geometry.get("coordinates", []):
            coords.extend(ring)
    elif geom_type == "MultiPolygon":
        for poly in geometry.get("coordinates", []):
            for ring in poly:
                coords.extend(ring)
    else:
        raise RasterAlignmentError(f"Unsupported geometry type: {geom_type}")

    if not coords:
        raise RasterAlignmentError("Geometry contains no coordinate points.")

    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]

    return min(lons), min(lats), max(lons), max(lats)

def create_canonical_aoi_grid(
    geometry: Dict[str, Any],
    target_resolution_m: float = 10.0,
    min_pixels: int = 16,
    max_pixels: int = 512,
) -> CanonicalGrid:
    """
    Creates a standardized, geospatially registered CanonicalGrid for an AOI.
    Projects the geographic bounds into the local UTM zone at target ground resolution (e.g., 10m).
    """
    min_lon, min_lat, max_lon, max_lat = extract_bounds_from_geometry(geometry)
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0

    epsg = calculate_utm_epsg(center_lon, center_lat)
    utm_zone = epsg % 100
    crs_str = f"EPSG:{epsg} (WGS 84 / UTM Zone {utm_zone}{'N' if center_lat >= 0 else 'S'})"

    # Project bounding corners to UTM
    x_min, y_min = wgs84_to_utm_approx(min_lon, min_lat, utm_zone)
    x_max, y_max = wgs84_to_utm_approx(max_lon, max_lat, utm_zone)

    # Ensure min < max
    east_min = min(x_min, x_max)
    east_max = max(x_min, x_max)
    north_min = min(y_min, y_max)
    north_max = max(y_min, y_max)

    width_m = east_max - east_min
    height_m = north_max - north_min

    # Calculate grid raster dimensions
    raw_w = int(math.ceil(width_m / target_resolution_m))
    raw_h = int(math.ceil(height_m / target_resolution_m))

    width = max(min_pixels, min(max_pixels, raw_w))
    height = max(min_pixels, min(max_pixels, raw_h))

    # Adjust exact resolution to fit pixel count
    actual_res_x = width_m / width if width > 0 else target_resolution_m
    actual_res_y = height_m / height if height > 0 else target_resolution_m
    effective_resolution = round((actual_res_x + actual_res_y) / 2.0, 2)

    # Affine transform: [a, b, c, d, e, f]
    # x = c + a*col + b*row
    # y = f + d*col + e*row
    # Top-left corner is (east_min, north_max)
    transform = [
        actual_res_x,      # a: pixel width
        0.0,               # b: row rotation
        east_min,          # c: top-left x (easting)
        0.0,               # d: col rotation
        -actual_res_y,     # e: pixel height (negative going down)
        north_max,         # f: top-left y (northing)
    ]

    return CanonicalGrid(
        crs=crs_str,
        epsg=epsg,
        width=width,
        height=height,
        resolution_m=effective_resolution,
        bounds=[round(east_min, 2), round(north_min, 2), round(east_max, 2), round(north_max, 2)],
        geo_bounds=[round(min_lon, 6), round(min_lat, 6), round(max_lon, 6), round(max_lat, 6)],
        transform=[round(val, 4) for val in transform],
        geometry=geometry,
    )

def resample_raster_to_canonical_grid(
    source_array: np.ndarray,
    target_shape: Tuple[int, int],
    method: str = "bilinear",
    is_discrete: bool = False,
    nodata_value: Optional[float] = None,
) -> np.ndarray:
    """
    Resamples a 2D raster array to match the canonical grid shape (height, width)
    with strict preservation of NoData / NaN boundaries.

    Methods:
    - 'bilinear': Standard continuous interpolation for reflectance and SAR backscatter.
    - 'nearest': Preserves integer discrete categories (cloud mask, quality mask).
    - 'bicubic': Smooth higher-order interpolation.
    """
    arr = np.asarray(source_array, dtype=np.float32)
    if arr.ndim != 2:
        raise RasterAlignmentError(f"Expected 2D raster array, got ndim={arr.ndim} with shape {arr.shape}")

    target_h, target_w = target_shape
    src_h, src_w = arr.shape

    if (src_h, src_w) == (target_h, target_w):
        return arr.copy()

    # Track missing values
    if nodata_value is not None:
        nan_mask = np.isnan(arr) | (arr == nodata_value)
    else:
        nan_mask = np.isnan(arr)

    has_nans = bool(nan_mask.any())

    # Filter choice
    if is_discrete or method == "nearest":
        resample_filter = Image.Resampling.NEAREST
    elif method == "bicubic":
        resample_filter = Image.Resampling.BICUBIC
    else:
        resample_filter = Image.Resampling.BILINEAR

    # Clean array for interpolation
    clean_arr = np.nan_to_num(arr, nan=0.0)
    img = Image.fromarray(clean_arr, mode="F")
    resized_img = img.resize((target_w, target_h), resample=resample_filter)
    result = np.array(resized_img, dtype=np.float32)

    # Re-apply NoData mask exactly
    if has_nans:
        mask_img = Image.fromarray((nan_mask.astype(np.uint8) * 255))
        resized_mask = np.array(mask_img.resize((target_w, target_h), resample=Image.Resampling.NEAREST)) > 127
        result[resized_mask] = np.nan

    return result

def align_all_multimodal_features(
    optical_features: Dict[str, np.ndarray],
    sar_features: Dict[str, np.ndarray],
    canonical_grid: CanonicalGrid,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Aligns both optical spectral bands/indices and SAR backscatter channels
    onto the identical (height, width) of the canonical AOI grid.
    Guarantees shape and dimensional consistency for downstream multimodal AI tensors.
    """
    target_shape = (canonical_grid.height, canonical_grid.width)
    aligned_opt: Dict[str, np.ndarray] = {}
    aligned_sar: Dict[str, np.ndarray] = {}

    for name, arr in optical_features.items():
        is_discrete = name.upper() in ("SCL", "DATAMASK", "MASK", "QUALITY_MASK", "CLOUD_MASK")
        aligned_opt[name] = resample_raster_to_canonical_grid(
            arr,
            target_shape=target_shape,
            method="nearest" if is_discrete else "bilinear",
            is_discrete=is_discrete,
        )

    for name, arr in sar_features.items():
        is_discrete = name.upper() in ("SAR_MASK", "QUALITY_MASK")
        aligned_sar[name] = resample_raster_to_canonical_grid(
            arr,
            target_shape=target_shape,
            method="nearest" if is_discrete else "bilinear",
            is_discrete=is_discrete,
        )

    return aligned_opt, aligned_sar
