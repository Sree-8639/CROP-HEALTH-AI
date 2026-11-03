import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional, Union, List

class RasterAlignmentError(ValueError):
    """Raised when raster bands cannot be properly aligned or resampled."""
    pass

def resample_band(
    arr: Union[np.ndarray, list],
    target_shape: Tuple[int, int],
    method: str = "bilinear",
    is_discrete: bool = False,
) -> np.ndarray:
    """
    Spatially resamples a 2D raster band to a target (height, width) grid
    while strictly preserving nodata / NaN values.

    Methods:
    - "bilinear": Standard smooth interpolation for continuous spectral reflectance.
    - "nearest": Nearest-neighbor interpolation for discrete classes (e.g., SCL, masks).

    Documentation of choice:
    Bilinear interpolation is the industry standard for continuous satellite surface reflectance
    (e.g., resampling 20 m bands B05/B11 to 10 m analysis grid) because it preserves local gradients
    without introducing staircase block artifacts. For categorical classification masks (SCL),
    nearest-neighbor is used to avoid inventing non-existent fractional class labels.
    """
    source_arr = np.asarray(arr, dtype=np.float32)

    if source_arr.ndim != 2:
        raise RasterAlignmentError(f"Expected 2D raster array, got ndim={source_arr.ndim}")

    target_h, target_w = target_shape
    src_h, src_w = source_arr.shape

    if (src_h, src_w) == (target_h, target_w):
        return source_arr.copy()

    # Identify and isolate NaNs to prevent corruption during interpolation
    nan_mask = np.isnan(source_arr)
    has_nans = bool(nan_mask.any())

    # Map method string to Pillow Resampling filter
    if is_discrete or method == "nearest":
        resample_filter = Image.Resampling.NEAREST
    elif method == "bicubic":
        resample_filter = Image.Resampling.BICUBIC
    else:
        resample_filter = Image.Resampling.BILINEAR

    # Replace NaNs with 0.0 temporarily for continuous interpolation
    clean_arr = np.nan_to_num(source_arr, nan=0.0)
    img = Image.fromarray(clean_arr, mode="F")

    # Pillow resize takes (width, height)
    resized_img = img.resize((target_w, target_h), resample=resample_filter)
    result = np.array(resized_img, dtype=np.float32)

    # Re-apply NaN mask using nearest-neighbor to ensure nodata boundaries remain exact
    if has_nans:
        mask_img = Image.fromarray(nan_mask.astype(np.uint8) * 255)
        resized_mask = np.array(mask_img.resize((target_w, target_h), resample=Image.Resampling.NEAREST)) > 0
        result[resized_mask] = np.nan

    return result

def align_bands(
    bands: Dict[str, np.ndarray],
    reference_band: str = "B08",
    target_band: Optional[str] = None,
    continuous_method: str = "bilinear",
) -> Dict[str, np.ndarray]:
    """
    Ensures all Sentinel-2 bands are geometrically aligned to a common analysis grid.
    If 20 m bands (B05, B11, SCL) have different dimensions from 10 m bands (B08, B04, B02),
    they are resampled to the reference band's grid.
    """
    if not bands:
        raise RasterAlignmentError("Band dictionary cannot be empty.")

    ref_key = target_band or reference_band

    # Determine reference shape (prefer B08 10m band)
    if ref_key in bands:
        target_shape = bands[ref_key].shape
    else:
        # Fall back to largest 2D dimension present in bands
        shapes = [v.shape for v in bands.values() if isinstance(v, np.ndarray) and v.ndim == 2]
        if not shapes:
            raise RasterAlignmentError("No 2D arrays found in bands dictionary.")
        target_shape = max(shapes, key=lambda s: s[0] * s[1])

    aligned: Dict[str, np.ndarray] = {}

    for name, arr in bands.items():
        is_discrete = name.upper() in ("SCL", "DATAMASK", "MASK", "VALID_MASK")
        aligned[name] = resample_band(
            arr,
            target_shape=target_shape,
            method="nearest" if is_discrete else continuous_method,
            is_discrete=is_discrete,
        )

    return aligned

def validate_band_shapes(*args, **kwargs) -> None:
    """
    Validates that all band arrays have identical 2D dimensions.
    Can be called with:
    - validate_band_shapes(arr1, arr2, ...)
    - validate_band_shapes({"B08": arr1, "B04": arr2})
    - validate_band_shapes(bands={"B08": arr1, "B04": arr2})
    """
    arrays: List[Tuple[str, np.ndarray]] = []
    if len(args) == 1 and isinstance(args[0], dict):
        arrays = list(args[0].items())
    elif "bands" in kwargs and isinstance(kwargs["bands"], dict):
        arrays = list(kwargs["bands"].items())
    elif args:
        arrays = [(f"Band_{i}", np.asarray(a)) for i, a in enumerate(args)]

    if not arrays:
        return

    shapes = {name: arr.shape for name, arr in arrays}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        raise RasterAlignmentError(
            f"Band shape mismatch detected before computation: {shapes}. Bands must be aligned first."
        )
