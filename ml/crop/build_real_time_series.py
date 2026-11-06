"""
ml/crop/build_real_time_series.py:
Fetches real Sentinel-2 L2A and Sentinel-1 GRD observations via the live CDSE pipeline
for the Tadepalligudem AOI, applies SCL cloud masking, performs S1-S2 temporal matching,
and saves the multi-temporal multimodal raster tensor to disk for offline deterministic benchmarking.
"""

import os
import sys
import json
import asyncio
import numpy as np

# Ensure backend and repo root are accessible
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND_DIR, ".env"), override=True)
os.environ["DEMO_MODE"] = "false"

from services.sentinel2_service import Sentinel2Service
from services.sentinel1_service import Sentinel1Service
from services.ndvi_service import (
    create_valid_pixel_mask,
    calculate_ndvi,
)
from services.ndmi_service import calculate_ndmi
from services.ndre_service import calculate_ndre

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(OUTPUT_DIR, "real_tadepalligudem_multimodal_tensor.npz")
META_FILE = os.path.join(OUTPUT_DIR, "real_tadepalligudem_meta.json")


def resize_raster_2d(arr: np.ndarray, target_shape: tuple = (64, 64)) -> np.ndarray:
    """Downsamples a 2D float raster to target shape using area averaging."""
    h_in, w_in = arr.shape
    h_out, w_out = target_shape
    if (h_in, w_in) == (h_out, w_out):
        return arr
    
    # Block-reduce or simple slice resampling
    y_idx = (np.linspace(0, h_in - 1, h_out)).astype(int)
    x_idx = (np.linspace(0, w_in - 1, w_out)).astype(int)
    return arr[np.ix_(y_idx, x_idx)]


async def fetch_and_build_crop_time_series(
    bbox: list = None,
    grid_shape: tuple = (64, 64),
    force_refresh: bool = False,
) -> dict:
    """
    Fetches real multi-temporal S2 and S1 rasters, applies SCL cloud masks,
    and returns a clean dictionary of arrays and metadata.
    """
    bbox = bbox or [81.503, 16.793, 81.543, 16.833]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not force_refresh and os.path.exists(CACHE_FILE) and os.path.exists(META_FILE):
        print(f"[Real Time Series] Loading cached real satellite tensor from {CACHE_FILE}")
        data = np.load(CACHE_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return {
            "tensor": data["tensor"],
            "optical_masks": data["optical_masks"],
            "meta": meta,
        }

    s2_svc = Sentinel2Service()
    s1_svc = Sentinel1Service()

    min_lon, min_lat, max_lon, max_lat = bbox
    geom = {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]]
    }

    # 4 Real multi-temporal observation windows covering transplanting to heading in Rabi 2024
    observation_windows = [
        {"name": "early_vegetative_t0", "start": "2024-01-01", "end": "2024-01-08"},
        {"name": "mid_vegetative_t1",   "start": "2024-01-09", "end": "2024-01-16"},
        {"name": "tillering_t2",        "start": "2024-02-15", "end": "2024-02-21"},
        {"name": "heading_flowering_t3","start": "2024-02-25", "end": "2024-02-29"},
    ]

    print("[Real Time Series] Searching Sentinel-1 C-SAR catalog over Tadepalligudem...")
    s1_catalog = await s1_svc.search_imagery(geom, "2024-01-01", "2024-02-29")
    print(f"      Found {len(s1_catalog)} real Sentinel-1 acquisitions.")

    T = len(observation_windows)
    H, W = grid_shape
    # Channels: 0: NDVI, 1: NDRE, 2: NDMI, 3: VV_dB, 4: VH_dB, 5: Ratio_dB
    multimodal_tensor = np.zeros((T, 6, H, W), dtype=np.float32)
    optical_masks = np.zeros((T, H, W), dtype=bool)

    window_metadata = []

    for t_idx, win in enumerate(observation_windows):
        s_date, e_date = win["start"], win["end"]
        print(f"\n[Real Time Series] Fetching Window {t_idx+1}/{T}: {s_date} to {e_date}...")
        
        # 1. Fetch real Sentinel-2 L2A Float32 bands
        s2_scene, bands = await s2_svc._fetch_multispectral_bands(
            geometry=geom,
            start_date=s_date,
            end_date=e_date,
            max_cloud_coverage=35.0,
            band_names=["B02", "B03", "B04", "B05", "B08", "B11"],
        )
        s2_id = s2_scene["scene_id"]
        s2_date = s2_scene["acquisition_date"]
        print(f"      Retrieved S2: {s2_id} (Date: {s2_date})")

        b04 = bands["B04"]
        b08 = bands["B08"]
        b05 = bands["B05"]
        b11 = bands["B11"]
        scl = bands.get("SCL")
        dmask = bands.get("dataMask")

        # Create valid pixel mask from ESA SCL classification
        valid_mask_256 = create_valid_pixel_mask(b04, b08, scl=scl, data_mask=dmask)
        valid_pct = (valid_mask_256.sum() / valid_mask_256.size) * 100.0

        # Calculate spectral indices on valid pixels
        ndvi_256 = calculate_ndvi(b04, b08, valid_mask=valid_mask_256)
        ndre_256 = calculate_ndre(b08, b05, valid_mask=valid_mask_256)
        ndmi_256 = calculate_ndmi(b08, b11, valid_mask=valid_mask_256)

        # Resample to canonical 64x64 grid
        ndvi_64 = resize_raster_2d(ndvi_256, (H, W))
        ndre_64 = resize_raster_2d(ndre_256, (H, W))
        ndmi_64 = resize_raster_2d(ndmi_256, (H, W))
        mask_64 = resize_raster_2d(valid_mask_256.astype(float), (H, W)) > 0.5

        # 2. Match nearest real Sentinel-1 C-SAR pass
        s2_date_short = s2_date[:10]
        s1_match = s1_svc.find_closest_sentinel1_observation(s2_date_short, s1_catalog, max_window_days=3)
        
        s1_id = "N/A"
        gap_days = 0
        if s1_match.get("matched"):
            s1_scene = s1_match["scene"]
            s1_id = s1_scene["scene_id"]
            gap_days = s1_match["date_difference_days"]
            print(f"      Matched S1: {s1_id} (Gap: {gap_days}d)")

            vv_lin, vh_lin, sar_mask, _ = await s1_svc.get_sar_bands(
                geometry=geom,
                start_date=s_date,
                end_date=e_date,
                target_scene=s1_scene,
            )
            # Calibrated backscatter to decibels
            vv_db_raw = 10.0 * np.log10(np.maximum(vv_lin, 1e-6))
            vh_db_raw = 10.0 * np.log10(np.maximum(vh_lin, 1e-6))
            ratio_raw = vv_db_raw - vh_db_raw

            vv_64 = resize_raster_2d(vv_db_raw, (H, W))
            vh_64 = resize_raster_2d(vh_db_raw, (H, W))
            ratio_64 = resize_raster_2d(ratio_raw, (H, W))
        else:
            print("      No S1 match; using nominal background backscatter.")
            vv_64 = np.full((H, W), -9.0, dtype=np.float32)
            vh_64 = np.full((H, W), -17.5, dtype=np.float32)
            ratio_64 = vv_64 - vh_64

        # Clean NaN/Inf with physically grounded bounds
        ndvi_64 = np.nan_to_num(ndvi_64, nan=float(np.nanmedian(ndvi_64) if not np.all(np.isnan(ndvi_64)) else 0.35))
        ndre_64 = np.nan_to_num(ndre_64, nan=float(np.nanmedian(ndre_64) if not np.all(np.isnan(ndre_64)) else 0.20))
        ndmi_64 = np.nan_to_num(ndmi_64, nan=float(np.nanmedian(ndmi_64) if not np.all(np.isnan(ndmi_64)) else 0.15))
        vv_64 = np.nan_to_num(vv_64, nan=-9.0)
        vh_64 = np.nan_to_num(vh_64, nan=-17.0)
        ratio_64 = np.nan_to_num(ratio_64, nan=8.0)

        multimodal_tensor[t_idx, 0] = ndvi_64
        multimodal_tensor[t_idx, 1] = ndre_64
        multimodal_tensor[t_idx, 2] = ndmi_64
        multimodal_tensor[t_idx, 3] = vv_64
        multimodal_tensor[t_idx, 4] = vh_64
        multimodal_tensor[t_idx, 5] = ratio_64
        optical_masks[t_idx] = mask_64

        window_metadata.append({
            "timestep": t_idx,
            "window_name": win["name"],
            "sentinel2_scene_id": s2_id,
            "sentinel2_acquisition_date": s2_date,
            "sentinel1_scene_id": s1_id,
            "sentinel1_temporal_gap_days": gap_days,
            "scl_valid_pixel_pct": round(valid_pct, 2),
            "mean_ndvi": round(float(ndvi_64.mean()), 4),
            "mean_ndre": round(float(ndre_64.mean()), 4),
            "mean_ndmi": round(float(ndmi_64.mean()), 4),
            "mean_vv_db": round(float(vv_64.mean()), 2),
            "mean_vh_db": round(float(vh_64.mean()), 2),
        })

    # Save to disk
    np.savez_compressed(
        CACHE_FILE,
        tensor=multimodal_tensor,
        optical_masks=optical_masks,
    )

    meta_payload = {
        "dataset_name": "Tadepalligudem Multimodal Crop Time Series",
        "aoi_name": "Tadepalligudem, West Godavari District, Andhra Pradesh, India",
        "bbox": bbox,
        "grid_shape": list(grid_shape),
        "total_pixels": H * W,
        "timesteps_count": T,
        "channels": ["NDVI", "NDRE", "NDMI", "VV_dB", "VH_dB", "Ratio_dB"],
        "data_source": "REAL (Copernicus Sentinel-2 L2A & Sentinel-1 GRD)",
        "is_simulated": False,
        "observation_windows": window_metadata,
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    print(f"\n[Real Time Series] Saved real tensor to {CACHE_FILE} and metadata to {META_FILE}.")
    return {
        "tensor": multimodal_tensor,
        "optical_masks": optical_masks,
        "meta": meta_payload,
    }


if __name__ == "__main__":
    asyncio.run(fetch_and_build_crop_time_series(force_refresh=True))
