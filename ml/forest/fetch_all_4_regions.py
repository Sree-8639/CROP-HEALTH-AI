"""
ml/forest/fetch_all_4_regions.py:
Fetches and aligns all 4 real forest benchmark regions:
1. Eastern Ghats (Maredumilli / Papikonda Corridor, AP) - 2020-2021
2. Hasdeo Arand PEKB Coal Block (Chhattisgarh) - 2020-2021
3. Western Ghats (Wayanad / Nilgiris Corridor, Kerala/TN) - 2021-2022
4. Dandeli-Anshi / Kali Tiger Reserve Corridor (Karnataka) - 2021-2022

Saves all 4 regions into ml/forest/data/real_forest_benchmark.npz and real_forest_meta.json.
"""

import os
import sys
import json
import asyncio
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import cv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import ml
if hasattr(ml, "__path__"):
    ml_backend_dir = os.path.join(BACKEND_DIR, "ml")
    if ml_backend_dir not in ml.__path__:
        ml.__path__.append(ml_backend_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND_DIR, ".env"), override=True)
os.environ["DEMO_MODE"] = "false"

from services.sentinel2_service import Sentinel2Service
from services.sentinel1_service import Sentinel1Service

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(OUTPUT_DIR, "real_forest_benchmark.npz")
META_FILE = os.path.join(OUTPUT_DIR, "real_forest_meta.json")

REGIONS = [
    {
        "id": "prim",
        "name": "Eastern Ghats Forest Corridor (Maredumilli Reserve), Andhra Pradesh, India",
        "bbox": [81.55, 17.45, 81.65, 17.55],
        "hansen_tile": "20N_080E",
        "years": (2020, 2021),
        "t0_s2": ("2020-01-01", "2020-01-20"),
        "t1_s2": ("2021-01-01", "2021-01-20"),
        "loss_years": (20, 21),
    },
    {
        "id": "hasdeo",
        "name": "Hasdeo Arand Forest (Parsa East & Kete Basan Sector), Chhattisgarh, India",
        "bbox": [82.75, 22.75, 82.85, 22.85],
        "hansen_tile": "30N_080E",
        "years": (2020, 2021),
        "t0_s2": ("2020-01-01", "2020-01-20"),
        "t1_s2": ("2021-01-01", "2021-01-20"),
        "loss_years": (20, 21),
    },
    {
        "id": "wayanad",
        "name": "Western Ghats / Wayanad-Nilgiris Montane Corridor, Kerala/Tamil Nadu, India",
        "bbox": [76.15, 11.60, 76.25, 11.70],
        "hansen_tile": "20N_070E",
        "years": (2021, 2022),
        "t0_s2": ("2021-01-01", "2021-02-28"),
        "t1_s2": ("2022-01-01", "2022-02-28"),
        "loss_years": (21, 22),
    },
    {
        "id": "dandeli",
        "name": "Dandeli-Anshi / Kali Tiger Reserve Corridor, Karnataka, India",
        "bbox": [74.65, 15.20, 74.75, 15.30],
        "hansen_tile": "20N_070E",
        "years": (2021, 2022),
        "t0_s2": ("2021-01-01", "2021-02-28"),
        "t1_s2": ("2022-01-01", "2022-02-28"),
        "loss_years": (21, 22),
    },
]


def fetch_hansen_window(bbox: list, tile_code: str, loss_years: tuple, target_shape=(512, 512)):
    min_lon, min_lat, max_lon, max_lat = bbox
    url_loss = f"https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/Hansen_GFC-2023-v1.11_lossyear_{tile_code}.tif"
    url_tc = f"https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/Hansen_GFC-2023-v1.11_treecover2000_{tile_code}.tif"

    print(f"  [Hansen GFC v1.11] Window from {tile_code} for bbox {bbox}...")
    with rasterio.open(url_loss) as src_loss, rasterio.open(url_tc) as src_tc:
        win = from_bounds(min_lon, min_lat, max_lon, max_lat, src_loss.transform)
        raw_lossyear = src_loss.read(1, window=win)
        raw_treecover = src_tc.read(1, window=win)

    lossyear = cv2.resize(raw_lossyear, target_shape, interpolation=cv2.INTER_NEAREST)
    treecover = cv2.resize(raw_treecover, target_shape, interpolation=cv2.INTER_NEAREST)

    y_min, y_max = loss_years
    loss_mask = ((lossyear >= y_min) & (lossyear <= y_max) & (treecover >= 30)).astype(np.uint8)
    print(f"    Canopy mean: {np.mean(treecover):.1f}%, Loss pixels ({y_min}-{y_max}): {np.count_nonzero(loss_mask)}")
    return treecover, lossyear, loss_mask


async def fetch_region_cdse(reg: dict, target_shape=(512, 512)):
    s2_svc = Sentinel2Service()
    s1_svc = Sentinel1Service()
    H, W = target_shape
    bbox = reg["bbox"]
    geom = {
        "type": "Polygon",
        "coordinates": [[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]]
    }

    t0_start, t0_end = reg["t0_s2"]
    t1_start, t1_end = reg["t1_s2"]

    print(f"  [CDSE S2] Fetching T0 ({t0_start} to {t0_end})...")
    s2_t0, bands_t0 = await s2_svc._fetch_multispectral_bands(
        geometry=geom, start_date=t0_start, end_date=t0_end,
        max_cloud_coverage=25.0, band_names=["B02", "B03", "B04", "B08", "B11"]
    )
    print(f"  [CDSE S2] Fetching T1 ({t1_start} to {t1_end})...")
    s2_t1, bands_t1 = await s2_svc._fetch_multispectral_bands(
        geometry=geom, start_date=t1_start, end_date=t1_end,
        max_cloud_coverage=25.0, band_names=["B02", "B03", "B04", "B08", "B11"]
    )

    def process_s2(bands):
        b02 = cv2.resize(bands["B02"].astype(np.float32), target_shape)
        b03 = cv2.resize(bands["B03"].astype(np.float32), target_shape)
        b04 = cv2.resize(bands["B04"].astype(np.float32), target_shape)
        b08 = cv2.resize(bands["B08"].astype(np.float32), target_shape)
        b11 = cv2.resize(bands["B11"].astype(np.float32), target_shape)
        nbr = (b08 - b11) / (b08 + b11 + 1e-6)
        ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
        scl = cv2.resize(bands.get("SCL", np.full(target_shape, 4)).astype(np.uint8), target_shape, interpolation=cv2.INTER_NEAREST)
        return np.stack([b02, b03, b04, b08, b11, nbr, ndvi], axis=0), scl

    s2_t0_arr, scl_t0 = process_s2(bands_t0)
    s2_t1_arr, scl_t1 = process_s2(bands_t1)

    print(f"  [CDSE S1] Fetching SAR T0 & T1...")
    vv0_lin, vh0_lin, _, meta_s1_t0 = await s1_svc.get_sar_bands(
        geometry=geom, start_date=t0_start, end_date=t0_end, width=W, height=H
    )
    vv1_lin, vh1_lin, _, meta_s1_t1 = await s1_svc.get_sar_bands(
        geometry=geom, start_date=t1_start, end_date=t1_end, width=W, height=H
    )

    def process_s1(vv_lin, vh_lin):
        vv_db = 10.0 * np.log10(np.clip(vv_lin, 1e-5, None))
        vh_db = 10.0 * np.log10(np.clip(vh_lin, 1e-5, None))
        ratio_db = vv_db - vh_db
        return np.stack([vv_db.astype(np.float32), vh_db.astype(np.float32), ratio_db.astype(np.float32)], axis=0)

    s1_t0_arr = process_s1(vv0_lin, vh0_lin)
    s1_t1_arr = process_s1(vv1_lin, vh1_lin)

    t0_tensor = np.concatenate([s2_t0_arr, s1_t0_arr], axis=0)
    t1_tensor = np.concatenate([s2_t1_arr, s1_t1_arr], axis=0)

    return {
        "t0": t0_tensor,
        "t1": t1_tensor,
        "scl_t0": scl_t0,
        "scl_t1": scl_t1,
        "s2_t0_id": s2_t0.get("scene_id", "S2_T0"),
        "s2_t1_id": s2_t1.get("scene_id", "S2_T1"),
        "s1_t0_id": meta_s1_t0.get("scene_id", "S1_T0"),
        "s1_t1_id": meta_s1_t1.get("scene_id", "S1_T1"),
    }


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing_data = {}
    if os.path.exists(CACHE_FILE):
        print(f"Loading existing cache from {CACHE_FILE}...")
        with np.load(CACHE_FILE) as npz:
            for k in npz.files:
                existing_data[k] = npz[k]

    meta_dict = {
        "dataset_name": "Copernicus CDSE Sentinel + Hansen GFC v1.11 Multi-Region Real Forest Benchmark",
        "data_source": "REAL",
        "is_simulated": False,
        "reference_label_source": "Hansen Global Forest Change v1.11 (treecover2000, lossyear)",
        "ground_truth_policy": "Strictly independent Landsat 30m Hansen GFC labels; NEVER derived from NBR/NDVI",
        "total_regions": len(REGIONS),
        "regions": {},
    }

    for idx, reg in enumerate(REGIONS):
        reg_id = reg["id"]
        print(f"\n[{idx+1}/{len(REGIONS)}] Processing Region: {reg['name']} ({reg_id})...")

        # Check if already present in cache
        tc_key = f"{reg_id}_treecover"
        t0_key = f"{reg_id}_t0"
        if tc_key in existing_data and t0_key in existing_data:
            print(f"  Region {reg_id} already in cache! Reusing cached arrays.")
            tc = existing_data[tc_key]
            ly = existing_data[f"{reg_id}_lossyear"]
            mask = existing_data[f"{reg_id}_loss_mask"]
            meta_dict["regions"][reg_id] = {
                "name": reg["name"],
                "bbox": reg["bbox"],
                "years": reg["years"],
                "treecover_mean_pct": round(float(np.mean(tc)), 2),
                "hansen_loss_pixels": int(np.count_nonzero(mask)),
            }
            continue

        tc, ly, mask = fetch_hansen_window(reg["bbox"], reg["hansen_tile"], reg["loss_years"])
        cdse = await fetch_region_cdse(reg)

        existing_data[f"{reg_id}_t0"] = cdse["t0"]
        existing_data[f"{reg_id}_t1"] = cdse["t1"]
        existing_data[f"{reg_id}_scl_t0"] = cdse["scl_t0"]
        existing_data[f"{reg_id}_scl_t1"] = cdse["scl_t1"]
        existing_data[f"{reg_id}_treecover"] = tc
        existing_data[f"{reg_id}_lossyear"] = ly
        existing_data[f"{reg_id}_loss_mask"] = mask

        meta_dict["regions"][reg_id] = {
            "name": reg["name"],
            "bbox": reg["bbox"],
            "years": reg["years"],
            "treecover_mean_pct": round(float(np.mean(tc)), 2),
            "hansen_loss_pixels": int(np.count_nonzero(mask)),
            "s2_t0_scene": cdse["s2_t0_id"],
            "s2_t1_scene": cdse["s2_t1_id"],
            "s1_t0_scene": cdse["s1_t0_id"],
            "s1_t1_scene": cdse["s1_t1_id"],
        }

    # Save complete 4-region benchmark
    print(f"\nSaving 4-region dataset to {CACHE_FILE}...")
    np.savez_compressed(CACHE_FILE, **existing_data)

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    print(f"Saved metadata to {META_FILE}.")
    print("=" * 80)
    print(f"BENCHMARK COMPLETE: 4 Real Regions successfully assembled in {CACHE_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

