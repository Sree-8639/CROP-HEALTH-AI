"""
ml/heat/build_expanded_heat_dataset.py:
Builds expanded Visakhapatnam Urban Heat Island (UHI) dataset with:
1. 6 real Landsat 8/9 Collection 2 Level-2 Surface Temperature (ST_B10) scenes across seasons:
   - Winter 1: 2023-01-06 (LC09)
   - Late Winter: 2023-02-15 (LC08)
   - Pre-Monsoon Spring: 2023-04-12 (LC09)
   - Peak Summer: 2023-05-22 (LC08)
   - Post-Monsoon: 2023-10-05 (LC09)
   - Mid-Winter Peak: 2023-12-24 (LC09)
2. Additional high-resolution physical predictors:
   - DEM (Copernicus GLO-30 30m Digital Elevation Model) & Topographic Slope (degrees)
   - Distance to Coast (km from Bay of Bengal maritime boundary via Euclidean Distance Transform)
   - Sentinel-1 C-SAR Local Texture (3x3 local standard deviation of VV and VH backscatter)
3. Aligned optical predictors: NDVI, NDBI, MNDWI, Broadband Albedo.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
import asyncio
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import cv2
from scipy.ndimage import distance_transform_edt

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

from services.sentinel1_service import Sentinel1Service

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(OUTPUT_DIR, "real_visakhapatnam_heat_benchmark.npz")
META_FILE = os.path.join(OUTPUT_DIR, "real_visakhapatnam_heat_meta.json")

BBOX = [83.15, 17.62, 83.35, 17.80]
SHAPE_30M = (256, 256)
SHAPE_10M = (768, 768)

LANDSAT_SCENES = [
    {"id": "LC09_L2SP_141048_20230106_02_T1", "date": "2023-01-06", "season": "Winter (Early)"},
    {"id": "LC08_L2SP_141048_20230215_02_T1", "date": "2023-02-15", "season": "Late Winter"},
    {"id": "LC09_L2SP_141048_20230412_02_T1", "date": "2023-04-12", "season": "Pre-Monsoon Spring"},
    {"id": "LC08_L2SP_141048_20230522_02_T1", "date": "2023-05-22", "season": "Peak Summer"},
    {"id": "LC09_L2SP_141048_20231005_02_T1", "date": "2023-10-05", "season": "Post-Monsoon"},
    {"id": "LC09_L2SP_141048_20231224_02_T1", "date": "2023-12-24", "season": "Mid-Winter Peak"},
]


def sign_asset_href(raw_href: str) -> str:
    sign_url = f"https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={urllib.parse.quote(raw_href, safe='')}"
    req = urllib.request.Request(sign_url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))["href"]


def read_window_cog(signed_href: str, bbox: list, target_shape: tuple) -> np.ndarray:
    with rasterio.open(signed_href) as src:
        proj_bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(*proj_bounds, src.transform)
        arr = src.read(1, window=win)
    return cv2.resize(arr.astype(np.float32), target_shape, interpolation=cv2.INTER_LINEAR)


def fetch_all_landsat_dates():
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    scene_ids = [s["id"] for s in LANDSAT_SCENES]
    query = {"collections": ["landsat-c2-l2"], "ids": scene_ids}
    req = urllib.request.Request(stac_url, data=json.dumps(query).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        features = json.loads(resp.read().decode("utf-8")).get("features", [])

    feat_by_id = {f["id"]: f for f in features}
    lst_scenes = {}

    for s_info in LANDSAT_SCENES:
        s_id = s_info["id"]
        if s_id not in feat_by_id:
            print(f"Warning: {s_id} not found in STAC search")
            continue
        feat = feat_by_id[s_id]
        asset_key = "lwir11" if "lwir11" in feat["assets"] else "ST_B10"
        print(f"Fetching Landsat LST for {s_info['date']} ({s_info['season']})...")
        signed_href = sign_asset_href(feat["assets"][asset_key]["href"])
        raw_dn = read_window_cog(signed_href, BBOX, SHAPE_30M)
        # USGS calibration: LST_C = DN * 0.00341802 + 149.0 - 273.15
        lst_celsius = raw_dn * 0.00341802 + 149.0 - 273.15
        lst_scenes[s_id] = {
            "scene_id": s_id,
            "date": s_info["date"],
            "season": s_info["season"],
            "lst": lst_celsius,
            "cloud_cover": feat["properties"]["eo:cloud_cover"],
        }
    return lst_scenes


def fetch_dem_and_slope():
    href = "https://elevationeuwest.blob.core.windows.net/copernicus-dem/COP30_hh/Copernicus_DSM_COG_10_N17_00_E083_00_DEM.tif"
    signed_href = sign_asset_href(href)
    print("Reading Copernicus GLO-30 DEM...")
    dem_raw = read_window_cog(signed_href, BBOX, SHAPE_30M)

    # Calculate slope in degrees using 30m grid spacing
    # cv2.Sobel or np.gradient
    dy, dx = np.gradient(dem_raw, 30.0, 30.0)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.rad2deg(slope_rad).astype(np.float32)

    dem_10m = cv2.resize(dem_raw, SHAPE_10M, interpolation=cv2.INTER_CUBIC)
    slope_10m = cv2.resize(slope_deg, SHAPE_10M, interpolation=cv2.INTER_LINEAR)

    return dem_raw, slope_deg, dem_10m, slope_10m


def calculate_dist_to_coast(mndwi_30m: np.ndarray, mndwi_10m: np.ndarray):
    print("Computing Distance to Coast (Bay of Bengal maritime boundary)...")
    land_mask_30m = (mndwi_30m < 0.15)
    # EDT from water boundary: distance in pixels * 0.030 km
    dist_30m_km = (distance_transform_edt(land_mask_30m) * 0.030).astype(np.float32)

    land_mask_10m = (mndwi_10m < 0.15)
    dist_10m_km = (distance_transform_edt(land_mask_10m) * 0.010).astype(np.float32)

    return dist_30m_km, dist_10m_km


async def fetch_sar_texture():
    print("Fetching Sentinel-1 SAR and computing spatial backscatter texture...")
    s1_svc = Sentinel1Service()
    geom = {
        "type": "Polygon",
        "coordinates": [[[BBOX[0], BBOX[1]], [BBOX[2], BBOX[1]], [BBOX[2], BBOX[3]], [BBOX[0], BBOX[3]], [BBOX[0], BBOX[1]]]]
    }
    vv_lin, vh_lin, _, meta = await s1_svc.get_sar_bands(geom, "2023-12-01", "2023-12-31", width=256, height=256)
    vv_db = 10.0 * np.log10(np.clip(vv_lin, 1e-5, None)).astype(np.float32)
    vh_db = 10.0 * np.log10(np.clip(vh_lin, 1e-5, None)).astype(np.float32)

    # Local standard deviation in 3x3 window: sqrt(E[X^2] - (E[X])^2)
    def local_std(img, ksize=3):
        mean = cv2.blur(img, (ksize, ksize))
        sq_mean = cv2.blur(img**2, (ksize, ksize))
        var = np.maximum(sq_mean - mean**2, 0.0)
        return np.sqrt(var).astype(np.float32)

    tex_vv_30m = local_std(vv_db, 3)
    tex_vh_30m = local_std(vh_db, 3)

    tex_vv_10m = cv2.resize(tex_vv_30m, SHAPE_10M, interpolation=cv2.INTER_LINEAR)
    tex_vh_10m = cv2.resize(tex_vh_30m, SHAPE_10M, interpolation=cv2.INTER_LINEAR)

    return {
        "vv_db_30m": vv_db,
        "vh_db_30m": vh_db,
        "texture_vv_30m": tex_vv_30m,
        "texture_vh_30m": tex_vh_30m,
        "texture_vv_10m": tex_vv_10m,
        "texture_vh_10m": tex_vh_10m,
    }


async def main():
    print("=" * 80)
    print("BUILDING EXPANDED REAL VISAKHAPATNAM HEAT DATASET")
    print("=" * 80)

    # Load existing optical predictors and seasonal LST from cache
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"{CACHE_FILE} not found. Run previous build script first.")

    cached = np.load(CACHE_FILE)
    cached_data = {k: cached[k] for k in cached.files}

    mndwi_30m = cached_data["features_30m_mndwi"]
    mndwi_10m = cached_data["features_10m_mndwi"]

    # 1. Fetch 6 Landsat scenes across seasons
    lst_scenes = fetch_all_landsat_dates()

    # 2. Fetch DEM and compute slope
    dem_30m, slope_30m, dem_10m, slope_10m = fetch_dem_and_slope()

    # 3. Distance to coast
    dist_coast_30m, dist_coast_10m = calculate_dist_to_coast(mndwi_30m, mndwi_10m)

    # 4. Sentinel-1 SAR texture
    sar_dict = await fetch_sar_texture()

    # Update cache
    cached_data["features_30m_dem"] = dem_30m
    cached_data["features_30m_slope"] = slope_30m
    cached_data["features_30m_dist_coast"] = dist_coast_30m
    cached_data["features_30m_sar_tex_vv"] = sar_dict["texture_vv_30m"]
    cached_data["features_30m_sar_tex_vh"] = sar_dict["texture_vh_30m"]

    cached_data["features_10m_dem"] = dem_10m
    cached_data["features_10m_slope"] = slope_10m
    cached_data["features_10m_dist_coast"] = dist_coast_10m
    cached_data["features_10m_sar_tex_vv"] = sar_dict["texture_vv_10m"]
    cached_data["features_10m_sar_tex_vh"] = sar_dict["texture_vh_10m"]

    # Save multi-date LST arrays
    date_keys = []
    meta_dates = []
    for s_info in LANDSAT_SCENES:
        s_id = s_info["id"]
        if s_id in lst_scenes:
            clean_date = s_info["date"].replace("-", "")
            key = f"lst_landsat_{clean_date}"
            cached_data[key] = lst_scenes[s_id]["lst"]
            date_keys.append(key)
            meta_dates.append({
                "scene_id": s_id,
                "date": s_info["date"],
                "season": s_info["season"],
                "npz_key": key,
                "mean_lst_celsius": round(float(np.nanmean(lst_scenes[s_id]["lst"])), 2),
                "cloud_cover": lst_scenes[s_id]["cloud_cover"],
            })

    print(f"\nSaving expanded heat benchmark to {CACHE_FILE}...")
    np.savez_compressed(CACHE_FILE, **cached_data)

    # Update metadata
    meta = {
        "dataset_name": "Planetary Computer Landsat + Copernicus DEM + Sentinel-1/2 Heat Benchmark",
        "aoi": "Visakhapatnam, Andhra Pradesh, India",
        "bbox": BBOX,
        "grid_shape_30m": list(SHAPE_30M),
        "grid_shape_10m": list(SHAPE_10M),
        "data_source": "REAL",
        "is_simulated": False,
        "landsat_dates_count": len(meta_dates),
        "landsat_dates": meta_dates,
        "predictors_list": [
            "NDVI (Canopy Vigor)",
            "NDBI (Built-up Density)",
            "MNDWI (Water / Moisture)",
            "Broadband Albedo (Liang 2001)",
            "Copernicus GLO-30 DEM Elevation (m)",
            "Topographic Slope (deg)",
            "Distance to Coast (km)",
            "Sentinel-1 SAR Texture VV (dB std)",
            "Sentinel-1 SAR Texture VH (dB std)",
        ],
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Updated metadata in {META_FILE}.")
    print("=" * 80)
    print("EXPANDED HEAT DATASET READY WITH 6 LANDSAT DATES + DEM/SLOPE/COAST/TEXTURE PREDICTORS!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

