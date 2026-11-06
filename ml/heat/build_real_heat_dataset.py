# pyright: reportMissingImports=false
"""
ml/heat/build_real_heat_dataset.py:
Fetches 100% REAL satellite observations for Visakhapatnam Urban Heat Island (UHI) analysis:
1. Landsat 9 Collection 2 Level-2 Surface Temperature (ST_B10) via Planetary Computer STAC
   Winter acquisition: 2023-12-24 (LC09_L2SP_141048_20231224_02_T1, cloud=1.6%)
   USGS Scale Equation: LST_Celsius = (DN * 0.00341802 + 149.0) - 273.15
2. Landsat 8 Collection 2 Level-2 Surface Temperature (ST_B10)
   Summer acquisition: 2023-05-22 (LC08_L2SP_141048_20230522_02_T1, cloud=10.0%)
3. Sentinel-2 Level-2A 10m optical predictors matched within 3 days of Landsat winter scene:
   Acquisition: 2023-12-21 (S2B_MSIL2A_20231221T050219_R119_T44QQE, cloud=0.25%)
   Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR1)
   Predictors: NDVI, NDBI, MNDWI, Broadband Albedo
4. MODIS Daily 1km Day vs Night LST (MYD11A1 / MOD11A1) for empirical diurnal cooling range.
Saves aligned arrays into ml/heat/data/real_visakhapatnam_heat_benchmark.npz
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, Tuple, Optional
import numpy as np  # type: ignore[import-not-found,import-untyped]
import rasterio  # type: ignore[import-not-found,import-untyped]
from rasterio.windows import from_bounds  # type: ignore[import-not-found,import-untyped]
from rasterio.warp import transform_bounds  # type: ignore[import-not-found,import-untyped]
import cv2  # type: ignore[import-not-found,import-untyped]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(OUTPUT_DIR, "real_visakhapatnam_heat_benchmark.npz")
META_FILE = os.path.join(OUTPUT_DIR, "real_visakhapatnam_heat_meta.json")

# Visakhapatnam AOI: [min_lon, min_lat, max_lon, max_lat]
BBOX = [83.15, 17.62, 83.35, 17.80]

# Target shapes: 30m grid is 256x256; 10m grid is 768x768 (exact 3x aggregation)
SHAPE_30M = (256, 256)
SHAPE_10M = (768, 768)


def sign_asset_href(raw_href: str) -> str:
    """Signs any Planetary Computer asset URL using the official signing endpoint."""
    sign_url = f"https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={urllib.parse.quote(raw_href, safe='')}"
    req = urllib.request.Request(sign_url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))["href"]


def read_window_cog(signed_href: str, bbox: list, target_shape: tuple) -> np.ndarray:
    """Reads spatial window from Cloud-Optimized GeoTIFF and resizes to target shape."""
    with rasterio.open(signed_href) as src:
        proj_bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(*proj_bounds, src.transform)
        arr = src.read(1, window=win)
    return cv2.resize(arr.astype(np.float32), target_shape, interpolation=cv2.INTER_LINEAR)


def fetch_landsat_lst(scene_id: str, datetime_range: str, season_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fetches Landsat Collection 2 Level-2 Surface Temperature ST_B10 (lwir11)."""
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    query = {
        "collections": ["landsat-c2-l2"],
        "bbox": BBOX,
        "datetime": datetime_range,
        "query": {"eo:cloud_cover": {"lt": 15}},
        "limit": 5,
    }
    req = urllib.request.Request(stac_url, data=json.dumps(query).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        features = json.loads(resp.read().decode("utf-8")).get("features", [])

    matched = None
    for f in features:
        if scene_id in f["id"] or matched is None:
            matched = f
            if scene_id in f["id"]:
                break

    if not matched:
        raise RuntimeError(f"No Landsat scene found for {datetime_range}")

    print(f"[{season_name}] Using Landsat scene: {matched['id']} (Date: {matched['properties']['datetime']}, Cloud: {matched['properties']['eo:cloud_cover']:.1f}%)")
    asset_key = "lwir11" if "lwir11" in matched["assets"] else "ST_B10"
    signed_href = sign_asset_href(matched["assets"][asset_key]["href"])

    raw_dn = read_window_cog(signed_href, BBOX, SHAPE_30M)

    # Official USGS Collection 2 Level-2 Surface Temperature calibration formula
    # LST_Kelvin = DN * 0.00341802 + 149.0
    # LST_Celsius = LST_Kelvin - 273.15
    lst_celsius = raw_dn * 0.00341802 + 149.0 - 273.15

    meta = {
        "scene_id": matched["id"],
        "datetime": matched["properties"]["datetime"],
        "cloud_cover": matched["properties"]["eo:cloud_cover"],
        "data_source": "USGS Landsat Collection 2 Level-2 (ST_B10)",
        "mean_lst_celsius": round(float(np.nanmean(lst_celsius)), 2),
        "min_lst_celsius": round(float(np.nanmin(lst_celsius)), 2),
        "max_lst_celsius": round(float(np.nanmax(lst_celsius)), 2),
    }
    return lst_celsius.astype(np.float32), meta


def fetch_sentinel2_predictors() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Fetches real Sentinel-2 L2A optical bands (B02, B03, B04, B08, B11)
    and computes high-resolution 10m thermal predictors + 30m aggregated predictors.
    """
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    query = {
        "collections": ["sentinel-2-l2a"],
        "bbox": BBOX,
        "datetime": "2023-12-20T00:00:00Z/2023-12-26T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": 5}},
        "limit": 1,
    }
    req = urllib.request.Request(stac_url, data=json.dumps(query).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        features = json.loads(resp.read().decode("utf-8")).get("features", [])

    if not features:
        raise RuntimeError("No Sentinel-2 scene matched within 3 days of Landsat acquisition.")

    feat = features[0]
    print(f"[Sentinel-2] Using scene: {feat['id']} (Date: {feat['properties']['datetime']}, Cloud: {feat['properties']['eo:cloud_cover']:.2f}%)")

    # Fetch 10m bands: B02, B03, B04, B08; 20m band: B11 (upsampled to 10m)
    bands_10m = {}
    for b_name in ["B02", "B03", "B04", "B08", "B11"]:
        href = sign_asset_href(feat["assets"][b_name]["href"])
        raw = read_window_cog(href, BBOX, SHAPE_10M)
        # Convert BOA reflectance scale (Planetary Computer S2 L2A is 0-10000)
        bands_10m[b_name] = np.clip(raw / 10000.0, 0.0, 1.0).astype(np.float32)

    # Compute high-resolution 10m predictors
    b02 = bands_10m["B02"]
    b03 = bands_10m["B03"]
    b04 = bands_10m["B04"]
    b08 = bands_10m["B08"]
    b11 = bands_10m["B11"]

    ndvi_10m = np.clip((b08 - b04) / (b08 + b04 + 1e-6), -1.0, 1.0)
    ndbi_10m = np.clip((b11 - b08) / (b11 + b08 + 1e-6), -1.0, 1.0)
    mndwi_10m = np.clip((b03 - b11) / (b03 + b11 + 1e-6), -1.0, 1.0)
    albedo_10m = np.clip(0.356 * b02 + 0.130 * b04 + 0.373 * b08 + 0.085 * b11 + 0.072 * (b11 * 0.85) - 0.0018, 0.02, 0.80)

    # Spatially aggregate 10m predictors to 30m (exact 3x3 block area mean)
    def agg_3x3(arr_10m):
        H, W = arr_10m.shape
        return arr_10m.reshape(H // 3, 3, W // 3, 3).mean(axis=(1, 3)).astype(np.float32)

    features_10m = {
        "ndvi": ndvi_10m.astype(np.float32),
        "ndbi": ndbi_10m.astype(np.float32),
        "mndwi": mndwi_10m.astype(np.float32),
        "albedo": albedo_10m.astype(np.float32),
        "b02": b02.astype(np.float32),
        "b03": b03.astype(np.float32),
        "b04": b04.astype(np.float32),
    }

    features_30m = {
        "ndvi": agg_3x3(ndvi_10m),
        "ndbi": agg_3x3(ndbi_10m),
        "mndwi": agg_3x3(mndwi_10m),
        "albedo": agg_3x3(albedo_10m),
    }

    meta = {
        "sentinel2_scene_id": feat["id"],
        "acquisition_date": feat["properties"]["datetime"],
        "cloud_cover": feat["properties"]["eo:cloud_cover"],
        "predictors": ["NDVI", "NDBI", "MNDWI", "Albedo"],
        "temporal_alignment": "Sentinel-2 acquisition is within 3 days of Landsat-9 winter scene (2023-12-21 vs 2023-12-24).",
    }
    return features_10m, features_30m, meta


from scipy.ndimage import distance_transform_edt


def resample_modis_to_grid(arr_modis: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Fills nodata/ocean pixels with nearest neighbor and resamples to target shape."""
    valid_mask = ~np.isnan(arr_modis)
    if not np.any(valid_mask):
        return np.full(target_shape, np.nan, dtype=np.float32)
    indices = distance_transform_edt(np.isnan(arr_modis), return_distances=False, return_indices=True)
    filled = arr_modis[tuple(indices)]
    return cv2.resize(filled.astype(np.float32), target_shape, interpolation=cv2.INTER_LINEAR)


def fetch_modis_diurnal_lst() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fetches real MODIS Terra/Aqua Day vs Night Surface Temperature
    to compute empirical diurnal temperature range (DTR).
    """
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    query = {
        "collections": ["modis-11A1-061"],
        "bbox": BBOX,
        "datetime": "2023-12-25T00:00:00Z/2023-12-31T23:59:59Z",
        "limit": 10,
    }
    req = urllib.request.Request(stac_url, data=json.dumps(query).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        features = json.loads(resp.read().decode("utf-8")).get("features", [])

    if not features:
        raise RuntimeError("No MODIS scene found for diurnal comparison.")

    # Find a feature with valid day and night overpass
    chosen = None
    for f in features:
        if "MOD11A1.A2023364" in f["id"]:
            chosen = f
            break
    if chosen is None:
        chosen = features[0]

    print(f"[MODIS Diurnal] Using scene: {chosen['id']}")

    href_day = sign_asset_href(chosen["assets"]["LST_Day_1km"]["href"])
    href_night = sign_asset_href(chosen["assets"]["LST_Night_1km"]["href"])

    with rasterio.open(href_day) as sd, rasterio.open(href_night) as sn:
        pb = transform_bounds("EPSG:4326", sd.crs, *BBOX)
        win = from_bounds(*pb, sd.transform)
        raw_day = sd.read(1, window=win).astype(np.float32)
        raw_night = sn.read(1, window=win).astype(np.float32)

    # MODIS LST scale: DN * 0.02 - 273.15
    raw_day_c = np.where(raw_day > 0, raw_day * 0.02 - 273.15, np.nan)
    raw_night_c = np.where(raw_night > 0, raw_night * 0.02 - 273.15, np.nan)

    mean_day_val = float(np.nanmean(raw_day_c))
    mean_night_val = float(np.nanmean(raw_night_c))

    # Resample to 30m grid (256x256)
    lst_day_30m = resample_modis_to_grid(raw_day_c, SHAPE_30M)
    lst_night_30m = resample_modis_to_grid(raw_night_c, SHAPE_30M)

    meta = {
        "modis_scene_id": chosen["id"],
        "mean_day_lst": round(mean_day_val, 2),
        "mean_night_lst": round(mean_night_val, 2),
        "mean_dtr_celsius": round(mean_day_val - mean_night_val, 2),
        "data_source": "MODIS Terra MOD11A1.061 Daily 1km Day/Night LST",
    }
    return lst_day_30m.astype(np.float32), lst_night_30m.astype(np.float32), meta


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 80)
    print("BUILDING REAL URBAN HEAT DATASET (LANDSAT 8/9 C2 L2 + SENTINEL-2 L2A + MODIS)")
    print("AOI: Visakhapatnam, Andhra Pradesh, India", BBOX)
    print("=" * 80)

    # Check if Landsat and Sentinel-2 are already cached
    if os.path.exists(CACHE_FILE):
        print(f"Loading existing Landsat and Sentinel-2 cache from {CACHE_FILE}...")
        existing = np.load(CACHE_FILE)
        lst_winter_30m = existing["lst_winter_30m"]
        lst_summer_30m = existing["lst_summer_30m"]
        features_10m = {
            "ndvi": existing["features_10m_ndvi"],
            "ndbi": existing["features_10m_ndbi"],
            "mndwi": existing["features_10m_mndwi"],
            "albedo": existing["features_10m_albedo"],
            "b02": existing["features_10m_b02"],
            "b03": existing["features_10m_b03"],
            "b04": existing["features_10m_b04"],
        }
        features_30m = {
            "ndvi": existing["features_30m_ndvi"],
            "ndbi": existing["features_30m_ndbi"],
            "mndwi": existing["features_30m_mndwi"],
            "albedo": existing["features_30m_albedo"],
        }
        meta_winter = {}
        meta_summer = {}
        meta_s2 = {}
        if os.path.exists(META_FILE):
            with open(META_FILE, "r") as mf:
                old_m = json.load(mf)
                meta_winter = old_m.get("landsat_winter_scene", {})
                meta_summer = old_m.get("landsat_summer_scene", {})
                meta_s2 = old_m.get("sentinel2_optical_scene", {})
    else:
        # 1. Landsat 9 Winter Baseline (Dec 24, 2023)
        lst_winter_30m, meta_winter = fetch_landsat_lst("LC09_L2SP_141048_20231224_02_T1", "2023-12-20T00:00:00Z/2023-12-28T23:59:59Z", "Winter Landsat-9")

        # 2. Landsat 8 Summer Pre-Monsoon Peak (May 22, 2023)
        lst_summer_30m, meta_summer = fetch_landsat_lst("LC08_L2SP_141048_20230522_02_T1", "2023-05-15T00:00:00Z/2023-05-30T23:59:59Z", "Summer Landsat-8")

        # 3. Sentinel-2 High-Resolution Predictors matched to winter Landsat
        features_10m, features_30m, meta_s2 = fetch_sentinel2_predictors()

    # 4. MODIS Diurnal Day vs Night
    lst_modis_day, lst_modis_night, meta_modis = fetch_modis_diurnal_lst()

    # Save to disk
    np.savez_compressed(
        CACHE_FILE,
        lst_winter_30m=lst_winter_30m,
        lst_summer_30m=lst_summer_30m,
        features_10m_ndvi=features_10m["ndvi"],
        features_10m_ndbi=features_10m["ndbi"],
        features_10m_mndwi=features_10m["mndwi"],
        features_10m_albedo=features_10m["albedo"],
        features_10m_b02=features_10m["b02"],
        features_10m_b03=features_10m["b03"],
        features_10m_b04=features_10m["b04"],
        features_30m_ndvi=features_30m["ndvi"],
        features_30m_ndbi=features_30m["ndbi"],
        features_30m_mndwi=features_30m["mndwi"],
        features_30m_albedo=features_30m["albedo"],
        lst_modis_day=lst_modis_day,
        lst_modis_night=lst_modis_night,
    )

    metadata = {
        "dataset_name": "Visakhapatnam Real Satellite Urban Heat Island Benchmark",
        "data_source": "REAL",
        "is_simulated": False,
        "geographic_extent": {
            "city": "Visakhapatnam",
            "state_country": "Andhra Pradesh, India",
            "bbox": BBOX,
            "crs": "EPSG:4326 (WGS 84)",
            "grid_shape_30m": list(SHAPE_30M),
            "grid_shape_10m": list(SHAPE_10M),
        },
        "landsat_winter_scene": meta_winter,
        "landsat_summer_scene": meta_summer,
        "sentinel2_optical_scene": meta_s2,
        "modis_diurnal_scene": meta_modis,
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print(f"REAL URBAN HEAT DATASET COMPLETE: Saved to {CACHE_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
