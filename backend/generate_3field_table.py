import os
import sys
import asyncio
import datetime
import numpy as np
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv('.env', override=True)
os.environ['DEMO_MODE'] = 'false'

from services.sentinel2_service import sentinel2_service
from services.sentinel1_service import sentinel1_service
from services.ndvi_service import create_valid_pixel_mask

FIELDS = [
    {
        "field_id": "AP_WG_01",
        "name": "Tadepalligudem Paddy",
        "lat": 16.810244,
        "lon": 81.526592,
    },
    {
        "field_id": "AP_WG_02",
        "name": "Bhimavaram Delta",
        "lat": 16.544893,
        "lon": 81.521241,
    },
    {
        "field_id": "AP_WG_03",
        "name": "Maruteru Research Station",
        "lat": 16.632553,
        "lon": 81.731511,
    },
]

# Evaluation target windows: December 2023, January 2024, February 2024
TARGET_WINDOWS = [
    ("2023-12-25", "2024-01-08"), # Centered around Dec 31 / Jan 05
    ("2024-01-10", "2024-01-26"), # Centered around Jan 12 / Jan 15 / Jan 24
    ("2024-02-12", "2024-02-28"), # Centered around Feb 17 / Feb 20
]

async def process_field_window(field, start_d, end_d):
    lat, lon = field["lat"], field["lon"]
    r_km = 1.0
    delta = r_km / 111.32
    geom = {
        'type': 'Polygon',
        'coordinates': [[[lon-delta, lat-delta], [lon+delta, lat-delta], [lon+delta, lat+delta], [lon-delta, lat+delta], [lon-delta, lat-delta]]]
    }
    
    # 1. Fetch S2
    all_req_bands = ['B02', 'B04', 'B05', 'B08', 'B11']
    try:
        s2_scene, bands = await sentinel2_service._fetch_multispectral_bands(
            geom, start_d, end_d, 35.0, all_req_bands
        )
        s2_id = s2_scene.get("scene_id") or "S2_L2A_Product"
        s2_date_str = s2_scene.get("acquisition_date")[:10]
        s2_dt = datetime.date.fromisoformat(s2_date_str)
        
        b04, b08 = bands["B04"], bands["B08"]
        scl, data_mask = bands.get("SCL"), bands.get("dataMask")
        mask = create_valid_pixel_mask(b04, b08, scl, data_mask)
        valid_pct = (np.count_nonzero(mask) / mask.size) * 100.0
        
        # Calculate optical index means on valid pixels
        if np.count_nonzero(mask) > 0:
            ndvi_mean = float(np.mean(((b08 - b04) / (b08 + b04 + 1e-9))[mask]))
            b11, b05, b02 = bands["B11"], bands["B05"], bands["B02"]
            ndmi_mean = float(np.mean(((b08 - b11) / (b08 + b11 + 1e-9))[mask]))
            ndre_mean = float(np.mean(((b08 - b05) / (b08 + b05 + 1e-9))[mask]))
            evi_mean = float(np.mean((2.5 * (b08 - b04) / (b08 + 6.0*b04 - 7.5*b02 + 1.0))[mask]))
        else:
            ndvi_mean = ndmi_mean = ndre_mean = evi_mean = float("nan")
    except Exception as e:
        s2_id = f"FAILED: {str(e)[:30]}"
        s2_date_str = start_d
        s2_dt = datetime.date.fromisoformat(start_d)
        valid_pct = 0.0
        ndvi_mean = ndmi_mean = ndre_mean = evi_mean = float("nan")

    # 2. Fetch S1
    try:
        s1_res = await sentinel1_service.get_sar_analysis(geom, start_d, end_d, s2_date=s2_date_str)
        s1_id = s1_res.get("product_info", {}).get("id") or s1_res.get("scene", {}).get("id") or "Sentinel-1 GRD L1"
        s1_date_str = s1_res.get("observation_date")[:10]
        s1_dt = datetime.date.fromisoformat(s1_date_str)
        gap_days = abs((s1_dt - s2_dt).days)
        
        sar_feat = s1_res.get("sar_features", {})
        vv_mean = sar_feat.get("vv", {}).get("statistics", {}).get("mean")
        vh_mean = sar_feat.get("vh", {}).get("statistics", {}).get("mean")
        vv_vh_db_diff = round(vv_mean - vh_mean, 2) if (vv_mean is not None and vh_mean is not None) else None
        vv_vh_linear_ratio = round(10.0 ** (vv_vh_db_diff / 10.0), 2) if vv_vh_db_diff is not None else None
    except Exception as e:
        s1_id = f"FAILED: {str(e)[:30]}"
        s1_date_str = "N/A"
        gap_days = 999
        vv_mean = vh_mean = vv_vh_db_diff = vv_vh_linear_ratio = None

    return {
        "field_id": field["field_id"],
        "field_name": field["name"],
        "s2_date": s2_date_str,
        "s1_date": s1_date_str,
        "s2_id": s2_id[:35] + "..." if len(s2_id) > 35 else s2_id,
        "s1_id": s1_id[:30] + "..." if len(s1_id) > 30 else s1_id,
        "gap_days": gap_days,
        "valid_pct": round(valid_pct, 1),
        "ndvi": round(ndvi_mean, 3) if not np.isnan(ndvi_mean) else "NaN",
        "ndmi": round(ndmi_mean, 3) if not np.isnan(ndmi_mean) else "NaN",
        "ndre": round(ndre_mean, 3) if not np.isnan(ndre_mean) else "NaN",
        "evi": round(evi_mean, 3) if not np.isnan(evi_mean) else "NaN",
        "vv_db": round(vv_mean, 2) if vv_mean is not None else "N/A",
        "vh_db": round(vh_mean, 2) if vh_mean is not None else "N/A",
        "sar_db_diff": vv_vh_db_diff,
        "sar_linear_ratio": vv_vh_linear_ratio,
    }

async def main():
    print("Executing multi-field real Copernicus acquisitions...")
    results = []
    for f in FIELDS:
        for (st, end) in TARGET_WINDOWS:
            print(f"Querying {f['name']} ({st} to {end})...")
            res = await process_field_window(f, st, end)
            results.append(res)
            
    print("\n" + "="*140)
    print("REAL COPERNICUS SATELLITE ACQUISITION TABLE (3 FIELDS x MULTIPLE DATES)")
    print("="*140)
    headers = [
        "Field", "S2 Date", "S1 Date", "Gap (d)", "Valid %", 
        "NDVI", "NDMI", "NDRE", "EVI", 
        "VV (dB)", "VH (dB)", "VV-VH (dB)", "VV/VH (lin)", "S2 Scene ID"
    ]
    fmt = "{:<16} {:<10} {:<10} {:<7} {:<8} {:<6} {:<6} {:<6} {:<6} {:<8} {:<8} {:<10} {:<11} {:<30}"
    print(fmt.format(*headers))
    print("-" * 140)
    for r in results:
        print(fmt.format(
            r["field_name"][:16],
            r["s2_date"],
            r["s1_date"],
            f"{r['gap_days']} d",
            f"{r['valid_pct']}%",
            str(r["ndvi"]),
            str(r["ndmi"]),
            str(r["ndre"]),
            str(r["evi"]),
            str(r["vv_db"]),
            str(r["vh_db"]),
            str(r["sar_db_diff"]),
            str(r["sar_linear_ratio"]),
            r["s2_id"]
        ))
    print("="*140)

if __name__ == '__main__':
    asyncio.run(main())

