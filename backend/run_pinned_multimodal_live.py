import asyncio
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Set DEMO_MODE to false to query live Copernicus APIs
os.environ["DEMO_MODE"] = "false"

from services.sentinel2_service import Sentinel2Service
from services.sentinel1_service import Sentinel1Service
from services.ndvi_service import (
    calculate_ndvi,
    create_valid_pixel_mask,
)
from services.ndmi_service import calculate_ndmi
from services.ndre_service import calculate_ndre
from services.evi_service import calculate_evi

FIELDS = [
    {
        "name": "Tadepalligudem",
        "bbox": [81.503, 16.793, 81.543, 16.833],
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[81.503, 16.793], [81.543, 16.793], [81.543, 16.833], [81.503, 16.833], [81.503, 16.793]]]
        }
    },
    {
        "name": "Bhimavaram",
        "bbox": [81.501, 16.525, 81.541, 16.565],
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[81.501, 16.525], [81.541, 16.525], [81.541, 16.565], [81.501, 16.565], [81.501, 16.525]]]
        }
    },
    {
        "name": "Maruteru",
        "bbox": [81.725, 16.614, 81.765, 16.654],
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[81.725, 16.614], [81.765, 16.614], [81.765, 16.654], [81.725, 16.654], [81.725, 16.614]]]
        }
    }
]

async def process_live_runs():
    s2_service = Sentinel2Service()
    s1_service = Sentinel1Service()
    
    # We test clear dates in Jan-Feb 2024:
    # 2024-01-05 (clear S2 pass)
    # 2024-01-10 (clear S2 pass with S1 on 2024-01-12, 2 days gap)
    # 2024-02-19 (clear S2 pass 9% cloud with S1 on 2024-02-17, 2 days gap)
    # 2024-02-29 (clear S2 pass 2.6% cloud with S1 on 2024-02-29, 0 days gap)
    target_dates = [
        ("2024-01-05", "2024-01-05"),
        ("2024-01-10", "2024-01-10"),
        ("2024-02-19", "2024-02-19"),
        ("2024-02-29", "2024-02-29"),
    ]
    
    results = []
    
    print("=" * 100, flush=True)
    print("STARTING LIVE EXECUTION: 3 FIELDS X PINNED DATES (DEMO_MODE=FALSE)", flush=True)
    print("=" * 100, flush=True)
    
    for f in FIELDS:
        fname = f["name"]
        geom = f["geometry"]
        print(f"\n>>> PROCESSING FIELD: {fname} <<<", flush=True)
        
        # Get S1 catalog for Jan-Feb 2024 to pair with
        s1_catalog = await s1_service.search_imagery(geom, "2024-01-01", "2024-02-29")
        print(f"Sentinel-1 catalog entries found for {fname}: {len(s1_catalog)}", flush=True)
        
        for s_date, e_date in target_dates:
            print(f"\n--- Field: {fname} | Target Date: {s_date} ---", flush=True)
            try:
                # 1. Fetch S2 with pinned timeRange
                s2_scene, bands = await s2_service._fetch_multispectral_bands(
                    geom, s_date, e_date, max_cloud_coverage=20.0,
                    band_names=["B02", "B03", "B04", "B08", "B11", "B05"]
                )
                s2_id = s2_scene["scene_id"]
                s2_acq = s2_scene["acquisition_date"]
                print(f"S2 Retrieved: {s2_id} (Acq: {s2_acq})", flush=True)
                
                b04 = bands["B04"]
                b08 = bands["B08"]
                b02 = bands["B02"]
                b11 = bands.get("B11")
                b05 = bands.get("B05")
                scl = bands.get("SCL")
                dmask = bands.get("dataMask")
                
                valid_mask = create_valid_pixel_mask(b04, b08, scl=scl, data_mask=dmask)
                valid_pct = round(float(np.mean(valid_mask) * 100.0), 2)
                
                # Compute optical indices
                ndvi_arr = calculate_ndvi(b04, b08, valid_mask=valid_mask)
                ndmi_arr = calculate_ndmi(b08, b11, valid_mask=valid_mask) if b11 is not None else None
                ndre_arr = calculate_ndre(b08, b05, valid_mask=valid_mask) if b05 is not None else None
                evi_arr = calculate_evi(b04, b08, b02, valid_mask=valid_mask)
                
                ndvi_mean = float(np.nanmean(ndvi_arr)) if not np.all(np.isnan(ndvi_arr)) else None
                ndmi_mean = float(np.nanmean(ndmi_arr)) if ndmi_arr is not None and not np.all(np.isnan(ndmi_arr)) else None
                ndre_mean = float(np.nanmean(ndre_arr)) if ndre_arr is not None and not np.all(np.isnan(ndre_arr)) else None
                evi_mean = float(np.nanmean(evi_arr)) if not np.all(np.isnan(evi_arr)) else None
                
                # 2. Match nearest S1 scene within ±3 days
                match = s1_service.find_closest_sentinel1_observation(s_date, s1_catalog, max_window_days=3)
                
                s1_id = "None"
                gap_days = "None"
                vv_db_mean = None
                vh_db_mean = None
                ratio_db_mean = None
                
                if match.get("matched"):
                    s1_scene = match["scene"]
                    s1_id = s1_scene["scene_id"]
                    gap_days = match["date_difference_days"]
                    print(f"Matched S1 Scene: {s1_id} (Gap: {gap_days} days)", flush=True)
                    
                    # Fetch SAR bands with exact pinned timeRange
                    vv_lin, vh_lin, sar_mask, sar_meta = await s1_service.get_sar_bands(
                        geom, s_date, e_date, target_scene=s1_scene
                    )
                    
                    # Compute radar backscatter in linear power then convert to dB
                    # Mask out <= -50 dB numerical floor
                    vv_db = 10.0 * np.log10(np.maximum(vv_lin, 1e-6))
                    vh_db = 10.0 * np.log10(np.maximum(vh_lin, 1e-6))
                    valid_sar = sar_mask & (vh_db > -50.0)
                    
                    if np.any(valid_sar):
                        # Linear power averages
                        vv_pow_avg = float(np.mean(vv_lin[valid_sar]))
                        vh_pow_avg = float(np.mean(vh_lin[valid_sar]))
                        vv_db_mean = round(10.0 * np.log10(vv_pow_avg), 3)
                        vh_db_mean = round(10.0 * np.log10(vh_pow_avg), 3)
                        # SAR difference in dB = 10 * log10(vv_lin / vh_lin)
                        ratio_db_mean = round(vv_db_mean - vh_db_mean, 3)
                else:
                    print(f"No S1 pass within ±3 days of {s_date}", flush=True)

                row = {
                    "field": fname,
                    "date": s_date,
                    "s2_id": s2_id,
                    "s1_id": s1_id,
                    "gap_days": gap_days,
                    "valid_pct": valid_pct,
                    "ndvi": round(ndvi_mean, 4) if ndvi_mean else None,
                    "ndmi": round(ndmi_mean, 4) if ndmi_mean else None,
                    "ndre": round(ndre_mean, 4) if ndre_mean else None,
                    "evi": round(evi_mean, 4) if evi_mean else None,
                    "vv_db": vv_db_mean,
                    "vh_db": vh_db_mean,
                    "ratio_db": ratio_db_mean,
                }
                results.append(row)
                print(f"Row result: {row}", flush=True)

            except Exception as exc:
                print(f"Error processing {fname} on {s_date}: {exc}", flush=True)
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 110, flush=True)
    print("FINAL RESULTS TABLE (3 FIELDS X PINNED DATES)", flush=True)
    print("=" * 110, flush=True)
    header = f"{'Field':<15} | {'Date':<10} | {'Gap':<5} | {'Valid%':<7} | {'NDVI':<6} | {'NDMI':<6} | {'NDRE':<6} | {'EVI':<6} | {'VV(dB)':<7} | {'VH(dB)':<7} | {'VV-VH(dB)':<9} | {'S2 ID':<35} | {'S1 ID':<35}"
    print(header, flush=True)
    print("-" * 110, flush=True)
    for r in results:
        line = f"{r['field']:<15} | {r['date']:<10} | {str(r['gap_days']):<5} | {r['valid_pct']:<7.2f} | {str(r['ndvi']):<6} | {str(r['ndmi']):<6} | {str(r['ndre']):<6} | {str(r['evi']):<6} | {str(r['vv_db']):<7} | {str(r['vh_db']):<7} | {str(r['ratio_db']):<9} | {r['s2_id'][:35]:<35} | {r['s1_id'][:35]:<35}"
        print(line, flush=True)

if __name__ == '__main__':
    asyncio.run(process_live_runs())
