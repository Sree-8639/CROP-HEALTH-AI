import os
import sys
import asyncio
import numpy as np
from PIL import Image, ImageDraw
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv('.env', override=True)
os.environ['DEMO_MODE'] = 'false'
from services.sentinel1_service import sentinel1_service

async def test_q4():
    lat = 16.810244
    lon = 81.526592
    r_km = 1.0
    delta = r_km / 111.32
    geom = {
        'type': 'Polygon',
        'coordinates': [[[lon-delta, lat-delta], [lon+delta, lat-delta], [lon+delta, lat+delta], [lon-delta, lat+delta], [lon-delta, lat-delta]]]
    }
    
    print("================================================================================")
    print("INVESTIGATION 4: SAR -60 dB NO-DATA MASKING, RADAR AVERAGING & +18 dB OVERLAY")
    print("================================================================================")
    
    # Fetch real SAR bands for Tadepalligudem (Scene: 2024-01-12)
    vv_raw, vh_raw, mask, s1_meta = await sentinel1_service.get_sar_bands(
        geom, '2024-01-10', '2024-01-14', width=256, height=256
    )
    
    # 1. Inspect raw floor pixels
    floor_pixels = (vh_raw <= 1e-5)
    print(f"Total raster pixels: {vh_raw.size} (256x256)")
    print(f"Pixels at or below 1e-5 (-60 dB floor): {np.count_nonzero(floor_pixels)} ({np.count_nonzero(floor_pixels)/vh_raw.size*100:.2f}%)")
    
    # Convert to dB with -60 dB replaced by NaN
    vh_clean = np.where(floor_pixels, np.nan, vh_raw)
    vv_clean = np.where(vv_raw <= 1e-5, np.nan, vv_raw)
    
    vh_db = 10.0 * np.log10(vh_clean)
    vv_db = 10.0 * np.log10(vv_clean)
    
    valid_sar_pixels = np.count_nonzero(~np.isnan(vh_db) & ~np.isnan(vv_db))
    sar_valid_pct = (valid_sar_pixels / vh_raw.size) * 100.0
    print(f"SAR Valid land pixels (excluding -60 dB): {valid_sar_pixels} ({sar_valid_pct:.2f}%)")
    
    # Linear-power averaging vs median vs naive dB mean
    # VH
    vh_naive_mean_db = float(np.nanmean(vh_db))
    vh_median_db = float(np.nanmedian(vh_db))
    vh_lin_mean = float(np.nanmean(vh_clean))
    vh_lin_power_avg_db = float(10.0 * np.log10(vh_lin_mean))
    vh_min_db = float(np.nanmin(vh_db))
    vh_max_db = float(np.nanmax(vh_db))
    vh_std_db = float(np.nanstd(vh_db))

    # VV
    vv_naive_mean_db = float(np.nanmean(vv_db))
    vv_median_db = float(np.nanmedian(vv_db))
    vv_lin_mean = float(np.nanmean(vv_clean))
    vv_lin_power_avg_db = float(10.0 * np.log10(vv_lin_mean))
    vv_min_db = float(np.nanmin(vv_db))
    vv_max_db = float(np.nanmax(vv_db))
    vv_std_db = float(np.nanstd(vv_db))

    print("\n--- RECOMPUTED VH BACKSCATTER STATISTICS ---")
    print(f"  • Min dB                      : {vh_min_db:.3f} dB (Real physical minimum, -60 dB floor eliminated)")
    print(f"  • Max dB                      : {vh_max_db:.3f} dB")
    print(f"  • Median dB (Recommended)     : {vh_median_db:.3f} dB")
    print(f"  • Linear-Power Average in dB  : {vh_lin_power_avg_db:.3f} dB  (10*log10(mean(linear)))")
    print(f"  • Naive Mean of dB            : {vh_naive_mean_db:.3f} dB")
    print(f"  • Std Dev in dB               : {vh_std_db:.3f} dB")

    print("\n--- RECOMPUTED VV BACKSCATTER STATISTICS ---")
    print(f"  • Min dB                      : {vv_min_db:.3f} dB")
    print(f"  • Max dB                      : {vv_max_db:.3f} dB")
    print(f"  • Median dB (Recommended)     : {vv_median_db:.3f} dB")
    print(f"  • Linear-Power Average in dB  : {vv_lin_power_avg_db:.3f} dB  (10*log10(mean(linear)))")
    print(f"  • Naive Mean of dB            : {vv_naive_mean_db:.3f} dB")
    print(f"  • Std Dev in dB               : {vv_std_db:.3f} dB")

    # Ratio: VV - VH (dB difference)
    db_diff = vv_db - vh_db
    print("\n--- RECOMPUTED SAR RATIO (VV_dB - VH_dB) ---")
    print(f"  • Median dB Difference        : {np.nanmedian(db_diff):.3f} dB")
    print(f"  • Linear-Power Ratio in dB    : {vv_lin_power_avg_db - vh_lin_power_avg_db:.3f} dB")
    print(f"  • Linear-Power Ratio (linear) : {10.0 ** ((vv_lin_power_avg_db - vh_lin_power_avg_db)/10.0):.3f}")

    # 4. Locate and Overlay +18 dB pixels
    high_vv_mask = (vv_db >= 10.0)
    super_high_mask = (vv_db >= 15.0)
    high_count = np.count_nonzero(high_vv_mask)
    super_high_count = np.count_nonzero(super_high_mask)
    print(f"\n--- HIGH BACKSCATTER PIXELS (ANOMALIES / URBAN REFLECTORS) ---")
    print(f"  • Pixels with VV >= +10 dB: {high_count} ({high_count/vv_raw.size*100:.3f}%)")
    print(f"  • Pixels with VV >= +15 dB (up to {vv_max_db:.2f} dB): {super_high_count}")
    
    # Where are they located?
    ys, xs = np.where(super_high_mask)
    for y, x in zip(ys[:10], xs[:10]):
        # Map pixel (y, x) to geographic coordinates
        pix_lat = lat + delta - (y / 256.0) * (2 * delta)
        pix_lon = lon - delta + (x / 256.0) * (2 * delta)
        print(f"    Pixel ({y:3d}, {x:3d}) -> VV={vv_db[y, x]:.2f} dB | Lat={pix_lat:.6f}, Lon={pix_lon:.6f}")

    # Create overlay image
    # Base grayscale from VV (-25 dB to 0 dB)
    vv_disp = np.nan_to_num(vv_db, nan=-25.0)
    vv_norm = np.clip((vv_disp - (-25.0)) / (0.0 - (-25.0)), 0.0, 1.0)
    base_gray = (vv_norm * 255.0).astype(np.uint8)
    overlay_rgb = np.stack([base_gray, base_gray, base_gray], axis=-1)
    
    # Color super-high reflectors in Neon Magenta/Red ([255, 0, 128]) with 3x3 halo
    for y, x in zip(ys, xs):
        y_min, y_max = max(0, y-2), min(255, y+3)
        x_min, x_max = max(0, x-2), min(255, x+3)
        overlay_rgb[y_min:y_max, x_min:x_max] = [255, 30, 80] # Neon Red highlight

    img = Image.fromarray(overlay_rgb, mode='RGB')
    out_dir = os.path.join(os.getcwd(), "evidence_chips")
    os.makedirs(out_dir, exist_ok=True)
    overlay_path = os.path.join(out_dir, "sar_vv_high_reflector_overlay.png")
    img.save(overlay_path)
    print(f"\nSaved +18 dB overlay map to: {overlay_path}")

if __name__ == '__main__':
    asyncio.run(test_q4())

