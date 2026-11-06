import os
import sys
import asyncio
import numpy as np
from PIL import Image
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv('.env', override=True)
os.environ['DEMO_MODE'] = 'false'
from services.sentinel2_service import sentinel2_service

# Official ESA SCL Color Map
SCL_COLORS = {
    0: (0, 0, 0),        # NO_DATA - Black
    1: (255, 0, 0),      # SATURATED_DEFECTIVE - Red
    2: (47, 47, 47),     # DARK_AREA - Dark Gray
    3: (100, 50, 0),     # CLOUD_SHADOW - Brown
    4: (0, 160, 0),      # VEGETATION - Green
    5: (255, 230, 90),   # NOT_VEGETATED - Yellow/Khaki
    6: (0, 0, 255),      # WATER - Blue
    7: (128, 128, 128),  # UNCLASSIFIED - Gray
    8: (190, 190, 190),  # CLOUD_MEDIUM_PROB - Light Gray
    9: (255, 255, 255),  # CLOUD_HIGH_PROB - White
    10: (0, 200, 255),   # THIN_CIRRUS - Cyan
    11: (255, 150, 255), # SNOW - Pink
}

def render_scl_rgb(scl_array):
    h, w = scl_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for code, color in SCL_COLORS.items():
        rgb[scl_array == code] = color
    return Image.fromarray(rgb, mode='RGB')

def render_truecolor_rgb(b04, b03, b02, gain=2.5):
    # Standard Sentinel-2 true-color gain scaling
    r = np.clip(b04 * gain * 255.0, 0, 255).astype(np.uint8)
    g = np.clip(b03 * gain * 255.0, 0, 255).astype(np.uint8)
    b = np.clip(b02 * gain * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb, mode='RGB')

async def generate_chips():
    lat = 16.810244
    lon = 81.526592
    r_km = 1.0
    delta = r_km / 111.32
    geom = {
        'type': 'Polygon',
        'coordinates': [[[lon-delta, lat-delta], [lon+delta, lat-delta], [lon+delta, lat+delta], [lon-delta, lat+delta], [lon-delta, lat-delta]]]
    }
    all_bands = ['B02', 'B03', 'B04', 'B05', 'B08', 'B11']
    
    out_dir = os.path.join(os.getcwd(), "evidence_chips")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Clean Scene (Exact date 2024-01-05)
    print("Fetching exact 2024-01-05 clean scene...")
    sc_clean, b_clean = await sentinel2_service._fetch_multispectral_bands(geom, '2024-01-05', '2024-01-05', 100.0, all_bands)
    tc_clean = render_truecolor_rgb(b_clean['B04'], b_clean['B03'], b_clean['B02'])
    scl_clean = render_scl_rgb(b_clean['SCL'])
    
    tc_clean_path = os.path.join(out_dir, "s2a_20240105_truecolor_clean.png")
    scl_clean_path = os.path.join(out_dir, "s2a_20240105_scl_clean.png")
    tc_clean.save(tc_clean_path)
    scl_clean.save(scl_clean_path)
    print(f"Saved: {tc_clean_path} and {scl_clean_path}")

    # 2. Cloudy Scene (Most recent in Jan 2024, sampled by the 30-day window)
    print("Fetching late January cloudy scene (sampled when window was 2024-01-01 to 2024-01-30)...")
    sc_cloud, b_cloud = await sentinel2_service._fetch_multispectral_bands(geom, '2024-01-01', '2024-01-30', 30.0, all_bands)
    tc_cloud = render_truecolor_rgb(b_cloud['B04'], b_cloud['B03'], b_cloud['B02'])
    scl_cloud = render_scl_rgb(b_cloud['SCL'])
    
    tc_cloud_path = os.path.join(out_dir, "s2a_latejan_truecolor_cloudy.png")
    scl_cloud_path = os.path.join(out_dir, "s2a_latejan_scl_cloudy.png")
    tc_cloud.save(tc_cloud_path)
    scl_cloud.save(scl_cloud_path)
    print(f"Saved: {tc_cloud_path} and {scl_cloud_path}")

if __name__ == '__main__':
    asyncio.run(generate_chips())

