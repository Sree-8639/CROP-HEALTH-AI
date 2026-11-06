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
from services.ndvi_service import create_valid_pixel_mask

async def test_q1():
    lat = 16.810244
    lon = 81.526592
    r_km = 1.0
    delta = r_km / 111.32
    geom = {
        'type': 'Polygon',
        'coordinates': [[[lon-delta, lat-delta], [lon+delta, lat-delta], [lon+delta, lat+delta], [lon-delta, lat+delta], [lon-delta, lat-delta]]]
    }
    all_req_bands = ['B02', 'B03', 'B04', 'B05', 'B08', 'B11']

    print("================================================================================")
    print("RUN 1: Exact parameters from inspect_mask_details.py (2024-01-01 to 2024-01-30, max_cloud=30.0)")
    print("================================================================================")
    sc1, b1 = await sentinel2_service._fetch_multispectral_bands(geom, '2024-01-01', '2024-01-30', 30.0, all_req_bands)
    print("Run 1 scene_info:", sc1.get('scene_id'), sc1.get('acquisition_date'))
    scl1 = b1.get('SCL')
    dm1 = b1.get('dataMask')
    mask1 = create_valid_pixel_mask(b1['B04'], b1['B08'], scl1, dm1)
    val_pct1 = (np.count_nonzero(mask1) / mask1.size) * 100.0
    ndvi1 = (b1['B08'] - b1['B04']) / (b1['B08'] + b1['B04'] + 1e-9)
    print(f"Run 1 Valid %: {val_pct1:.2f}% | Masked Mean NDVI: {np.mean(ndvi1[mask1]):.4f} | Raw Mean NDVI: {np.mean(ndvi1):.4f}")
    
    print("\nRun 1 SCL Class Histogram:")
    for code in range(12):
        cnt = int(np.count_nonzero(scl1 == code))
        if cnt > 0:
            print(f"  Class {code:2d}: {cnt:5d} pixels ({cnt/scl1.size*100:5.2f}%)")

    print("\n================================================================================")
    print("RUN 2: Exact parameters from generate_3field_table.py (2023-12-25 to 2024-01-08, max_cloud=35.0)")
    print("================================================================================")
    sc2, b2 = await sentinel2_service._fetch_multispectral_bands(geom, '2023-12-25', '2024-01-08', 35.0, all_req_bands)
    print("Run 2 scene_info:", sc2.get('scene_id'), sc2.get('acquisition_date'))
    scl2 = b2.get('SCL')
    dm2 = b2.get('dataMask')
    mask2 = create_valid_pixel_mask(b2['B04'], b2['B08'], scl2, dm2)
    val_pct2 = (np.count_nonzero(mask2) / mask2.size) * 100.0
    ndvi2 = (b2['B08'] - b2['B04']) / (b2['B08'] + b2['B04'] + 1e-9)
    print(f"Run 2 Valid %: {val_pct2:.2f}% | Masked Mean NDVI: {np.mean(ndvi2[mask2]):.4f} | Raw Mean NDVI: {np.mean(ndvi2):.4f}")

    print("\nRun 2 SCL Class Histogram:")
    for code in range(12):
        cnt = int(np.count_nonzero(scl2 == code))
        if cnt > 0:
            print(f"  Class {code:2d}: {cnt:5d} pixels ({cnt/scl2.size*100:5.2f}%)")

    # Check if bands are identical or different
    b_diff = np.max(np.abs(b1['B04'] - b2['B04']))
    print(f"\nMax difference between Run 1 and Run 2 B04 band: {b_diff}")

if __name__ == '__main__':
    asyncio.run(test_q1())

