import os
import sys
import asyncio
import numpy as np
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv('.env', override=True)
os.environ['DEMO_MODE'] = 'false'
from services.sentinel2_service import sentinel2_service
from services.ndvi_service import create_valid_pixel_mask

async def inspect_mask():
    lat, lon = 16.810244, 81.526592
    r_km = 1.0
    delta = r_km / 111.32
    geom = {
        'type': 'Polygon',
        'coordinates': [[[lon-delta, lat-delta], [lon+delta, lat-delta], [lon+delta, lat+delta], [lon-delta, lat+delta], [lon-delta, lat-delta]]]
    }
    all_req_bands = ['B02', 'B04', 'B05', 'B08', 'B11']
    scene_info, bands = await sentinel2_service._fetch_multispectral_bands(
        geom, '2024-01-01', '2024-01-30', 30.0, all_req_bands
    )
    scl = bands.get('SCL')
    data_mask = bands.get('dataMask')
    b04 = bands['B04']
    b08 = bands['B08']
    
    print('================================================================')
    print('S2 MASK INSPECTION: WHY ARE 82% INVALID WITH 0.74% CLOUD?')
    print('================================================================')
    print(f'Total pixels in raster: {b04.size} (Shape: {b04.shape})')
    
    if data_mask is not None:
        dm_valid = np.count_nonzero(data_mask > 0)
        dm_invalid = np.count_nonzero(data_mask == 0)
        print(f'\ndataMask (Satellite Swath / Spatial Footprint):')
        print(f'  • Inside satellite granule footprint (dataMask > 0): {dm_valid} ({dm_valid/data_mask.size*100:.2f}%)')
        print(f'  • Outside granule boundary / NoData (dataMask == 0): {dm_invalid} ({dm_invalid/data_mask.size*100:.2f}%)')
    
    if scl is not None:
        print('\nSentinel-2 SCL (Scene Classification Layer) Pixel Breakdown:')
        scl_names = {
            0: 'NO_DATA (Missing / Out of orbit swath)',
            1: 'SATURATED_OR_DEFECTIVE',
            2: 'DARK_AREA_PIXELS',
            3: 'CLOUD_SHADOWS',
            4: 'VEGETATION',
            5: 'NOT_VEGETATED / BARE SOIL',
            6: 'WATER',
            7: 'UNCLASSIFIED',
            8: 'CLOUD_MEDIUM_PROBABILITY',
            9: 'CLOUD_HIGH_PROBABILITY',
            10: 'THIN_CIRRUS',
            11: 'SNOW / ICE'
        }
        for code in range(12):
            cnt = int(np.count_nonzero(scl == code))
            if cnt > 0:
                pct = (cnt / scl.size) * 100.0
                is_rejected = code in [0, 1, 3, 8, 9, 10, 11]
                status = "REJECTED BY MASK" if is_rejected else "KEPT IN MASK"
                print(f'  Class {code:2d} - {scl_names.get(code, "Unknown"):40s}: {cnt:5d} pixels ({pct:5.2f}%) [{status}]')
                
    raw_ndvi = (b08 - b04) / (b08 + b04 + 1e-9)
    print('\n----------------------------------------------------------------')
    print('NDVI STATISTICAL COMPARISON: WITHOUT MASK vs WITH MASK')
    print('----------------------------------------------------------------')
    print('1. WITHOUT MASK (Raw unmasked raster - includes NoData/0 pixels):')
    print(f'   • Pixel count: {raw_ndvi.size}')
    print(f'   • Mean NDVI  : {np.nanmean(raw_ndvi):.4f}')
    print(f'   • Min NDVI   : {np.nanmin(raw_ndvi):.4f}')
    print(f'   • Max NDVI   : {np.nanmax(raw_ndvi):.4f}')
    print(f'   • Std Dev    : {np.nanstd(raw_ndvi):.4f}')
    
    mask = create_valid_pixel_mask(b04, b08, scl, data_mask)
    masked_ndvi = raw_ndvi[mask]
    print('\n2. WITH MASK (Strictly valid land pixels):')
    print(f'   • Pixel count: {masked_ndvi.size} ({masked_ndvi.size/b04.size*100:.2f}% of bounding box)')
    print(f'   • Mean NDVI  : {np.mean(masked_ndvi):.4f}')
    print(f'   • Min NDVI   : {np.min(masked_ndvi):.4f}')
    print(f'   • Max NDVI   : {np.max(masked_ndvi):.4f}')
    print(f'   • Std Dev    : {np.std(masked_ndvi):.4f}')

if __name__ == '__main__':
    asyncio.run(inspect_mask())

