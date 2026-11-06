import os
import sys
import asyncio
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv('.env', override=True)
os.environ['DEMO_MODE'] = 'false'
from services.sentinel1_service import sentinel1_service

async def test_q2():
    lat = 16.810244
    lon = 81.526592
    r_km = 1.0
    delta = r_km / 111.32
    geom = {
        'type': 'Polygon',
        'coordinates': [[[lon-delta, lat-delta], [lon+delta, lat-delta], [lon+delta, lat+delta], [lon-delta, lat+delta], [lon-delta, lat-delta]]]
    }
    
    print("================================================================================")
    print("INVESTIGATION 2: ARE '01-26' AND '01-30' THE SAME SCENE?")
    print("================================================================================")
    
    # 1. Search S1 from 2024-01-01 to 2024-01-30
    print("\nQuerying Sentinel-1 catalog (2024-01-01 to 2024-01-30):")
    scenes = await sentinel1_service.search_imagery(geom, '2024-01-01', '2024-01-30')
    for idx, s in enumerate(scenes):
        print(f"\nScene #{idx+1}:")
        for k in ['id', 'Name', 'date', 'acquisition_date', 'properties', 'GeoFootprint']:
            if k in s:
                val = s[k]
                if isinstance(val, dict):
                    print(f"  {k}: {list(val.keys())}")
                    for subk, subv in val.items():
                        print(f"    {subk}: {subv}")
                else:
                    print(f"  {k}: {val}")

    # Now let's call get_sar_analysis with both windows and inspect the actual Process API payload / scene returned
    print("\n--------------------------------------------------------------------------------")
    print("Call A: get_sar_analysis(geom, '2024-01-01', '2024-01-30')")
    res_a = await sentinel1_service.get_sar_analysis(geom, '2024-01-01', '2024-01-30')
    print("  Result A Observation Date:", res_a.get('observation_date'))
    print("  Result A Product Info:", res_a.get('product_info'))
    print("  Result A VV Mean:", res_a.get('sar_features', {}).get('vv', {}).get('statistics', {}).get('mean'))

    print("\nCall B: get_sar_analysis(geom, '2024-01-10', '2024-01-26', s2_date='2024-01-10')")
    res_b = await sentinel1_service.get_sar_analysis(geom, '2024-01-10', '2024-01-26', s2_date='2024-01-10')
    print("  Result B Observation Date:", res_b.get('observation_date'))
    print("  Result B Product Info:", res_b.get('product_info'))
    print("  Result B VV Mean:", res_b.get('sar_features', {}).get('vv', {}).get('statistics', {}).get('mean'))

if __name__ == '__main__':
    asyncio.run(test_q2())

