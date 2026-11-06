import os
import asyncio
import json
from dotenv import load_dotenv

# Load backend .env and force DEMO_MODE=false
load_dotenv('.env', override=True)
os.environ['DEMO_MODE'] = 'false'

from services.sentinel2_service import sentinel2_service
from services.sentinel1_service import sentinel1_service
from services.fusion_service import multimodal_fusion_service

async def run_live_test():
    # Real field in Tadepalligudem, West Godavari, Andhra Pradesh, India
    lat = 16.810244
    lon = 81.526592
    r_km = 1.0
    delta_deg = r_km / 111.32
    geometry = {
        'type': 'Polygon',
        'coordinates': [[
            [lon - delta_deg, lat - delta_deg],
            [lon + delta_deg, lat - delta_deg],
            [lon + delta_deg, lat + delta_deg],
            [lon - delta_deg, lat + delta_deg],
            [lon - delta_deg, lat - delta_deg],
        ]]
    }
    
    # 30-day temporal window
    start_date = '2024-01-01'
    end_date = '2024-01-30'
    max_cloud = 30.0
    
    print('================================================================')
    print('STEP 1: LIVE SENTINEL-2 RETRIEVAL & SPECTRAL INDICES (CDSE)')
    print('================================================================')
    s2_res = None
    try:
        s2_res = await sentinel2_service.calculate_all_indices_for_aoi(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
            max_cloud_coverage=max_cloud,
        )
        print('SUCCESS: Sentinel-2 Retrieval Succeeded!')
        print('  Product / Scene ID:', s2_res.get('scene', {}).get('id'))
        print('  Acquisition Date:', s2_res.get('scene', {}).get('acquisition_date'))
        print('  Cloud Coverage (%):', s2_res.get('quality', {}).get('cloud_coverage'))
        print('  Valid Pixel (%):', s2_res.get('quality', {}).get('valid_percentage'))
        print('  Is Demo Flag:', s2_res.get('is_demo'))
        indices = s2_res.get('indices', {})
        for idx in ['ndvi', 'ndmi', 'ndre', 'evi']:
            st = indices.get(idx, {}).get('statistics', {})
            print(f'  {idx.upper()}: mean={st.get("mean")}, min={st.get("min")}, max={st.get("max")}, std={st.get("std")}')
    except Exception as e:
        print('FAILED: Sentinel-2 Retrieval Failed:')
        print('  Exception Type:', type(e).__name__)
        print('  Exception Message:', str(e))
        import traceback
        traceback.print_exc()

    print('\n================================================================')
    print('STEP 2: LIVE SENTINEL-1 SAR RETRIEVAL (CDSE)')
    print('================================================================')
    s1_res = None
    try:
        s1_res = await sentinel1_service.get_sar_analysis(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
        )
        print('SUCCESS: Sentinel-1 SAR Retrieval Succeeded!')
        print('  Product / Scene ID:', s1_res.get('scene', {}).get('id'))
        print('  Observation Date:', s1_res.get('observation_date'))
        print('  Orbit Direction:', s1_res.get('orbit_direction'))
        print('  Instrument Mode:', s1_res.get('instrument_mode'))
        print('  Is Demo Flag:', s1_res.get('is_demo'))
        sar_feat = s1_res.get('sar_features', {})
        for feat in ['vv', 'vh', 'vv_vh_ratio']:
            st = sar_feat.get(feat, {}).get('statistics', {})
            print(f'  {feat.upper()}: mean={st.get("mean")} dB, min={st.get("min")} dB, max={st.get("max")} dB, std={st.get("std")}')
    except Exception as e:
        print('FAILED: Sentinel-1 Retrieval Failed:')
        print('  Exception Type:', type(e).__name__)
        print('  Exception Message:', str(e))
        import traceback
        traceback.print_exc()

    print('\n================================================================')
    print('STEP 3: MULTIMODAL FEATURE FUSION (S2 + S1)')
    print('================================================================')
    try:
        fused = await multimodal_fusion_service.execute_multimodal_fusion(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
            max_cloud_coverage=max_cloud,
            temporal_window_days=7,
        )
        print('SUCCESS: End-to-End Multimodal Fusion Succeeded!')
        print('  Is Demo Flag:', fused.get('is_demo'))
        print('  Temporal Match Status:', fused.get('temporal_match', {}).get('status'))
        print('  Sentinel-2 Acquisition:', fused.get('temporal_match', {}).get('sentinel2_date'))
        print('  Sentinel-1 Acquisition:', fused.get('temporal_match', {}).get('sentinel1_date'))
        print('  Temporal Gap (days):', fused.get('temporal_match', {}).get('date_difference_days'))
        print('  Canonical Grid:', fused.get('grid'))
        print('  Quality Summary:', fused.get('quality_summary'))
        feat_stats = fused.get('features', {})
        print('  Fused Features Count:', len(feat_stats))
        for fname in ['NDVI', 'NDMI', 'NDRE', 'EVI', 'VV', 'VH', 'VV_VH_RATIO']:
            f_obj = feat_stats.get(fname, {})
            st = f_obj.get('statistics', {})
            print(f'    {fname}: valid_pct={f_obj.get("valid_percentage")}%, mean={st.get("mean")}, std={st.get("std")}')
    except Exception as e:
        print('FAILED: Multimodal Fusion Failed:')
        print('  Exception Type:', type(e).__name__)
        print('  Exception Message:', str(e))
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(run_live_test())
