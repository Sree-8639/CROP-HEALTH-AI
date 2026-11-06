import asyncio
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__)))
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
from services.copernicus_auth import CopernicusAuthService
import httpx

AOIS = {
    "Tadepalligudem": [81.503, 16.793, 81.543, 16.833],
    "Bhimavaram": [81.501, 16.525, 81.541, 16.565],
    "Maruteru": [81.725, 16.614, 81.765, 16.654]
}

async def query_s1_catalog():
    auth = CopernicusAuthService()
    token = await auth.get_access_token()
    
    url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    for aoi_name, bbox in AOIS.items():
        print(f"\n==========================================")
        print(f"Querying S1 Catalog for {aoi_name} (Jan-Feb 2024)")
        print(f"Bbox: {bbox}")
        print(f"==========================================")
        
        payload = {
            "bbox": bbox,
            "datetime": "2024-01-01T00:00:00Z/2024-02-29T23:59:59Z",
            "collections": ["sentinel-1-grd"],
            "limit": 100
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(url, json=payload, headers=headers)
            if res.status_code != 200:
                print(f"Error {res.status_code}: {res.text}")
                continue
            data = res.json()
            features = data.get("features", [])
            print(f"Total Sentinel-1 GRD scenes found: {len(features)}")
            
            # Sort by datetime
            features.sort(key=lambda x: x.get("properties", {}).get("datetime", ""))
            
            for f in features:
                props = f.get("properties", {})
                sid = f.get("id")
                dt = props.get("datetime")
                orbit_direction = props.get("sat:orbit_state")
                relative_orbit = props.get("sat:relative_orbit")
                platform = props.get("platform")
                polarizations = props.get("s1:polarizations")
                print(f"ID: {sid}")
                print(f"  Datetime: {dt} | Dir: {orbit_direction} | Rel Orbit: {relative_orbit} | Plat: {platform} | Pols: {polarizations}")

if __name__ == '__main__':
    asyncio.run(query_s1_catalog())
