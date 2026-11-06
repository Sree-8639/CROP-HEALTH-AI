import os
import sys
import asyncio
import httpx
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv('.env', override=True)
os.environ['DEMO_MODE'] = 'false'
from services.copernicus_auth import copernicus_auth

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

async def query_s1_catalog(geom):
    token = await copernicus_auth.get_access_token()
    base_url = "https://sh.dataspace.copernicus.eu"
    url = f"{base_url}/api/v1/catalog/1.0.0/search"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/geo+json, application/json, */*",
    }
    payload = {
        "collections": ["sentinel-1-grd"],
        "datetime": "2024-01-01T00:00:00Z/2024-02-29T23:59:59Z",
        "intersects": geom,
        "limit": 100,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        data = resp.json()
        return data.get("features", [])

async def main():
    print("================================================================================")
    print("ALL SENTINEL-1 ACQUISITIONS (JAN - FEB 2024) PER AOI")
    print("================================================================================")
    
    for f in FIELDS:
        lat, lon = f["lat"], f["lon"]
        r_km = 1.0
        delta = r_km / 111.32
        geom = {
            'type': 'Polygon',
            'coordinates': [[[lon-delta, lat-delta], [lon+delta, lat-delta], [lon+delta, lat+delta], [lon-delta, lat+delta], [lon-delta, lat-delta]]]
        }
        features = await query_s1_catalog(geom)
        print(f"\n--- Field: {f['name']} ({lat}, {lon}) --- Total acquisitions: {len(features)}")
        # Sort chronologically
        features.sort(key=lambda x: x.get("properties", {}).get("datetime", ""))
        for idx, feat in enumerate(features):
            props = feat.get("properties", {})
            dt = props.get("datetime")
            rel_orbit = props.get("sat:relative_orbit") or props.get("s1:relative_orbit_number") or props.get("orbitNumber")
            orbit_dir = props.get("sat:orbit_state") or props.get("orbitDirection") or "descending"
            platform = props.get("platform", "Sentinel-1A")
            pid = feat.get("id", "N/A")
            print(f"  [{idx+1:2d}] Date: {dt} | Orbit: {rel_orbit} | State: {orbit_dir:10s} | ID: {pid}")

if __name__ == '__main__':
    asyncio.run(main())

