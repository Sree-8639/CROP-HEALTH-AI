import os
import sys
import json
import asyncio
import urllib.request
import urllib.parse
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

load_dotenv(os.path.join(BACKEND_DIR, ".env"), override=True)

from services.copernicus_auth import copernicus_auth

async def main():
    token = await copernicus_auth.get_access_token()
    print("Token acquired successfully:", token[:15] + "...")

    # Query Papikonda AOI [81.35, 17.55, 81.55, 17.75]
    wkt = "POLYGON((81.35 17.55, 81.55 17.55, 81.55 17.75, 81.35 17.75, 81.35 17.55))"
    
    # 1. S2 T0 (Early 2020)
    filter_t0 = f"Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL2A') and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and ContentDate/Start ge 2020-01-01T00:00:00.000Z and ContentDate/Start le 2020-02-28T23:59:59.000Z"
    url_t0 = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?" + urllib.parse.urlencode({
        "$filter": filter_t0, "$top": 3, "$orderby": "ContentDate/Start asc"
    })
    req = urllib.request.Request(url_t0, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("T0 S2 Products (2020):", len(res.get("value", [])))
        for p in res.get("value", []):
            print("  S2 T0:", p["Name"], p["ContentDate"]["Start"])

    # 2. S2 T1 (Early 2021)
    filter_t1 = f"Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL2A') and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and ContentDate/Start ge 2021-01-01T00:00:00.000Z and ContentDate/Start le 2021-02-28T23:59:59.000Z"
    url_t1 = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?" + urllib.parse.urlencode({
        "$filter": filter_t1, "$top": 3, "$orderby": "ContentDate/Start asc"
    })
    req = urllib.request.Request(url_t1, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("T1 S2 Products (2021):", len(res.get("value", [])))
        for p in res.get("value", []):
            print("  S2 T1:", p["Name"], p["ContentDate"]["Start"])

if __name__ == "__main__":
    asyncio.run(main())

