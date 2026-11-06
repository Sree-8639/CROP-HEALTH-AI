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

    # Query Hasdeo Arand AOI
    wkt = "POLYGON((82.55 22.75, 82.75 22.75, 82.75 22.95, 82.55 22.95, 82.55 22.75))"
    filter_expr = f"Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL2A') and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and ContentDate/Start ge 2021-01-01T00:00:00.000Z and ContentDate/Start le 2021-03-31T23:59:59.000Z"
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?" + urllib.parse.urlencode({
        "$filter": filter_expr,
        "$top": 5,
        "$orderby": "ContentDate/Start asc"
    })

    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Found S2 products:", len(res.get("value", [])))
        for p in res.get("value", []):
            print("  S2:", p["Name"], p["ContentDate"]["Start"])

if __name__ == "__main__":
    asyncio.run(main())
