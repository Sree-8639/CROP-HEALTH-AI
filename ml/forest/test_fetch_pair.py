import os
import sys
import asyncio
import time
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

load_dotenv(os.path.join(BACKEND_DIR, ".env"), override=True)
os.environ["DEMO_MODE"] = "false"

from services.sentinel2_service import Sentinel2Service
from services.sentinel1_service import Sentinel1Service

async def main():
    s2 = Sentinel2Service()
    s1 = Sentinel1Service()
    geom = {
        "type": "Polygon",
        "coordinates": [[[81.55, 17.45], [81.65, 17.45], [81.65, 17.55], [81.55, 17.55], [81.55, 17.45]]]
    }
    
    # 1. Fetch S2 T0 (Jan 2020)
    print("Fetching S2 T0 (Jan 2020)...")
    s2_t0, bands_t0 = await s2._fetch_multispectral_bands(
        geometry=geom, start_date="2020-01-01", end_date="2020-01-10",
        max_cloud_coverage=20.0, band_names=["B02", "B03", "B04", "B08", "B11"]
    )
    print("S2 T0 Scene:", s2_t0["scene_id"])

    # 2. Fetch S2 T1 (Jan 2021)
    print("Fetching S2 T1 (Jan 2021)...")
    s2_t1, bands_t1 = await s2._fetch_multispectral_bands(
        geometry=geom, start_date="2021-01-01", end_date="2021-01-15",
        max_cloud_coverage=20.0, band_names=["B02", "B03", "B04", "B08", "B11"]
    )
    print("S2 T1 Scene:", s2_t1["scene_id"])

    # 3. Fetch S1 passes
    s1_cat = await s1.search_imagery(geom, "2020-01-01", "2021-01-31")
    print("S1 passes found:", len(s1_cat))

if __name__ == "__main__":
    asyncio.run(main())

