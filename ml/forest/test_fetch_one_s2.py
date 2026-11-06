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

async def main():
    s2 = Sentinel2Service()
    geom = {
        "type": "Polygon",
        "coordinates": [[[81.35, 17.55], [81.45, 17.55], [81.45, 17.65], [81.35, 17.65], [81.35, 17.55]]]
    }
    t0 = time.time()
    print("Requesting S2 scene for 2020-01-01 to 2020-01-10...")
    scene_info, bands = await s2._fetch_multispectral_bands(
        geometry=geom,
        start_date="2020-01-01",
        end_date="2020-01-10",
        max_cloud_coverage=20.0,
        band_names=["B02", "B04", "B08", "B11"]
    )
    print(f"Done in {time.time() - t0:.2f}s! Scene: {scene_info['scene_id']}")
    print("Bands:", list(bands.keys()), "Shapes:", [b.shape for b in bands.values()])

if __name__ == "__main__":
    asyncio.run(main())

