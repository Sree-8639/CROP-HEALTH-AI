import asyncio
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__)))
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
from services.copernicus_auth import CopernicusAuthService
import httpx
import tifffile
import io

async def test_meta():
    print("Testing auth...", flush=True)
    auth = CopernicusAuthService()
    token = await auth.get_access_token()
    print("Got token!", flush=True)
    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    # In Sentinel Hub Evalscript V3, samples has date components or scenes.orbits
    # But wait, what if we output date components as bands?
    # e.g., year, month, day, hour, minute!
    # In V3: scenes.orbits[0].dateFrom or scenes[0].date
    # Or in evaluatePixel: samples[0] has date? No, evaluatePixel receives (samples, scenes) if mosaicking is SIMPLE or ORBIT.
    # In default SINGLE mosaicking:
    # Let's test what scenes object contains:
    evalscript = """//VERSION=3
function setup() {
  return {
    input: [
      {
        bands: ["B04", "B08", "SCL", "dataMask"],
        metadata: ["bounds"]
      }
    ],
    output: {
      bands: 4,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  return [sample.B04, sample.B08, sample.SCL, sample.dataMask];
}
"""
    payload = {
        "input": {
            "bounds": {"bbox": [81.503, 16.793, 81.543, 16.833]},
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": "2024-01-05T05:00:00Z",
                        "to": "2024-01-05T05:30:00Z"
                    }
                }
            }]
        },
        "output": {
            "width": 64,
            "height": 64,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        },
        "evalscript": evalscript
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(url, json=payload, headers=headers)
        print("Status:", r.status_code)
        if r.status_code == 200:
            print("Successfully retrieved 64x64 TIFF with pinned timeRange!")
            data = tifffile.imread(io.BytesIO(r.content))
            print("Shape:", data.shape)
        else:
            print("Error:", r.text)

if __name__ == '__main__':
    asyncio.run(test_meta())
