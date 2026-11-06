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

async def test_date_bands():
    auth = CopernicusAuthService()
    token = await auth.get_access_token()
    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    # Evalscript with SIMPLE mosaicking to get scene date
    evalscript = """//VERSION=3
function setup() {
  return {
    input: [
      {
        bands: ["B04", "B08", "SCL", "dataMask"]
      }
    ],
    mosaicking: "SIMPLE",
    output: {
      bands: 7,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(samples, scenes) {
  // scenes[0].date is a Javascript Date object
  let d = scenes[0].date;
  let yr = d.getUTCFullYear();
  let mo = d.getUTCMonth() + 1;
  let dy = d.getUTCDate();
  let hr = d.getUTCHours();
  let mi = d.getUTCMinutes();
  let sc = d.getUTCSeconds();
  return [samples[0].B04, samples[0].B08, samples[0].SCL, samples[0].dataMask, yr * 10000 + mo * 100 + dy, hr * 100 + mi, sc];
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
            "width": 16,
            "height": 16,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        },
        "evalscript": evalscript
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(url, json=payload, headers=headers)
        print("Status:", r.status_code, flush=True)
        if r.status_code == 200:
            arr = tifffile.imread(io.BytesIO(r.content))
            print("Retrieved bands shape:", arr.shape, flush=True)
            # Band 4 is YYYYMMDD, Band 5 is HHMM, Band 6 is SS
            print(f"Date pixel [0,0]: YYYYMMDD = {arr[0, 0, 4]}, HHMM = {arr[0, 0, 5]}, SS = {arr[0, 0, 6]}", flush=True)
        else:
            print("Error:", r.text, flush=True)

if __name__ == '__main__':
    asyncio.run(test_date_bands())
