import asyncio
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__)))
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
from services.copernicus_auth import CopernicusAuthService
import httpx
import json

async def test_meta():
    auth = CopernicusAuthService()
    token = await auth.get_access_token()
    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/tar"  # When requesting userdata + tiff, format is tar or multipart
    }
    
    # Or test updateOutputMetadata
    evalscript = """//VERSION=3
function setup() {
  return {
    input: [
      {
        bands: ["B04", "B08", "SCL", "dataMask"]
      }
    ],
    output: [
      {
        id: "default",
        bands: 4,
        sampleType: "FLOAT32"
      },
      {
        id: "userdata",
        type: "application/json"
      }
    ]
  };
}

function evaluatePixel(sample) {
  return {
    default: [sample.B04, sample.B08, sample.SCL, sample.dataMask]
  };
}

function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
  outputMetadata.userData = {
    orbits: scenes.orbits,
    inputBounds: inputMetadata.bounds
  };
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
            "responses": [
                {"identifier": "default", "format": {"type": "image/tiff"}},
                {"identifier": "userdata", "format": {"type": "application/json"}}
            ]
        },
        "evalscript": evalscript
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        print("Status:", r.status_code, flush=True)
        if r.status_code == 200:
            print("Got tar/multipart response! Length:", len(r.content), flush=True)
            # Inspect tar content
            import tarfile
            import io
            with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:*") as tar:
                for member in tar.getmembers():
                    print("Tar member:", member.name)
                    if member.name.endswith(".json"):
                        f = tar.extractfile(member)
                        print("UserData JSON:", f.read().decode("utf-8"))
        else:
            print("Error:", r.text, flush=True)

if __name__ == '__main__':
    asyncio.run(test_meta())
