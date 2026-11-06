import asyncio
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__)))
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
from services.copernicus_auth import CopernicusAuthService
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        # Search for T051151 in CDSE OData
        url1 = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=contains(Name,'20240105T051151')"
        r1 = await client.get(url1)
        print("Search for 20240105T051151:")
        for p in r1.json().get('value', []):
            print(" -", p.get('Name'))

        # Search for T050211 in CDSE OData
        url2 = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=contains(Name,'20240105T050211') and contains(Name,'T44QND')"
        r2 = await client.get(url2)
        print("\nSearch for 20240105T050211 over T44QND (Tadepalligudem):")
        for p in r2.json().get('value', []):
            print(" -", p.get('Name'))

if __name__ == '__main__':
    asyncio.run(main())
