import asyncio
import httpx

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Search all Sentinel-1 GRDH products intersecting the polygon over Jan-Feb 2024
        poly = "POLYGON((81.4 16.4, 81.8 16.4, 81.8 16.9, 81.4 16.9, 81.4 16.4))"
        url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-1' and ContentDate/Start gt 2024-01-01T00:00:00.000Z and ContentDate/Start lt 2024-02-29T23:59:59.000Z and OData.CSC.Intersects(area=geography'SRID=4326;{poly}')&$top=100"
        res = await client.get(url)
        val = res.json().get('value', [])
        print(f"Total Sentinel-1 products: {len(val)}")
        grd_products = [p for p in val if "GRDH" in p.get("Name", "")]
        print(f"Total GRDH products: {len(grd_products)}")
        for p in grd_products:
            print(" -", p.get("Name"))

if __name__ == '__main__':
    asyncio.run(main())
