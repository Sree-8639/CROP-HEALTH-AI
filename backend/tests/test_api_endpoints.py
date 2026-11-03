import pytest
from httpx import AsyncClient, ASGITransport
from main import app
from services.sentinel2_service import Sentinel2Service

SAMPLE_AOI = {
    "type": "Polygon",
    "coordinates": [
        [
            [81.516592, 16.800244],
            [81.536592, 16.800244],
            [81.536592, 16.820244],
            [81.516592, 16.820244],
            [81.516592, 16.800244],
        ]
    ],
}

@pytest.mark.asyncio
async def test_api_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["phase"] in (4, 5, 6)

@pytest.mark.asyncio
async def test_satellite_search_invalid_dates():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "geometry": SAMPLE_AOI,
            "start_date": "2026-10-26",
            "end_date": "2026-10-01", # Invalid: start > end
            "max_cloud_coverage": 30.0,
        }
        response = await client.post("/api/v1/satellite/search", json=payload)
        assert response.status_code == 400
        assert "cannot be after end date" in response.json()["detail"]

@pytest.mark.asyncio
async def test_satellite_search_invalid_geometry():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "geometry": {"type": "LineString", "coordinates": [[10, 10], [20, 20]]},
            "start_date": "2026-10-01",
            "end_date": "2026-10-26",
            "max_cloud_coverage": 30.0,
        }
        response = await client.post("/api/v1/satellite/search", json=payload)
        assert response.status_code == 400
        assert "Unsupported geometry type" in response.json()["detail"]

@pytest.mark.asyncio
async def test_satellite_preview_demo_mode():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "geometry": SAMPLE_AOI,
            "start_date": "2026-10-01",
            "end_date": "2026-10-26",
            "max_cloud_coverage": 30.0,
        }
        response = await client.post("/api/v1/satellite/preview", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["is_demo"] is True
        assert data["image_base64"].startswith("data:image/png;base64,")
        assert data["bands"] == ["B04", "B03", "B02", "B08"]
        assert data["processing_level"] == "Level-2A"
        assert data["data_source"] == "Copernicus Sentinel-2"

@pytest.mark.asyncio
async def test_ndvi_analysis_endpoint_demo():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "geometry": SAMPLE_AOI,
            "start_date": "2026-10-01",
            "end_date": "2026-10-26",
            "max_cloud_coverage": 30.0,
            "aoi_area_km2": 12.57,
        }
        response = await client.post("/api/v1/analysis/ndvi", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["is_demo"] is True
        assert data["ndvi_map"].startswith("data:image/png;base64,")
        assert "statistics" in data
        stats = data["statistics"]
        assert stats["valid_pixels"] > 0
        assert stats["mean_ndvi"] is not None
        assert "vegetation_vigor_breakdown" in stats

@pytest.mark.asyncio
async def test_ndvi_analysis_invalid_dates():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "geometry": SAMPLE_AOI,
            "start_date": "2026-10-26",
            "end_date": "2026-10-01", # Invalid
            "max_cloud_coverage": 30.0,
        }
        response = await client.post("/api/v1/analysis/ndvi", json=payload)
        assert response.status_code == 400
        assert "cannot be after end date" in response.json()["detail"]

