import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
import base64

from services.sentinel2_service import (
    Sentinel2Service,
    NoImageryFoundError,
    InvalidAOIError,
    InvalidDateRangeError,
    Sentinel2Error,
)
from services.copernicus_auth import CopernicusAuthService, CopernicusCredentialsMissingError

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

def test_invalid_aoi_geometry():
    service = Sentinel2Service()
    with pytest.raises(InvalidAOIError):
        service.validate_inputs({}, "2026-10-01", "2026-10-26", 30.0)

    with pytest.raises(InvalidAOIError):
        service.validate_inputs({"type": "Point", "coordinates": [81.5, 16.8]}, "2026-10-01", "2026-10-26", 30.0)

def test_invalid_date_range():
    service = Sentinel2Service()
    # Start after End
    with pytest.raises(InvalidDateRangeError):
        service.validate_inputs(SAMPLE_AOI, "2026-10-26", "2026-10-01", 30.0)

    # Malformed format
    with pytest.raises(InvalidDateRangeError):
        service.validate_inputs(SAMPLE_AOI, "10-01-2026", "2026-10-26", 30.0)

@pytest.mark.asyncio
async def test_missing_credentials_demo_mode_false():
    mock_auth = CopernicusAuthService(client_id="", client_secret="")
    service = Sentinel2Service(auth_service=mock_auth)

    with patch.object(Sentinel2Service, "is_demo_mode", False):
        with pytest.raises(CopernicusCredentialsMissingError):
            await service.search_imagery(SAMPLE_AOI, "2026-10-01", "2026-10-26", 30.0)

@pytest.mark.asyncio
async def test_demo_mode_search_and_preview():
    mock_auth = CopernicusAuthService(client_id="", client_secret="")
    service = Sentinel2Service(auth_service=mock_auth)

    with patch.object(Sentinel2Service, "is_demo_mode", True):
        # Demo search
        search_res = await service.search_imagery(SAMPLE_AOI, "2026-10-01", "2026-10-26", 30.0)
        assert search_res["status"] == "success"
        assert search_res["is_demo"] is True
        assert "S2A_MSIL2A_DEMO" in search_res["scene_id"]
        assert search_res["bands"] == ["B04", "B03", "B02", "B08"]

        # Demo preview
        preview_res = await service.get_preview_image(SAMPLE_AOI, "2026-10-01", "2026-10-26", 30.0)
        assert preview_res["status"] == "success"
        assert preview_res["is_demo"] is True
        assert preview_res["image_base64"].startswith("data:image/png;base64,")

@pytest.mark.asyncio
async def test_no_imagery_found_live_api():
    mock_auth = CopernicusAuthService(client_id="valid-id", client_secret="valid-secret")
    service = Sentinel2Service(auth_service=mock_auth)

    mock_catalog_resp = MagicMock(spec=httpx.Response)
    mock_catalog_resp.status_code = 200
    mock_catalog_resp.json.return_value = {
        "features": []  # Empty results
    }

    with patch.object(Sentinel2Service, "is_demo_mode", False):
        with patch.object(mock_auth, "get_access_token", new_callable=AsyncMock) as mock_token:
            mock_token.return_value = "mock-token"
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_catalog_resp
                with pytest.raises(NoImageryFoundError) as exc_info:
                    await service.search_imagery(SAMPLE_AOI, "2026-10-01", "2026-10-26", 30.0)
                assert "No suitable Sentinel-2 imagery was found" in str(exc_info.value)

@pytest.mark.asyncio
async def test_successful_satellite_live_api_search_and_preview():
    mock_auth = CopernicusAuthService(client_id="valid-id", client_secret="valid-secret")
    service = Sentinel2Service(auth_service=mock_auth)

    mock_catalog_resp = MagicMock(spec=httpx.Response)
    mock_catalog_resp.status_code = 200
    mock_catalog_resp.json.return_value = {
        "features": [
            {
                "id": "S2B_MSIL2A_20261015T052029_N0500_R019_T44QND",
                "properties": {
                    "datetime": "2026-10-15T05:20:29Z",
                    "eo:cloud_cover": 3.45,
                },
            }
        ]
    }

    # 1x1 transparent PNG byte mock
    tiny_png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    mock_process_resp = MagicMock(spec=httpx.Response)
    mock_process_resp.status_code = 200
    mock_process_resp.content = tiny_png_bytes

    with patch.object(Sentinel2Service, "is_demo_mode", False):
        with patch.object(mock_auth, "get_access_token", new_callable=AsyncMock) as mock_token:
            mock_token.return_value = "mock-token"
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                # First post is search catalog, second is process API
                mock_post.side_effect = [mock_catalog_resp, mock_process_resp]

                preview_res = await service.get_preview_image(SAMPLE_AOI, "2026-10-01", "2026-10-26", 30.0)
                assert preview_res["status"] == "success"
                assert preview_res["is_demo"] is False
                assert preview_res["scene_id"] == "S2B_MSIL2A_20261015T052029_N0500_R019_T44QND"
                assert preview_res["cloud_coverage"] == 3.45
                assert preview_res["data_source"] == "Copernicus Sentinel-2"
                assert preview_res["image_base64"].startswith("data:image/png;base64,")
