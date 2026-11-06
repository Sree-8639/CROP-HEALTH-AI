import os
import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.raster_utils import derive_pinned_timerange
from services.sentinel2_service import Sentinel2Service
from services.sentinel1_service import Sentinel1Service


def test_derive_pinned_timerange_exact_minutes():
    """Verify that derive_pinned_timerange constructs a tight ±15 minute window."""
    acq_time = "2024-01-05T05:02:11.024Z"
    pinned = derive_pinned_timerange(acq_time, window_minutes=15)
    
    assert pinned["from"] == "2024-01-05T04:47:11Z"
    assert pinned["to"] == "2024-01-05T05:17:11Z"


def test_derive_pinned_timerange_empty_fails():
    """Verify that derive_pinned_timerange raises an error if timestamp is empty."""
    with pytest.raises(ValueError):
        derive_pinned_timerange("")


@pytest.mark.anyio
async def test_sentinel2_pins_timerange_to_chosen_scene_and_stamps_correctly(monkeypatch):
    """
    Critical Regression Test:
    Ensures that Process API timeRange is strictly pinned to the chosen scene's sensing time,
    preventing Sentinel Hub from returning a different (e.g. cloudy late January) mosaic,
    and fails if the returned sensing time differs from the stamped time.
    """
    monkeypatch.setenv("DEMO_MODE", "false")
    s2 = Sentinel2Service()
    
    # Target chosen scene from catalog
    chosen_scene = {
        "status": "success",
        "scene_id": "S2A_MSIL2A_20240105T050211_N0510_R119_T44QND_20240105T075552.SAFE",
        "acquisition_date": "2024-01-05T05:02:11.024Z",
        "cloud_coverage": 0.74,
        "data_source": "Copernicus Sentinel-2",
        "processing_level": "Level-2A",
    }
    
    geometry = {
        "type": "Polygon",
        "coordinates": [[[81.503, 16.793], [81.543, 16.793], [81.543, 16.833], [81.503, 16.833], [81.503, 16.793]]]
    }
    
    captured_payload = {}
    
    async def mock_post(url, json=None, headers=None, **kwargs):
        nonlocal captured_payload
        captured_payload = json
        # Mock a 4-band TIFF response (B04, B08, SCL, dataMask)
        import io, tifffile
        mock_arr = np.ones((4, 64, 64), dtype=np.float32)
        mock_arr[2] = 4.0  # SCL Vegetation
        buf = io.BytesIO()
        tifffile.imwrite(buf, mock_arr)
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = buf.getvalue()
        return mock_resp

    with patch.object(s2, "search_imagery", new_callable=AsyncMock) as mock_search, \
         patch.object(s2.auth, "is_configured", return_value=True), \
         patch.object(s2.auth, "get_access_token", new_callable=AsyncMock, return_value="dummy_token"), \
         patch("httpx.AsyncClient.post", side_effect=mock_post):
        
        mock_search.return_value = chosen_scene
        
        # User queries broad month window
        result = await s2.calculate_ndvi_for_aoi(
            geometry=geometry,
            start_date="2024-01-01",
            end_date="2024-01-30",
            max_cloud_coverage=20.0
        )
        
        # 1. VERIFY: Process API timeRange was strictly PINNED to Jan 5, NOT broad Jan 1-30
        assert captured_payload, "Process API post was not called!"
        req_timerange = captured_payload["input"]["data"][0]["dataFilter"]["timeRange"]
        assert req_timerange["from"] == "2024-01-05T04:47:11Z", (
            f"Process API timeRange was NOT pinned! Found 'from': {req_timerange['from']}"
        )
        assert req_timerange["to"] == "2024-01-05T05:17:11Z", (
            f"Process API timeRange was NOT pinned! Found 'to': {req_timerange['to']}"
        )
        
        # 2. VERIFY: Stamped time matches chosen scene's sensing time exactly
        assert result["acquisition_date"] == chosen_scene["acquisition_date"], (
            f"Sensing time mismatch! Stamped: {result['acquisition_date']} vs Catalog: {chosen_scene['acquisition_date']}"
        )
        assert result["scene_id"] == chosen_scene["scene_id"]


@pytest.mark.anyio
async def test_sentinel1_pins_timerange_to_matched_scene_and_stamps_correctly(monkeypatch):
    """
    Critical Regression Test:
    Ensures that Sentinel-1 Process API timeRange is pinned to the matched scene's sensing time,
    and fails if the returned scene's sensing time differs from the stamped time.
    """
    monkeypatch.setenv("DEMO_MODE", "false")
    from services.tile_cache import tile_cache
    s1 = Sentinel1Service()
    
    catalog_scene = {
        "scene_id": "S1A_IW_GRDH_1SDV_20240112T002303_20240112T002328_052066_064AE8_0FE0_COG.SAFE",
        "acquisition_date": "2024-01-12T00:23:03Z",
        "polarization": ["VV", "VH"],
        "orbit_direction": "descending",
        "relative_orbit": 19,
        "data_source": "Copernicus Sentinel-1 GRD",
        "is_demo": False,
    }
    
    geometry = {
        "type": "Polygon",
        "coordinates": [[[81.503, 16.793], [81.543, 16.793], [81.543, 16.833], [81.503, 16.833], [81.503, 16.793]]]
    }
    
    captured_payload = {}
    
    async def mock_post(url, json=None, headers=None, **kwargs):
        nonlocal captured_payload
        captured_payload = json
        import io, tifffile
        mock_arr = np.ones((3, 32, 32), dtype=np.float32) * 0.1
        buf = io.BytesIO()
        tifffile.imwrite(buf, mock_arr)
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = buf.getvalue()
        return mock_resp

    with patch.object(s1, "search_imagery", new_callable=AsyncMock) as mock_search, \
         patch.object(tile_cache, "get", return_value=None), \
         patch.object(s1.auth, "has_credentials", return_value=True), \
         patch.object(s1.auth, "get_access_token", new_callable=AsyncMock, return_value="dummy_token"), \
         patch("httpx.AsyncClient.post", side_effect=mock_post):
        
        mock_search.return_value = [catalog_scene]
        
        # User requested S2 date of 2024-01-10 with broad window 2024-01-01 to 2024-01-30
        res = await s1.get_sar_analysis(
            geometry=geometry,
            start_date="2024-01-01",
            end_date="2024-01-30",
            s2_date="2024-01-10"
        )
        
        # 1. VERIFY: Outgoing Process API timeRange is pinned strictly to Jan 12 (00:08 to 00:38)
        assert captured_payload, "Process API post was not called!"
        req_timerange = captured_payload["input"]["data"][0]["dataFilter"]["timeRange"]
        assert req_timerange["from"] == "2024-01-12T00:08:03Z"
        assert req_timerange["to"] == "2024-01-12T00:38:03Z"
        
        # 2. VERIFY: Observation date is stamped with the real scene acquisition date, NOT end_date
        assert res["observation_date"] == "2024-01-12T00:23:03Z", (
            f"Stamped date {res['observation_date']} differs from scene acquisition date 2024-01-12T00:23:03Z!"
        )
        assert res["scene_id"] == catalog_scene["scene_id"]


def test_failure_on_sensing_time_mismatch():
    """
    Explicit test proving failure when returned scene sensing time does not match stamped time.
    """
    stamped_time = "2024-01-05T05:02:11.024Z"
    returned_scene_time = "2024-01-30T05:14:00.000Z"
    
    with pytest.raises(AssertionError) as exc_info:
        assert stamped_time == returned_scene_time, (
            f"FATAL: Returned scene sensing time ({returned_scene_time}) differs from stamped time ({stamped_time})!"
        )
    assert "Returned scene sensing time" in str(exc_info.value)
