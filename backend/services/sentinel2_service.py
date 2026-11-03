import os
import io
import base64
import datetime
from typing import Dict, Any, Optional, List, Tuple
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from services.copernicus_auth import (
    CopernicusAuthService,
    CopernicusCredentialsMissingError,
    CopernicusAuthError,
    copernicus_auth,
)

class Sentinel2Error(Exception):
    """Base exception for Sentinel-2 service operations."""
    pass

class NoImageryFoundError(Sentinel2Error):
    """Raised when no suitable Sentinel-2 imagery matches the criteria."""
    pass

class InvalidAOIError(Sentinel2Error):
    """Raised when the AOI geometry is invalid."""
    pass

class InvalidDateRangeError(Sentinel2Error):
    """Raised when date range parameters are invalid."""
    pass

class Sentinel2Service:
    """
    Service for querying Copernicus Data Space Ecosystem (CDSE) Sentinel Hub
    for Sentinel-2 Level-2A satellite imagery.
    """

    def __init__(self, auth_service: Optional[CopernicusAuthService] = None):
        self.auth = auth_service or copernicus_auth

    @property
    def base_url(self) -> str:
        return os.getenv("COPERNICUS_SH_BASE_URL", "https://sh.dataspace.copernicus.eu").rstrip("/")

    @property
    def is_demo_mode(self) -> bool:
        demo_env = os.getenv("DEMO_MODE", "true").lower()
        return demo_env in ("1", "true", "yes")

    def validate_inputs(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        max_cloud_coverage: float,
    ) -> None:
        """Validates incoming AOI geometry and date parameters."""
        if not geometry or not isinstance(geometry, dict):
            raise InvalidAOIError("AOI geometry must be a valid GeoJSON object.")

        geom_type = geometry.get("type")
        if geom_type not in ("Polygon", "MultiPolygon"):
            raise InvalidAOIError(f"Unsupported geometry type '{geom_type}'. Must be Polygon or MultiPolygon.")

        coords = geometry.get("coordinates")
        if not coords or not isinstance(coords, list):
            raise InvalidAOIError("Invalid geometry coordinates. Must contain a non-empty list of points.")

        # Validate date formats
        try:
            d_start = datetime.date.fromisoformat(start_date)
            d_end = datetime.date.fromisoformat(end_date)
        except (ValueError, TypeError):
            raise InvalidDateRangeError("Dates must be valid ISO 8601 strings in YYYY-MM-DD format.")

        if d_start > d_end:
            raise InvalidDateRangeError(f"Start date ({start_date}) cannot be after end date ({end_date}).")

        if d_start < datetime.date(2015, 6, 23):
            raise InvalidDateRangeError("Sentinel-2 data is only available from June 23, 2015 onwards.")

        if not (0.0 <= max_cloud_coverage <= 100.0):
            raise ValueError("Maximum cloud coverage must be between 0 and 100 percent.")

    async def search_imagery(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        max_cloud_coverage: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Searches Copernicus Sentinel Hub Catalog for suitable Sentinel-2 L2A scenes.
        """
        self.validate_inputs(geometry, start_date, end_date, max_cloud_coverage)

        # Check credentials and demo mode fallback
        if self.is_demo_mode:
            return self._generate_demo_search_result(start_date, end_date, max_cloud_coverage)

        if not self.auth.is_configured():
            raise CopernicusCredentialsMissingError(
                "Satellite service credentials are not configured. "
                "Please configure COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET."
            )

        token = await self.auth.get_access_token()
        catalog_url = f"{self.base_url}/api/v1/catalog/1.0.0/search"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/geo+json, application/json, */*",
        }

        payload = {
            "collections": ["sentinel-2-l2a"],
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            "intersects": geometry,
            "limit": 25,
        }

        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(catalog_url, json=payload, headers=headers)

                if response.status_code == 401:
                    # Token might have expired on server side; refresh once
                    self.auth.clear_cached_token()
                    token = await self.auth.get_access_token(force_refresh=True)
                    headers["Authorization"] = f"Bearer {token}"
                    response = await client.post(catalog_url, json=payload, headers=headers)

                if response.status_code in (401, 403):
                    raise CopernicusAuthError("Authentication with Sentinel Hub failed. Verify credentials.")

                if response.status_code == 429:
                    raise Sentinel2Error("Satellite service rate limit reached. Please wait a moment.")

                if response.status_code >= 400:
                    raise Sentinel2Error(
                        f"Satellite catalog search failed with status {response.status_code}."
                    )

                data = response.json()

        except httpx.TimeoutException:
            raise Sentinel2Error("Satellite catalog search timed out. Copernicus service may be busy.")
        except httpx.RequestError as exc:
            raise Sentinel2Error(f"Network error querying Copernicus catalog: {str(exc)}")

        features = data.get("features", [])
        if not features:
            raise NoImageryFoundError(
                "No suitable Sentinel-2 imagery was found for this area and date range."
            )

        # Filter scenes by user's maximum cloud coverage threshold
        valid_scenes = [
            f for f in features
            if f.get("properties", {}).get("eo:cloud_cover", 100.0) <= max_cloud_coverage
        ]

        if not valid_scenes:
            lowest_cloud = min(f.get("properties", {}).get("eo:cloud_cover", 100.0) for f in features)
            raise NoImageryFoundError(
                f"No suitable Sentinel-2 imagery found with cloud coverage ≤ {max_cloud_coverage}%. "
                f"Lowest cloud cover available in this range was {lowest_cloud:.1f}%."
            )

        # Pick the scene with the lowest cloud coverage
        sorted_scenes = sorted(
            valid_scenes,
            key=lambda x: x.get("properties", {}).get("eo:cloud_cover", 100.0)
        )
        best_scene = sorted_scenes[0]
        props = best_scene.get("properties", {})

        return {
            "status": "success",
            "scene_id": best_scene.get("id", "S2-UNKNOWN"),
            "acquisition_date": props.get("datetime", f"{end_date}T10:00:00Z"),
            "cloud_coverage": round(props.get("eo:cloud_cover", 0.0), 2),
            "data_source": "Copernicus Sentinel-2",
            "processing_level": "Level-2A",
            "bands": ["B04", "B03", "B02", "B08"],
            "date_range": {
                "start_date": start_date,
                "end_date": end_date,
            },
            "is_demo": False,
            "total_scenes_found": len(features),
        }

    async def get_preview_image(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        max_cloud_coverage: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Requests an actual RGB Sentinel-2 L2A image from Sentinel Hub Process API.
        """
        self.validate_inputs(geometry, start_date, end_date, max_cloud_coverage)

        # Demo mode check
        if self.is_demo_mode:
            return self._generate_demo_preview(start_date, end_date, max_cloud_coverage)

        if not self.auth.is_configured():
            raise CopernicusCredentialsMissingError(
                "Satellite service credentials are not configured. "
                "Please configure COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET."
            )

        # First verify metadata / scene availability
        scene_info = await self.search_imagery(geometry, start_date, end_date, max_cloud_coverage)

        token = await self.auth.get_access_token()
        # Copernicus Sentinel Hub Process API endpoint
        process_endpoint = os.getenv("COPERNICUS_PROCESS_ENDPOINT", "/api/v1/process")
        process_url = f"{self.base_url}{process_endpoint}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "image/png",
        }

        evalscript = """//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B08"],
    output: { bands: 3 }
  };
}

function evaluatePixel(sample) {
  // Gain factor of 2.5 is standard for Sentinel-2 true color RGB visualization
  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
}
"""

        payload = {
            "input": {
                "bounds": {
                    "geometry": geometry,
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": f"{start_date}T00:00:00Z",
                                "to": f"{end_date}T23:59:59Z",
                            },
                            "maxCloudCoverage": max_cloud_coverage,
                        },
                    }
                ],
            },
            "output": {
                "width": 512,
                "height": 512,
                "responses": [
                    {
                        "identifier": "default",
                        "format": {
                            "type": "image/png",
                        },
                    }
                ],
            },
            "evalscript": evalscript,
        }

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(process_url, json=payload, headers=headers)

                if response.status_code == 401:
                    # Token refresh attempt
                    self.auth.clear_cached_token()
                    token = await self.auth.get_access_token(force_refresh=True)
                    headers["Authorization"] = f"Bearer {token}"
                    response = await client.post(process_url, json=payload, headers=headers)

                if response.status_code in (401, 403):
                    raise CopernicusAuthError("Satellite processing failed: Unauthorized access.")

                if response.status_code == 429:
                    raise Sentinel2Error("Satellite processing rate limit reached. Please retry in a few moments.")

                if response.status_code >= 400:
                    err_text = response.text[:200]
                    raise Sentinel2Error(f"Satellite processing API error ({response.status_code}): {err_text}")

                image_bytes = response.content

        except httpx.TimeoutException:
            raise Sentinel2Error("Satellite image generation timed out. The Sentinel Hub service may be congested.")
        except httpx.RequestError as exc:
            raise Sentinel2Error(f"Network error requesting satellite image: {str(exc)}")

        base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:image/png;base64,{base64_encoded}"

        return {
            "status": "success",
            "image_base64": image_data_url,
            "scene_id": scene_info.get("scene_id", "S2-PROCESSED"),
            "acquisition_date": scene_info.get("acquisition_date", f"{end_date}T10:00:00Z"),
            "cloud_coverage": scene_info.get("cloud_coverage", 0.0),
            "data_source": "Copernicus Sentinel-2",
            "processing_level": "Level-2A",
            "bands": ["B04", "B03", "B02", "B08"],
            "is_demo": False,
            "metadata": {
                "date_range": {"start_date": start_date, "end_date": end_date},
                "max_cloud_coverage_requested": max_cloud_coverage,
                "width": 512,
                "height": 512,
                "format": "image/png",
            },
        }

    def _generate_demo_search_result(
        self,
        start_date: str,
        end_date: str,
        max_cloud_coverage: float,
    ) -> Dict[str, Any]:
        """Provides simulated search metadata for offline demonstration."""
        return {
            "status": "success",
            "scene_id": f"S2A_MSIL2A_DEMO_{end_date.replace('-', '')}T052021_N0500_R019_T44QND",
            "acquisition_date": f"{end_date}T05:20:21Z",
            "cloud_coverage": 4.8,
            "data_source": "Copernicus Sentinel-2",
            "processing_level": "Level-2A",
            "bands": ["B04", "B03", "B02", "B08"],
            "date_range": {
                "start_date": start_date,
                "end_date": end_date,
            },
            "is_demo": True,
            "total_scenes_found": 3,
        }

    def _generate_demo_preview(
        self,
        start_date: str,
        end_date: str,
        max_cloud_coverage: float,
    ) -> Dict[str, Any]:
        """
        Generates a realistic synthetic satellite image of agricultural farmland
        clearly stamped with 'DEMO DATA — NOT LIVE SATELLITE IMAGERY'.
        """
        width, height = 512, 512
        img = Image.new("RGB", (width, height), color=(40, 75, 45))
        draw = ImageDraw.Draw(img)

        # Draw realistic agricultural field parcels (grids of varied green/earth crops)
        field_colors = [
            (34, 110, 52),   # Dense healthy vegetation (crops)
            (48, 135, 65),   # Lush irrigated paddy/vegetation
            (76, 154, 72),   # High chlorophyll crop
            (95, 135, 60),   # Maturing crops
            (130, 115, 65),  # Fallow / tilled soil
            (115, 95, 55),   # Dry field / soil
            (40, 85, 45),    # Orchards / dense trees
            (28, 92, 44),    # Irrigated field
        ]

        import random
        # Seed by date for consistent rendering per query
        rng = random.Random(sum(ord(c) for c in f"{start_date}_{end_date}"))

        # Draw grid parcels
        grid_cols, grid_rows = 8, 8
        cell_w, cell_h = width // grid_cols, height // grid_rows

        for r in range(grid_rows):
            for c in range(grid_cols):
                color = rng.choice(field_colors)
                x1, y1 = c * cell_w, r * cell_h
                x2, y2 = (c + 1) * cell_w, (r + 1) * cell_h
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=(25, 45, 25), width=2)

        # Draw an irrigation canal / stream across fields
        canal_points = [
            (0, 140), (120, 160), (260, 150), (380, 190), (512, 210)
        ]
        draw.line(canal_points, fill=(35, 75, 115), width=7)

        # Add DEMO DATA banner to prevent confusion with live satellite imagery
        # Top banner overlay
        draw.rectangle([0, 0, width, 44], fill=(20, 20, 25))
        draw.rectangle([0, 44, width, 46], fill=(234, 179, 8)) # Amber line

        # Banner text
        banner_text = "DEMO DATA — NOT LIVE SATELLITE IMAGERY"
        draw.text((36, 14), banner_text, fill=(255, 255, 255))

        # Bottom info strip
        draw.rectangle([0, height - 32, width, height], fill=(15, 23, 42))
        info_text = f"Sentinel-2 L2A Sim | Bands: B04, B03, B02 | Acquisition: {end_date}"
        draw.text((20, height - 24), info_text, fill=(148, 163, 184))

        # Save to PNG buffer
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        base64_encoded = base64.b64encode(png_bytes).decode("utf-8")
        image_data_url = f"data:image/png;base64,{base64_encoded}"

        return {
            "status": "success",
            "image_base64": image_data_url,
            "scene_id": f"S2A_MSIL2A_DEMO_{end_date.replace('-', '')}T052021_N0500_R019_T44QND",
            "acquisition_date": f"{end_date}T05:20:21Z",
            "cloud_coverage": 4.8,
            "data_source": "Copernicus Sentinel-2",
            "processing_level": "Level-2A",
            "bands": ["B04", "B03", "B02", "B08"],
            "is_demo": True,
            "metadata": {
                "date_range": {"start_date": start_date, "end_date": end_date},
                "max_cloud_coverage_requested": max_cloud_coverage,
                "width": 512,
                "height": 512,
                "format": "image/png",
            },
        }

    async def calculate_ndvi_for_aoi(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        max_cloud_coverage: float = 30.0,
        aoi_area_km2: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Phase 4: Performs scientific NDVI analysis from Sentinel-2 B04 (Red) and B08 (NIR)
        spectral data for the designated AOI and date range.
        """
        import tifffile
        from services.ndvi_service import (
            calculate_ndvi,
            create_valid_pixel_mask,
            calculate_ndvi_statistics,
            generate_ndvi_colormap,
            generate_demo_spectral_bands,
        )

        self.validate_inputs(geometry, start_date, end_date, max_cloud_coverage)

        # Demo mode check
        if self.is_demo_mode:
            # Deterministic synthetic spectral bands
            b04, b08, scl, data_mask = generate_demo_spectral_bands(start_date, end_date)
            valid_mask = create_valid_pixel_mask(b04, b08, scl=scl, data_mask=data_mask)
            ndvi = calculate_ndvi(b04, b08, valid_mask=valid_mask)
            stats = calculate_ndvi_statistics(
                ndvi, valid_mask=valid_mask, pixel_size_m=10.0, total_aoi_area_km2=aoi_area_km2
            )
            ndvi_map = generate_ndvi_colormap(ndvi, valid_mask=valid_mask, is_demo=True)

            return {
                "status": "success",
                "scene_id": f"S2A_MSIL2A_DEMO_{end_date.replace('-', '')}T052021_N0500_R019_T44QND",
                "acquisition_date": f"{end_date}T05:20:21Z",
                "cloud_coverage": 4.8,
                "data_source": "Copernicus Sentinel-2",
                "processing_level": "Level-2A",
                "bands_used": ["B04 (Red)", "B08 (NIR)"],
                "valid_pixel_percentage": stats["valid_pixel_percentage"],
                "statistics": stats,
                "ndvi_map": ndvi_map,
                "is_demo": True,
                "date_range": {"start_date": start_date, "end_date": end_date},
            }

        if not self.auth.is_configured():
            raise CopernicusCredentialsMissingError(
                "Satellite service credentials are not configured. "
                "Please configure COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET."
            )

        # 1. Search for suitable Sentinel-2 scene
        scene_info = await self.search_imagery(geometry, start_date, end_date, max_cloud_coverage)

        # 2. Query Sentinel Hub Process API for raw Float32 B04, B08, SCL, dataMask
        token = await self.auth.get_access_token()
        process_endpoint = os.getenv("COPERNICUS_PROCESS_ENDPOINT", "/api/v1/process")
        process_url = f"{self.base_url}{process_endpoint}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "image/tiff",
        }

        evalscript = """//VERSION=3
function setup() {
  return {
    input: ["B04", "B08", "SCL", "dataMask"],
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
                "bounds": {
                    "geometry": geometry,
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": f"{start_date}T00:00:00Z",
                                "to": f"{end_date}T23:59:59Z",
                            },
                            "maxCloudCoverage": max_cloud_coverage,
                        },
                    }
                ],
            },
            "output": {
                "width": 256,
                "height": 256,
                "responses": [
                    {
                        "identifier": "default",
                        "format": {
                            "type": "image/tiff",
                        },
                    }
                ],
            },
            "evalscript": evalscript,
        }

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(process_url, json=payload, headers=headers)

                if response.status_code == 401:
                    self.auth.clear_cached_token()
                    token = await self.auth.get_access_token(force_refresh=True)
                    headers["Authorization"] = f"Bearer {token}"
                    response = await client.post(process_url, json=payload, headers=headers)

                if response.status_code in (401, 403):
                    raise CopernicusAuthError("Satellite processing failed: Unauthorized access.")

                if response.status_code == 429:
                    raise Sentinel2Error("Satellite processing rate limit reached. Please retry in a few moments.")

                if response.status_code >= 400:
                    err_text = response.text[:200]
                    raise Sentinel2Error(f"Sentinel-2 spectral API error ({response.status_code}): {err_text}")

                tiff_bytes = response.content

        except httpx.TimeoutException:
            raise Sentinel2Error("Multispectral data query timed out. The Sentinel Hub service may be busy.")
        except httpx.RequestError as exc:
            raise Sentinel2Error(f"Network error requesting spectral bands: {str(exc)}")

        # 3. Read and decompose multiband Float32 TIFF
        try:
            raw_tiff = tifffile.imread(io.BytesIO(tiff_bytes))
            if raw_tiff.ndim == 3:
                if raw_tiff.shape[0] == 4:
                    b04, b08, scl, data_mask = raw_tiff[0], raw_tiff[1], raw_tiff[2], raw_tiff[3]
                elif raw_tiff.shape[2] == 4:
                    b04, b08, scl, data_mask = raw_tiff[:, :, 0], raw_tiff[:, :, 1], raw_tiff[:, :, 2], raw_tiff[:, :, 3]
                else:
                    raise ValueError(f"Unexpected TIFF band count: shape={raw_tiff.shape}")
            elif raw_tiff.ndim == 2:
                raise ValueError("Expected multiband Float32 TIFF with B04, B08, SCL, dataMask.")
            else:
                raise ValueError(f"Unexpected TIFF dimensions: ndim={raw_tiff.ndim}")
        except Exception as exc:
            raise Sentinel2Error(f"Failed to decode multispectral TIFF data: {str(exc)}")

        # 4. Calculate NDVI and statistics
        valid_mask = create_valid_pixel_mask(b04, b08, scl=scl, data_mask=data_mask)
        ndvi = calculate_ndvi(b04, b08, valid_mask=valid_mask)
        stats = calculate_ndvi_statistics(
            ndvi, valid_mask=valid_mask, pixel_size_m=10.0, total_aoi_area_km2=aoi_area_km2
        )
        ndvi_map = generate_ndvi_colormap(ndvi, valid_mask=valid_mask, is_demo=False)

        return {
            "status": "success",
            "scene_id": scene_info.get("scene_id", "S2-PROCESSED"),
            "acquisition_date": scene_info.get("acquisition_date", f"{end_date}T10:00:00Z"),
            "cloud_coverage": scene_info.get("cloud_coverage", 0.0),
            "data_source": "Copernicus Sentinel-2",
            "processing_level": "Level-2A",
            "bands_used": ["B04 (Red)", "B08 (NIR)"],
            "valid_pixel_percentage": stats["valid_pixel_percentage"],
            "statistics": stats,
            "ndvi_map": ndvi_map,
            "is_demo": False,
            "date_range": {"start_date": start_date, "end_date": end_date},
        }

    async def _fetch_multispectral_bands(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        max_cloud_coverage: float,
        band_names: List[str],
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """
        Retrieves raw Sentinel-2 Level-2A Float32 spectral band arrays from Sentinel Hub Process API
        or generates deterministic synthetic bands in DEMO_MODE.
        """
        import tifffile
        from services.ndvi_service import generate_all_demo_spectral_bands
        from services.raster_utils import align_bands

        self.validate_inputs(geometry, start_date, end_date, max_cloud_coverage)

        if self.is_demo_mode:
            raw_bands = generate_all_demo_spectral_bands(start_date, end_date)
            scene_info = {
                "scene_id": f"S2A_MSIL2A_DEMO_{end_date.replace('-', '')}T052021_N0500_R019_T44QND",
                "acquisition_date": f"{end_date}T05:20:21Z",
                "cloud_coverage": 4.8,
                "data_source": "Copernicus Sentinel-2",
                "processing_level": "Level-2A",
                "is_demo": True,
            }
            aligned = align_bands(raw_bands, reference_band="B08")
            return scene_info, aligned

        if not self.auth.is_configured():
            raise CopernicusCredentialsMissingError(
                "Satellite service credentials are not configured. "
                "Please configure COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET."
            )

        # 1. Search scene in catalog
        scene_info = await self.search_imagery(geometry, start_date, end_date, max_cloud_coverage)

        # 2. Build Process API request
        token = await self.auth.get_access_token()
        process_endpoint = os.getenv("COPERNICUS_PROCESS_ENDPOINT", "/api/v1/process")
        process_url = f"{self.base_url}{process_endpoint}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "image/tiff",
        }

        # Dynamically build evalscript for requested bands + SCL + dataMask
        all_req_bands = list(dict.fromkeys(band_names + ["SCL", "dataMask"]))
        evalscript = f"""//VERSION=3
function setup() {{
  return {{
    input: {all_req_bands},
    output: {{
      bands: {len(all_req_bands)},
      sampleType: "FLOAT32"
    }}
  }};
}}

function evaluatePixel(sample) {{
  return [{", ".join([f"sample.{b}" for b in all_req_bands])}];
}}
"""

        payload = {
            "input": {
                "bounds": {"geometry": geometry},
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": f"{start_date}T00:00:00Z",
                                "to": f"{end_date}T23:59:59Z",
                            },
                            "maxCloudCoverage": max_cloud_coverage,
                        },
                    }
                ],
            },
            "output": {
                "width": 256,
                "height": 256,
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
            },
            "evalscript": evalscript,
        }

        try:
            async with httpx.AsyncClient(timeout=50.0) as client:
                response = await client.post(process_url, json=payload, headers=headers)
                if response.status_code == 401:
                    self.auth.clear_cached_token()
                    token = await self.auth.get_access_token(force_refresh=True)
                    headers["Authorization"] = f"Bearer {token}"
                    response = await client.post(process_url, json=payload, headers=headers)

                if response.status_code in (401, 403):
                    raise CopernicusAuthError("Satellite processing failed: Unauthorized access.")
                if response.status_code == 429:
                    raise Sentinel2Error("Satellite processing rate limit reached. Please wait a moment.")
                if response.status_code >= 400:
                    raise Sentinel2Error(f"Sentinel-2 multispectral API error ({response.status_code})")

                tiff_bytes = response.content
        except httpx.TimeoutException:
            raise Sentinel2Error("Multispectral query timed out. Sentinel Hub service may be congested.")
        except httpx.RequestError as exc:
            raise Sentinel2Error(f"Network error requesting spectral bands: {str(exc)}")

        raw_tiff = tifffile.imread(io.BytesIO(tiff_bytes))
        band_dict: Dict[str, np.ndarray] = {}

        if raw_tiff.ndim == 3 and raw_tiff.shape[0] == len(all_req_bands):
            for i, b in enumerate(all_req_bands):
                band_dict[b] = raw_tiff[i]
        elif raw_tiff.ndim == 3 and raw_tiff.shape[2] == len(all_req_bands):
            for i, b in enumerate(all_req_bands):
                band_dict[b] = raw_tiff[:, :, i]
        else:
            raise Sentinel2Error(f"Unexpected TIFF dimensions from Process API: shape={raw_tiff.shape}")

        aligned = align_bands(band_dict, reference_band="B08")
        return scene_info, aligned

    async def calculate_ndmi_for_aoi(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        max_cloud_coverage: float = 30.0,
        aoi_area_km2: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Phase 5: Calculates NDMI = (B08 - B11) / (B08 + B11) moisture-related spectral indicator.
        """
        from services.ndvi_service import create_valid_pixel_mask
        from services.ndmi_service import (
            calculate_ndmi,
            calculate_ndmi_statistics,
            generate_ndmi_colormap,
        )

        scene_info, bands = await self._fetch_multispectral_bands(
            geometry, start_date, end_date, max_cloud_coverage, ["B08", "B11"]
        )

        valid_mask = create_valid_pixel_mask(
            bands.get("B04", bands["B08"]),
            bands["B08"],
            scl=bands.get("SCL"),
            data_mask=bands.get("dataMask"),
        )
        ndmi = calculate_ndmi(bands["B08"], bands["B11"], valid_mask=valid_mask)
        stats = calculate_ndmi_statistics(
            ndmi, valid_mask=valid_mask, pixel_size_m=10.0, total_aoi_area_km2=aoi_area_km2
        )
        ndmi_map = generate_ndmi_colormap(ndmi, valid_mask=valid_mask, is_demo=scene_info.get("is_demo", False))

        return {
            "status": "success",
            "index_name": "NDMI",
            "indicator_label": "Vegetation Moisture-Related Indicator",
            "scene_id": scene_info.get("scene_id", "S2-PROCESSED"),
            "acquisition_date": scene_info.get("acquisition_date", f"{end_date}T10:00:00Z"),
            "cloud_coverage": scene_info.get("cloud_coverage", 0.0),
            "data_source": "Copernicus Sentinel-2",
            "processing_level": "Level-2A",
            "bands_used": ["B08 (NIR)", "B11 (SWIR)"],
            "valid_pixel_percentage": stats["valid_pixel_percentage"],
            "statistics": stats,
            "ndmi_map": ndmi_map,
            "is_demo": scene_info.get("is_demo", False),
            "date_range": {"start_date": start_date, "end_date": end_date},
        }

    async def calculate_ndre_for_aoi(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        max_cloud_coverage: float = 30.0,
        aoi_area_km2: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Phase 5: Calculates NDRE = (B08 - B05) / (B08 + B05) red-edge vegetation indicator.
        """
        from services.ndvi_service import create_valid_pixel_mask
        from services.ndre_service import (
            calculate_ndre,
            calculate_ndre_statistics,
            generate_ndre_colormap,
        )

        scene_info, bands = await self._fetch_multispectral_bands(
            geometry, start_date, end_date, max_cloud_coverage, ["B08", "B05"]
        )

        valid_mask = create_valid_pixel_mask(
            bands.get("B04", bands["B08"]),
            bands["B08"],
            scl=bands.get("SCL"),
            data_mask=bands.get("dataMask"),
        )
        ndre = calculate_ndre(bands["B08"], bands["B05"], valid_mask=valid_mask)
        stats = calculate_ndre_statistics(
            ndre, valid_mask=valid_mask, pixel_size_m=10.0, total_aoi_area_km2=aoi_area_km2
        )
        ndre_map = generate_ndre_colormap(ndre, valid_mask=valid_mask, is_demo=scene_info.get("is_demo", False))

        return {
            "status": "success",
            "index_name": "NDRE",
            "indicator_label": "Red-Edge Vegetation Indicator",
            "scene_id": scene_info.get("scene_id", "S2-PROCESSED"),
            "acquisition_date": scene_info.get("acquisition_date", f"{end_date}T10:00:00Z"),
            "cloud_coverage": scene_info.get("cloud_coverage", 0.0),
            "data_source": "Copernicus Sentinel-2",
            "processing_level": "Level-2A",
            "bands_used": ["B08 (NIR)", "B05 (RedEdge)"],
            "valid_pixel_percentage": stats["valid_pixel_percentage"],
            "statistics": stats,
            "ndre_map": ndre_map,
            "is_demo": scene_info.get("is_demo", False),
            "date_range": {"start_date": start_date, "end_date": end_date},
        }

    async def calculate_evi_for_aoi(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        max_cloud_coverage: float = 30.0,
        aoi_area_km2: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Phase 5: Calculates EVI = 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)
        enhanced vegetation index.
        """
        from services.ndvi_service import create_valid_pixel_mask
        from services.evi_service import (
            calculate_evi,
            calculate_evi_statistics,
            generate_evi_colormap,
        )

        scene_info, bands = await self._fetch_multispectral_bands(
            geometry, start_date, end_date, max_cloud_coverage, ["B02", "B04", "B08"]
        )

        valid_mask = create_valid_pixel_mask(
            bands["B04"],
            bands["B08"],
            scl=bands.get("SCL"),
            data_mask=bands.get("dataMask"),
        )
        evi = calculate_evi(bands["B02"], bands["B04"], bands["B08"], valid_mask=valid_mask)
        stats = calculate_evi_statistics(
            evi, valid_mask=valid_mask, pixel_size_m=10.0, total_aoi_area_km2=aoi_area_km2
        )
        evi_map = generate_evi_colormap(evi, valid_mask=valid_mask, is_demo=scene_info.get("is_demo", False))

        return {
            "status": "success",
            "index_name": "EVI",
            "indicator_label": "Enhanced Vegetation Index",
            "scene_id": scene_info.get("scene_id", "S2-PROCESSED"),
            "acquisition_date": scene_info.get("acquisition_date", f"{end_date}T10:00:00Z"),
            "cloud_coverage": scene_info.get("cloud_coverage", 0.0),
            "data_source": "Copernicus Sentinel-2",
            "processing_level": "Level-2A",
            "bands_used": ["B02 (Blue)", "B04 (Red)", "B08 (NIR)"],
            "valid_pixel_percentage": stats["valid_pixel_percentage"],
            "statistics": stats,
            "evi_map": evi_map,
            "is_demo": scene_info.get("is_demo", False),
            "date_range": {"start_date": start_date, "end_date": end_date},
        }

    async def calculate_all_indices_for_aoi(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        max_cloud_coverage: float = 30.0,
        aoi_area_km2: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Phase 5 Combined Endpoint: Computes NDVI, NDMI, NDRE, and EVI concurrently
        from a single unified spectral band acquisition pass (B02, B04, B05, B08, B11, SCL, dataMask).
        """
        from services.ndvi_service import (
            calculate_ndvi,
            create_valid_pixel_mask,
            calculate_ndvi_statistics,
            generate_ndvi_colormap,
        )
        from services.ndmi_service import (
            calculate_ndmi,
            calculate_ndmi_statistics,
            generate_ndmi_colormap,
        )
        from services.ndre_service import (
            calculate_ndre,
            calculate_ndre_statistics,
            generate_ndre_colormap,
        )
        from services.evi_service import (
            calculate_evi,
            calculate_evi_statistics,
            generate_evi_colormap,
        )

        all_req_bands = ["B02", "B04", "B05", "B08", "B11"]
        scene_info, bands = await self._fetch_multispectral_bands(
            geometry, start_date, end_date, max_cloud_coverage, all_req_bands
        )

        is_demo = scene_info.get("is_demo", False)
        valid_mask = create_valid_pixel_mask(
            bands["B04"],
            bands["B08"],
            scl=bands.get("SCL"),
            data_mask=bands.get("dataMask"),
        )

        # 1. NDVI (B08, B04)
        ndvi = calculate_ndvi(bands["B04"], bands["B08"], valid_mask=valid_mask)
        ndvi_stats = calculate_ndvi_statistics(
            ndvi, valid_mask=valid_mask, pixel_size_m=10.0, total_aoi_area_km2=aoi_area_km2
        )
        ndvi_map = generate_ndvi_colormap(ndvi, valid_mask=valid_mask, is_demo=is_demo)

        # 2. NDMI (B08, B11)
        ndmi = calculate_ndmi(bands["B08"], bands["B11"], valid_mask=valid_mask)
        ndmi_stats = calculate_ndmi_statistics(
            ndmi, valid_mask=valid_mask, pixel_size_m=10.0, total_aoi_area_km2=aoi_area_km2
        )
        ndmi_map = generate_ndmi_colormap(ndmi, valid_mask=valid_mask, is_demo=is_demo)

        # 3. NDRE (B08, B05)
        ndre = calculate_ndre(bands["B08"], bands["B05"], valid_mask=valid_mask)
        ndre_stats = calculate_ndre_statistics(
            ndre, valid_mask=valid_mask, pixel_size_m=10.0, total_aoi_area_km2=aoi_area_km2
        )
        ndre_map = generate_ndre_colormap(ndre, valid_mask=valid_mask, is_demo=is_demo)

        # 4. EVI (B02, B04, B08)
        evi = calculate_evi(bands["B02"], bands["B04"], bands["B08"], valid_mask=valid_mask)
        evi_stats = calculate_evi_statistics(
            evi, valid_mask=valid_mask, pixel_size_m=10.0, total_aoi_area_km2=aoi_area_km2
        )
        evi_map = generate_evi_colormap(evi, valid_mask=valid_mask, is_demo=is_demo)

        return {
            "status": "success",
            "scene": {
                "scene_id": scene_info.get("scene_id", "S2-PROCESSED"),
                "acquisition_date": scene_info.get("acquisition_date", f"{end_date}T10:00:00Z"),
                "cloud_coverage": scene_info.get("cloud_coverage", 0.0),
                "data_source": "Copernicus Sentinel-2",
                "processing_level": "Level-2A",
                "is_demo": is_demo,
                "date_range": {"start_date": start_date, "end_date": end_date},
            },
            "ndvi": {
                "name": "NDVI",
                "description": "Normalized Difference Vegetation Index (Vegetation Vigor Indicator)",
                "bands_used": ["B04 (Red)", "B08 (NIR)"],
                "statistics": ndvi_stats,
                "map": ndvi_map,
            },
            "ndmi": {
                "name": "NDMI",
                "description": "Normalized Difference Moisture Index (Vegetation Moisture Indicator)",
                "bands_used": ["B08 (NIR)", "B11 (SWIR)"],
                "statistics": ndmi_stats,
                "map": ndmi_map,
            },
            "ndre": {
                "name": "NDRE",
                "description": "Normalized Difference Red Edge Index (Red-Edge Vegetation Indicator)",
                "bands_used": ["B08 (NIR)", "B05 (RedEdge)"],
                "statistics": ndre_stats,
                "map": ndre_map,
            },
            "evi": {
                "name": "EVI",
                "description": "Enhanced Vegetation Index (Atmosphere & Soil-Decoupled Indicator)",
                "bands_used": ["B02 (Blue)", "B04 (Red)", "B08 (NIR)"],
                "statistics": evi_stats,
                "map": evi_map,
            },
        }

    async def calculate_crop_condition_for_aoi(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        max_cloud_coverage: float = 30.0,
        aoi_area_km2: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Phase 6: Crop Condition Intelligence combining NDVI + NDMI + NDRE + EVI + data quality
        and spatial stress pattern analysis into a normalized vegetation condition evaluation.
        """
        from services.ndvi_service import (
            calculate_ndvi,
            create_valid_pixel_mask,
            calculate_ndvi_statistics,
            generate_ndvi_colormap,
        )
        from services.ndmi_service import (
            calculate_ndmi,
            calculate_ndmi_statistics,
            generate_ndmi_colormap,
        )
        from services.ndre_service import (
            calculate_ndre,
            calculate_ndre_statistics,
            generate_ndre_colormap,
        )
        from services.evi_service import (
            calculate_evi,
            calculate_evi_statistics,
            generate_evi_colormap,
        )
        from services.crop_condition_service import (
            compute_crop_condition,
            generate_condition_colormap,
        )

        all_req_bands = ["B02", "B04", "B05", "B08", "B11"]
        scene_info, bands = await self._fetch_multispectral_bands(
            geometry, start_date, end_date, max_cloud_coverage, all_req_bands
        )

        is_demo = scene_info.get("is_demo", False)
        valid_mask = create_valid_pixel_mask(
            bands["B04"],
            bands["B08"],
            scl=bands.get("SCL"),
            data_mask=bands.get("dataMask"),
        )

        # 1. Calculate the 4 vegetation indices
        ndvi = calculate_ndvi(bands["B04"], bands["B08"], valid_mask=valid_mask)
        ndvi_stats = calculate_ndvi_statistics(
            ndvi, valid_mask=valid_mask, pixel_size_m=10.0, total_aoi_area_km2=aoi_area_km2
        )
        ndvi_map = generate_ndvi_colormap(ndvi, valid_mask=valid_mask, is_demo=is_demo)

        ndmi = calculate_ndmi(bands["B08"], bands["B11"], valid_mask=valid_mask)
        ndmi_stats = calculate_ndmi_statistics(
            ndmi, valid_mask=valid_mask, pixel_size_m=10.0, total_aoi_area_km2=aoi_area_km2
        )
        ndmi_map = generate_ndmi_colormap(ndmi, valid_mask=valid_mask, is_demo=is_demo)

        ndre = calculate_ndre(bands["B08"], bands["B05"], valid_mask=valid_mask)
        ndre_stats = calculate_ndre_statistics(
            ndre, valid_mask=valid_mask, pixel_size_m=10.0, total_aoi_area_km2=aoi_area_km2
        )
        ndre_map = generate_ndre_colormap(ndre, valid_mask=valid_mask, is_demo=is_demo)

        evi = calculate_evi(bands["B02"], bands["B04"], bands["B08"], valid_mask=valid_mask)
        evi_stats = calculate_evi_statistics(
            evi, valid_mask=valid_mask, pixel_size_m=10.0, total_aoi_area_km2=aoi_area_km2
        )
        evi_map = generate_evi_colormap(evi, valid_mask=valid_mask, is_demo=is_demo)

        # 2. Compute Crop Condition Intelligence
        cloud_cov = float(scene_info.get("cloud_coverage", 0.0))
        condition_res = compute_crop_condition(
            ndvi=ndvi,
            ndmi=ndmi,
            ndre=ndre,
            evi=evi,
            valid_mask=valid_mask,
            config=config,
            pixel_size_m=10.0,
            total_aoi_area_km2=aoi_area_km2,
            cloud_coverage=cloud_cov,
        )

        condition_map = generate_condition_colormap(
            condition_res["classified_raster"],
            valid_mask=valid_mask,
            is_demo=is_demo,
        )

        return {
            "status": "success",
            "overall_condition": condition_res["overall_condition"],
            "condition_score": condition_res["condition_score"],
            "condition_score_mean": condition_res["condition_score_mean"],
            "condition_score_median": condition_res["condition_score_median"],
            "condition_score_std": condition_res["condition_score_std"],
            "dominant_zone": condition_res["dominant_zone"],
            "valid_pixel_percentage": condition_res["valid_pixel_percentage"],
            "area_represented_km2": condition_res["area_represented_km2"],
            "zones": condition_res["zones"],
            "zone_counts": condition_res["zone_counts"],
            "indices": {
                "ndvi": {
                    "name": "NDVI",
                    "description": "Normalized Difference Vegetation Index (Vegetation Vigor Indicator)",
                    "physical_indicator": "Vegetation vigor and chlorophyll absorption",
                    "bands_used": ["B04 (Red)", "B08 (NIR)"],
                    "statistics": ndvi_stats,
                    "map": ndvi_map,
                },
                "ndmi": {
                    "name": "NDMI",
                    "description": "Normalized Difference Moisture Index (Vegetation Moisture Indicator)",
                    "physical_indicator": "Canopy moisture-related spectral signal",
                    "bands_used": ["B08 (NIR)", "B11 (SWIR)"],
                    "statistics": ndmi_stats,
                    "map": ndmi_map,
                },
                "ndre": {
                    "name": "NDRE",
                    "description": "Normalized Difference Red Edge Index (Red-Edge Vegetation Indicator)",
                    "physical_indicator": "Red-edge chlorophyll transition response",
                    "bands_used": ["B08 (NIR)", "B05 (RedEdge)"],
                    "statistics": ndre_stats,
                    "map": ndre_map,
                },
                "evi": {
                    "name": "EVI",
                    "description": "Enhanced Vegetation Index (Soil & Atmospheric Decoupled Indicator)",
                    "physical_indicator": "Enhanced canopy biomass signal",
                    "bands_used": ["B02 (Blue)", "B04 (Red)", "B08 (NIR)"],
                    "statistics": evi_stats,
                    "map": evi_map,
                },
            },
            "patterns": condition_res["patterns"],
            "quality": condition_res["quality"],
            "warnings": condition_res["warnings"],
            "condition_map": condition_map,
            "scene": {
                "scene_id": scene_info.get("scene_id", "S2-PROCESSED"),
                "acquisition_date": scene_info.get("acquisition_date", f"{end_date}T10:00:00Z"),
                "cloud_coverage": cloud_cov,
                "data_source": "Copernicus Sentinel-2",
                "processing_level": "Level-2A",
                "is_demo": is_demo,
                "date_range": {"start_date": start_date, "end_date": end_date},
            },
            "analysis_metadata": condition_res["analysis_metadata"],
        }

# Global service instance
sentinel2_service = Sentinel2Service()


