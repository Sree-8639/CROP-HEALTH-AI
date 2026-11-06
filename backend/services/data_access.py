"""
backend/services/data_access.py:
Common Data-Access Layer shared across Crop, Deforestation, and Heat modules.

Provides unified, robust interfaces for:
- Copernicus Data Space Ecosystem (CDSE) authentication
- Sentinel-2 Level-2A multispectral observations & SCL cloud masking
- Sentinel-1 C-band SAR GRD radar backscatter
- Standardized Canonical AOI Grid & reprojection
- Multilevel tile caching (memory + disk)
- Transient error retry with exponential backoff
- Provenance tracking (Live vs. Demo, timestamps, sensor, quality)
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from services.copernicus_auth import (
    CopernicusAuthService,
    CopernicusAuthError,
    copernicus_auth,
)
from services.tile_cache import tile_cache, TileCache
from services.http_retry import request_with_retry
from services.provenance import build_provenance_badge
from services.raster_alignment import CanonicalGrid, create_canonical_aoi_grid
from services.sentinel2_service import sentinel2_service, Sentinel2Service
from services.sentinel1_service import sentinel1_service, Sentinel1Service


class CommonDataAccessLayer:
    """
    Unified Data-Access Layer shared by /api/crop, /api/forest, and /api/heat.
    Encapsulates raw satellite queries, authentication, caching, and spatial alignment.
    """

    def __init__(
        self,
        auth: Optional[CopernicusAuthService] = None,
        s2_svc: Optional[Sentinel2Service] = None,
        s1_svc: Optional[Sentinel1Service] = None,
        cache: Optional[TileCache] = None,
    ):
        self.auth = auth or copernicus_auth
        self.s2 = s2_svc or sentinel2_service
        self.s1 = s1_svc or sentinel1_service
        self.cache = cache or tile_cache

    @property
    def is_demo_mode(self) -> bool:
        demo_env = os.getenv("DEMO_MODE", "true").lower()
        return demo_env in ("1", "true", "yes")

    async def get_auth_token(self, force_refresh: bool = False) -> str:
        """Retrieves active Copernicus OAuth2 bearer token."""
        return await self.auth.get_access_token(force_refresh=force_refresh)

    def get_canonical_grid(
        self, geometry: Dict[str, Any], target_resolution_m: float = 10.0
    ) -> CanonicalGrid:
        """Derives metric UTM projection and bounding grid for the AOI."""
        return create_canonical_aoi_grid(geometry, target_resolution_m=target_resolution_m)

    async def search_sentinel2(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        max_cloud_coverage: float = 30.0,
    ) -> Dict[str, Any]:
        """Searches for Sentinel-2 Level-2A acquisitions over AOI."""
        return await self.s2.search_imagery(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
            max_cloud_coverage=max_cloud_coverage,
        )

    async def fetch_sentinel2_bands(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        band_names: Optional[List[str]] = None,
        max_cloud_coverage: float = 30.0,
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """
        Retrieves aligned Float32 Sentinel-2 reflectance arrays + SCL mask,
        backed by tile cache.
        """
        bands = band_names or ["B02", "B03", "B04", "B05", "B08", "B11"]
        return await self.s2._fetch_multispectral_bands(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
            max_cloud_coverage=max_cloud_coverage,
            band_names=bands,
        )

    async def search_sentinel1(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        orbit_direction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Queries Copernicus catalog for Sentinel-1 GRD acquisitions."""
        return await self.s1.search_imagery(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
            orbit_direction=orbit_direction,
        )

    async def fetch_sentinel1_bands(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        width: int = 128,
        height: int = 128,
        target_scene: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Retrieves calibrated Sentinel-1 linear backscatter bands (VV, VH)
        and valid mask, backed by tile cache.
        """
        return await self.s1.get_sar_bands(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
            width=width,
            height=height,
            target_scene=target_scene,
        )

    def generate_cache_key(
        self,
        prefix: str,
        geometry: Dict[str, Any],
        date_str: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generates deterministic SHA-256 tile cache key."""
        return self.cache.generate_cache_key(prefix, geometry, date_str, extra)

    def build_provenance(
        self,
        is_demo: bool,
        sensor: str,
        optical_date: Optional[str] = None,
        sar_date: Optional[str] = None,
        temporal_gap_days: Optional[int] = None,
        cloud_coverage_percentage: Optional[float] = None,
        valid_pixel_percentage: Optional[float] = None,
        scene_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Constructs standardized provenance badge."""
        return build_provenance_badge(
            is_demo=is_demo,
            sensor=sensor,
            optical_date=optical_date,
            sar_date=sar_date,
            temporal_gap_days=temporal_gap_days,
            cloud_coverage_percentage=cloud_coverage_percentage,
            valid_pixel_percentage=valid_pixel_percentage,
            scene_id=scene_id,
        )


# Global singleton data access layer
data_access_layer = CommonDataAccessLayer()

