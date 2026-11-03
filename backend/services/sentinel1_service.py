import os
import io
import math
import base64
import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
import httpx
import numpy as np
from PIL import Image

from services.copernicus_auth import (
    CopernicusAuthService,
    CopernicusCredentialsMissingError,
    CopernicusAuthError,
    copernicus_auth,
)
from services.raster_utils import resample_band, align_bands, validate_band_shapes, RasterAlignmentError

class Sentinel1Error(Exception):
    """Base exception for Sentinel-1 SAR service operations."""
    pass

class NoSARDataFoundError(Sentinel1Error):
    """Raised when no suitable Sentinel-1 SAR acquisition matches the query criteria."""
    pass

class InvalidSARGeometryError(Sentinel1Error):
    """Raised when the AOI geometry is invalid for SAR processing."""
    pass

class InvalidSARDatesError(Sentinel1Error):
    """Raised when temporal search window parameters are invalid."""
    pass

class Sentinel1Service:
    """
    Service for querying Copernicus Data Space Ecosystem (CDSE) / Sentinel Hub
    for Sentinel-1 C-band Synthetic Aperture Radar (SAR) GRD observations,
    extracting VV and VH backscatter, computing VV/VH cross-ratios,
    performing temporal alignment with Sentinel-2 optical data, and generating map overlays.
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
    ) -> None:
        """Validates incoming AOI geometry and date parameters."""
        if not geometry or not isinstance(geometry, dict):
            raise InvalidSARGeometryError("AOI geometry must be a valid GeoJSON object.")

        geom_type = geometry.get("type")
        if geom_type not in ("Polygon", "MultiPolygon"):
            raise InvalidSARGeometryError(f"Unsupported geometry type '{geom_type}'. Must be Polygon or MultiPolygon.")

        coords = geometry.get("coordinates")
        if not coords or not isinstance(coords, list):
            raise InvalidSARGeometryError("Invalid geometry coordinates. Must contain a non-empty list of points.")

        try:
            start_dt = datetime.date.fromisoformat(start_date)
            end_dt = datetime.date.fromisoformat(end_date)
        except (ValueError, TypeError) as exc:
            raise InvalidSARDatesError(f"Malformed date string. Expected YYYY-MM-DD: {str(exc)}")

        if start_dt > end_dt:
            raise InvalidSARDatesError(
                f"Start date ({start_date}) cannot be later than end date ({end_date})."
            )

    async def search_imagery(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        orbit_direction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Queries Copernicus Catalog for available Sentinel-1 GRD acquisitions over the AOI.
        Falls back deterministically to realistic SAR metadata in DEMO_MODE.
        """
        self.validate_inputs(geometry, start_date, end_date)

        if self.is_demo_mode or not self.auth.has_credentials():
            return self._generate_demo_sar_catalog(geometry, start_date, end_date)

        token = await self.auth.get_access_token()
        catalog_url = f"{self.base_url}/api/v1/catalog/1.0.0/search"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/geo+json, application/json, */*",
        }


        # Query Copernicus for Sentinel-1 GRD
        payload: Dict[str, Any] = {
            "collections": ["sentinel-1-grd"],
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            "intersects": geometry,
            "limit": 20,
        }
        if orbit_direction in ("ascending", "descending"):
            payload["query"] = {"sat:orbit_state": {"eq": orbit_direction}}

        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(catalog_url, json=payload, headers=headers)
                if response.status_code == 401:
                    self.auth.clear_cached_token()
                    token = await self.auth.get_access_token(force_refresh=True)
                    headers["Authorization"] = f"Bearer {token}"
                    response = await client.post(catalog_url, json=payload, headers=headers)

                if response.status_code != 200:
                    # In case of remote API hiccup, return demo catalog if permitted
                    if self.is_demo_mode:
                        return self._generate_demo_sar_catalog(geometry, start_date, end_date)
                    raise Sentinel1Error(f"Sentinel-1 catalog search failed with status {response.status_code}.")

                data = response.json()
                features = data.get("features", [])
                if not features:
                    raise NoSARDataFoundError(
                        f"No Sentinel-1 SAR observations found for the AOI between {start_date} and {end_date}."
                    )

                results = []
                for f in features:
                    props = f.get("properties", {})
                    results.append({
                        "scene_id": f.get("id", "S1-UNKNOWN"),
                        "acquisition_date": props.get("datetime", f"{start_date}T00:00:00Z"),
                        "product_type": "Sentinel-1 GRD",
                        "polarization": props.get("sar:polarizations", ["VV", "VH"]),
                        "orbit_direction": props.get("sat:orbit_state", "descending"),
                        "instrument_mode": props.get("sar:instrument_mode", "IW"),
                        "resolution_m": 10,
                        "data_source": "Copernicus Sentinel-1",
                        "is_demo": False,
                    })
                return results

        except (httpx.RequestError, httpx.TimeoutException) as exc:
            if self.is_demo_mode:
                return self._generate_demo_sar_catalog(geometry, start_date, end_date)
            raise Sentinel1Error(f"Network error querying Sentinel-1 catalog: {str(exc)}")

    def _generate_demo_sar_catalog(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
    ) -> List[Dict[str, Any]]:
        """Generates realistic, deterministic Sentinel-1 acquisitions across the date range."""
        start_dt = datetime.date.fromisoformat(start_date)
        end_dt = datetime.date.fromisoformat(end_date)
        total_days = max(1, (end_dt - start_dt).days)

        # Sentinel-1 revisit is typically 6-12 days over land
        step_days = max(6, min(12, total_days // 4 if total_days >= 24 else 6))
        results = []
        curr = start_dt + datetime.timedelta(days=min(2, total_days // 2))

        idx = 1
        while curr <= end_dt:
            date_str = curr.isoformat()
            orbit = "descending" if idx % 2 == 0 else "ascending"
            scene_id = f"S1A_IW_GRDH_1SDV_{curr.strftime('%Y%m%d')}T003815_DEMO_{idx:03d}"
            results.append({
                "scene_id": scene_id,
                "acquisition_date": f"{date_str}T00:38:15Z",
                "product_type": "Sentinel-1 GRD",
                "polarization": ["VV", "VH"],
                "orbit_direction": orbit,
                "instrument_mode": "IW",
                "resolution_m": 10,
                "data_source": "Copernicus Sentinel-1 (Simulated Demo)",
                "is_demo": True,
            })
            curr += datetime.timedelta(days=step_days)
            idx += 1

        if not results:
            # At least one observation
            results.append({
                "scene_id": f"S1A_IW_GRDH_1SDV_{end_date.replace('-', '')}T003815_DEMO_001",
                "acquisition_date": f"{end_date}T00:38:15Z",
                "product_type": "Sentinel-1 GRD",
                "polarization": ["VV", "VH"],
                "orbit_direction": "descending",
                "instrument_mode": "IW",
                "resolution_m": 10,
                "data_source": "Copernicus Sentinel-1 (Simulated Demo)",
                "is_demo": True,
            })
        else:
            last_date_str = results[-1]["acquisition_date"][:10]
            last_dt = datetime.date.fromisoformat(last_date_str)
            if (end_dt - last_dt).days > 2:
                near_end = max(start_dt, end_dt - datetime.timedelta(days=1))
                results.append({
                    "scene_id": f"S1A_IW_GRDH_1SDV_{near_end.strftime('%Y%m%d')}T003815_DEMO_{idx:03d}",
                    "acquisition_date": f"{near_end.isoformat()}T00:38:15Z",
                    "product_type": "Sentinel-1 GRD",
                    "polarization": ["VV", "VH"],
                    "orbit_direction": "descending",
                    "instrument_mode": "IW",
                    "resolution_m": 10,
                    "data_source": "Copernicus Sentinel-1 (Simulated Demo)",
                    "is_demo": True,
                })
        return results

    def find_closest_sentinel1_observation(
        self,
        s2_date_str: str,
        s1_catalog: List[Dict[str, Any]],
        max_window_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Temporal Matching:
        Identifies the closest Sentinel-1 observation to a given Sentinel-2 date.
        If no observation falls within max_window_days, returns an explicit unavailable status
        without fabricating missing observations.
        """
        if not s1_catalog:
            return {
                "matched": False,
                "sentinel2_date": s2_date_str,
                "sentinel1_date": None,
                "date_difference_days": None,
                "message": "Sentinel-1 observation unavailable for this time window.",
            }

        s2_date = datetime.date.fromisoformat(s2_date_str.split("T")[0])
        best_match = None
        min_diff = 999999

        for item in s1_catalog:
            raw_date = item.get("acquisition_date", "")
            if not raw_date:
                continue
            s1_date = datetime.date.fromisoformat(raw_date.split("T")[0])
            diff = (s1_date - s2_date).days
            abs_diff = abs(diff)

            if abs_diff <= max_window_days and abs_diff < min_diff:
                min_diff = abs_diff
                best_match = (item, diff, s1_date.isoformat())

        if best_match is None:
            return {
                "matched": False,
                "sentinel2_date": s2_date_str,
                "sentinel1_date": None,
                "date_difference_days": None,
                "message": f"Sentinel-1 observation unavailable within ±{max_window_days} days of Sentinel-2 observation.",
            }

        scene, diff_days, s1_date_iso = best_match
        return {
            "matched": True,
            "sentinel2_date": s2_date_str.split("T")[0],
            "sentinel1_date": s1_date_iso,
            "date_difference_days": diff_days,
            "scene_id": scene.get("scene_id"),
            "polarization": scene.get("polarization", ["VV", "VH"]),
            "orbit_direction": scene.get("orbit_direction", "descending"),
            "data_source": scene.get("data_source", "Copernicus Sentinel-1"),
            "is_demo": scene.get("is_demo", False),
            "message": f"Matched Sentinel-1 acquisition ({abs(diff_days)} day{'s' if abs(diff_days) != 1 else ''} offset).",
        }

    async def get_sar_bands(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        width: int = 128,
        height: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Retrieves calibrated Sentinel-1 GRD linear backscatter bands (VV, VH)
        and the valid pixel mask.
        Returns:
            (vv_linear, vh_linear, valid_mask, scene_metadata)
        """
        self.validate_inputs(geometry, start_date, end_date)

        if self.is_demo_mode or not self.auth.has_credentials():
            return self._generate_deterministic_sar_data(geometry, start_date, end_date, width, height)

        token = await self.auth.get_access_token()
        process_url = f"{self.base_url}/api/v1/process"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "image/tiff",
        }

        # Evalscript retrieving orthorectified Gamma0/Sigma0 VV and VH
        evalscript = """//VERSION=3
function setup() {
  return {
    input: ["VV", "VH", "dataMask"],
    output: { bands: 3, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(samples) {
  return [samples.VV, samples.VH, samples.dataMask];
}"""

        payload = {
            "input": {
                "bounds": {
                    "geometry": geometry,
                    "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}
                },
                "data": [
                    {
                        "type": "sentinel-1-grd",
                        "dataFilter": {
                            "timeRange": {
                                "from": f"{start_date}T00:00:00Z",
                                "to": f"{end_date}T23:59:59Z"
                            },
                            "acquisitionMode": "IW",
                            "polarization": "DV"
                        },
                        "processing": {
                            "orthorectify": True,
                            "backscatterCoeff": "GAMMA0_TERRAIN"
                        }
                    }
                ]
            },
            "output": {
                "width": width,
                "height": height,
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
            },
            "evalscript": evalscript
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(process_url, json=payload, headers=headers)
                if resp.status_code == 401:
                    self.auth.clear_cached_token()
                    token = await self.auth.get_access_token(force_refresh=True)
                    headers["Authorization"] = f"Bearer {token}"
                    resp = await client.post(process_url, json=payload, headers=headers)

                if resp.status_code == 200 and resp.content:
                    import tifffile
                    with io.BytesIO(resp.content) as bio:
                        tiff_data = tifffile.imread(bio)
                        # Shape is (H, W, 3) or (3, H, W)
                        if tiff_data.ndim == 3:
                            if tiff_data.shape[0] == 3:
                                vv_linear = tiff_data[0].astype(np.float32)
                                vh_linear = tiff_data[1].astype(np.float32)
                                mask = (tiff_data[2] > 0.5)
                            else:
                                vv_linear = tiff_data[:, :, 0].astype(np.float32)
                                vh_linear = tiff_data[:, :, 1].astype(np.float32)
                                mask = (tiff_data[:, :, 2] > 0.5)

                            # Clean extreme values
                            vv_linear = np.nan_to_num(vv_linear, nan=0.0, posinf=0.0, neginf=0.0)
                            vh_linear = np.nan_to_num(vh_linear, nan=0.0, posinf=0.0, neginf=0.0)
                            return vv_linear, vh_linear, mask, {
                                "scene_id": "S1_GRD_LIVE_OBSERVATION",
                                "acquisition_date": f"{end_date}T00:38:15Z",
                                "data_source": "Copernicus Sentinel-1 GRD",
                                "is_demo": False,
                                "polarization": ["VV", "VH"],
                            }
        except Exception:
            pass

        # Fallback to deterministic demo SAR data if live fetch failed
        return self._generate_deterministic_sar_data(geometry, start_date, end_date, width, height)

    def _generate_deterministic_sar_data(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        width: int = 128,
        height: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generates realistic, continuous synthetic C-band SAR backscatter arrays.
        Grounds features in agricultural physical properties:
        - VV linear: ~0.05 to 0.25 (-13 dB to -6 dB, sensitive to canopy structure & surface roughness)
        - VH linear: ~0.008 to 0.05 (-21 dB to -13 dB, volumetric cross-pol scattering from biomass)
        - VV/VH ratio: ~3 to 8 (4.7 dB to 9.0 dB, canopy density indicator)
        """
        # Derive spatial seed from AOI center coordinates
        coords = geometry.get("coordinates", [[[81.52, 16.81]]])
        first_ring = coords[0] if coords else [[81.52, 16.81]]
        lons = [p[0] for p in first_ring if len(p) >= 2]
        lats = [p[1] for p in first_ring if len(p) >= 2]
        c_lon = sum(lons) / max(1, len(lons))
        c_lat = sum(lats) / max(1, len(lats))

        # Generate smooth spatial field
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        dist_sq = x * x + y * y

        # Field patterns (crop furrows, canopy zones, slight boundary variation)
        spatial_pattern = (
            0.5 * np.cos(3.0 * x + (c_lon % 3.0)) * np.sin(3.0 * y + (c_lat % 3.0))
            + 0.25 * np.cos(6.0 * x)
            - 0.3 * dist_sq
        )
        spatial_pattern = (spatial_pattern - spatial_pattern.min()) / max(1e-5, (spatial_pattern.max() - spatial_pattern.min()))

        # Typical agricultural C-band backscatter in dB:
        # VV: -14 dB to -7 dB
        # VH: -22 dB to -13 dB
        vv_db = -14.0 + 7.0 * spatial_pattern
        vh_db = -22.0 + 9.0 * spatial_pattern

        # Convert to linear power: linear = 10^(dB / 10)
        vv_linear = (10.0 ** (vv_db / 10.0)).astype(np.float32)
        vh_linear = (10.0 ** (vh_db / 10.0)).astype(np.float32)

        # Circular AOI valid mask
        valid_mask = dist_sq <= 1.0

        metadata = {
            "scene_id": f"S1A_IW_GRDH_1SDV_{end_date.replace('-', '')}T003815_DEMO",
            "acquisition_date": f"{end_date}T00:38:15Z",
            "product_type": "Sentinel-1 GRD",
            "polarization": ["VV", "VH"],
            "orbit_direction": "descending",
            "instrument_mode": "IW",
            "resolution_m": 10,
            "data_source": "Copernicus Sentinel-1 (Deterministic Demo)",
            "is_demo": True,
        }
        return vv_linear, vh_linear, valid_mask, metadata

    def compute_sar_features(
        self,
        vv_linear: np.ndarray,
        vh_linear: np.ndarray,
        valid_mask: np.ndarray,
        total_aoi_area_km2: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Computes calibrated SAR features:
        - VV Backscatter (linear & dB)
        - VH Backscatter (linear & dB)
        - VV/VH Cross-Ratio (linear & dB difference)
        Calculates robust descriptive statistics and generates visual map colormaps.
        """
        # Validate shapes
        validate_band_shapes({"VV": vv_linear, "VH": vh_linear, "Mask": valid_mask})

        # Safeguard numerical inputs
        vv_clean = np.maximum(vv_linear, 1e-6)
        vh_clean = np.maximum(vh_linear, 1e-6)

        # Convert to Decibels: dB = 10 * log10(linear)
        vv_db = 10.0 * np.log10(vv_clean)
        vh_db = 10.0 * np.log10(vh_clean)

        # VV/VH Cross-ratio (linear) and dB difference (VV_dB - VH_dB)
        # Note: 10 * log10(VV / VH) = VV_dB - VH_dB
        vv_vh_ratio_linear = vv_clean / vh_clean
        vv_vh_db_diff = vv_db - vh_db

        # Apply valid mask
        valid_pixels = int(np.count_nonzero(valid_mask))
        total_pixels = int(valid_mask.size)
        valid_pct = round((valid_pixels / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0

        area_km2 = (
            total_aoi_area_km2
            if total_aoi_area_km2 is not None
            else round((valid_pixels * 100) / 1_000_000, 3)
        )

        def _calc_stats(arr: np.ndarray) -> Dict[str, Optional[float]]:
            if valid_pixels == 0:
                return {"mean": None, "median": None, "min": None, "max": None, "std": None}
            valid_vals = arr[valid_mask]
            return {
                "mean": round(float(np.mean(valid_vals)), 3),
                "median": round(float(np.median(valid_vals)), 3),
                "min": round(float(np.min(valid_vals)), 3),
                "max": round(float(np.max(valid_vals)), 3),
                "std": round(float(np.std(valid_vals)), 3),
            }

        vv_stats = _calc_stats(vv_db)
        vh_stats = _calc_stats(vh_db)
        ratio_stats = _calc_stats(vv_vh_db_diff)
        ratio_linear_stats = _calc_stats(vv_vh_ratio_linear)

        # Generate base64 colormaps
        vv_map = self._generate_sar_colormap(vv_db, valid_mask, min_val=-22.0, max_val=-5.0, palette="gray")
        vh_map = self._generate_sar_colormap(vh_db, valid_mask, min_val=-28.0, max_val=-12.0, palette="viridis")
        ratio_map = self._generate_sar_colormap(vv_vh_db_diff, valid_mask, min_val=2.0, max_val=12.0, palette="turbo")

        return {
            "features": {
                "vv": {
                    "unit": "dB",
                    "description": "VV Co-polarized Backscatter: sensitive to surface roughness, soil moisture, and vertical stalk structure",
                    "statistics": vv_stats,
                    "map": vv_map,
                },
                "vh": {
                    "unit": "dB",
                    "description": "VH Cross-polarized Backscatter: dominated by volumetric depolarizing scattering within vegetative canopy biomass",
                    "statistics": vh_stats,
                    "map": vh_map,
                },
                "vv_vh_ratio": {
                    "unit": "dB difference",
                    "description": "VV/VH Cross-Ratio (VV_dB - VH_dB): diagnostic indicator of canopy structural density vs surface dominance",
                    "statistics": ratio_stats,
                    "linear_statistics": ratio_linear_stats,
                    "map": ratio_map,
                },
            },
            "quality": {
                "valid_pixels": valid_pixels,
                "total_pixels": total_pixels,
                "valid_pixel_percentage": valid_pct,
                "area_represented_km2": area_km2,
                "quality_level": "Excellent" if valid_pct >= 90 else ("Good" if valid_pct >= 75 else "Fair"),
            },
            "raw_arrays": {
                "vv_db": vv_db,
                "vh_db": vh_db,
                "vv_vh_db_diff": vv_vh_db_diff,
                "valid_mask": valid_mask,
            }
        }

    def _generate_sar_colormap(
        self,
        data: np.ndarray,
        valid_mask: np.ndarray,
        min_val: float,
        max_val: float,
        palette: str = "gray",
    ) -> str:
        """Renders 2D SAR feature grid into a Base64 PNG with transparent nodata."""
        norm = np.clip((data - min_val) / max(1e-5, (max_val - min_val)), 0.0, 1.0)
        h, w = data.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        if palette == "gray":
            # Classical SAR Radar Grayscale (0 = dark, 255 = bright backscatter)
            intensity = (norm * 255).astype(np.uint8)
            rgba[:, :, 0] = intensity
            rgba[:, :, 1] = intensity
            rgba[:, :, 2] = intensity
        elif palette == "viridis":
            # Deep purple to teal to yellow for cross-polarization biomass
            rgba[:, :, 0] = (255 * (0.2 + 0.8 * norm)).astype(np.uint8)
            rgba[:, :, 1] = (255 * (0.1 + 0.85 * (norm ** 1.2))).astype(np.uint8)
            rgba[:, :, 2] = (255 * (0.4 + 0.5 * (1.0 - norm))).astype(np.uint8)
        else: # Turbo / Pseudocolor for ratio
            # Cyan to Emerald to Amber to Red
            rgba[:, :, 0] = (255 * np.clip(1.5 * norm - 0.2, 0.0, 1.0)).astype(np.uint8)
            rgba[:, :, 1] = (255 * np.sin(norm * np.pi)).astype(np.uint8)
            rgba[:, :, 2] = (255 * np.clip(1.2 * (1.0 - norm), 0.0, 1.0)).astype(np.uint8)

        rgba[:, :, 3] = np.where(valid_mask, 220, 0).astype(np.uint8)

        img = Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    async def get_sar_analysis(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        aoi_area_km2: Optional[float] = None,
        s2_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Complete high-level SAR analysis pipeline:
        1. Catalog search for Sentinel-1 GRD acquisitions
        2. Temporal matching against Sentinel-2 date if provided
        3. Retrieval/generation of calibrated VV and VH bands
        4. Computation of SAR metrics, statistics, and colormaps
        5. Returning structured multimodal response
        """
        self.validate_inputs(geometry, start_date, end_date)

        # 1. Catalog search
        catalog = await self.search_imagery(geometry, start_date, end_date)

        # 2. Temporal matching if S2 date provided
        temporal_match = None
        if s2_date:
            temporal_match = self.find_closest_sentinel1_observation(s2_date, catalog, max_window_days=7)

        # 3. Download/synthesize bands
        vv_lin, vh_lin, mask, scene_meta = await self.get_sar_bands(geometry, start_date, end_date)

        # 4. Compute SAR features & colormaps
        features_result = self.compute_sar_features(
            vv_lin, vh_lin, mask, total_aoi_area_km2=aoi_area_km2
        )

        return {
            "status": "success",
            "observation_date": scene_meta.get("acquisition_date", f"{end_date}T00:38:15Z"),
            "product_type": scene_meta.get("product_type", "Sentinel-1 GRD"),
            "polarization": scene_meta.get("polarization", ["VV", "VH"]),
            "orbit_direction": scene_meta.get("orbit_direction", "descending"),
            "scene_id": scene_meta.get("scene_id"),
            "data_source": scene_meta.get("data_source"),
            "is_demo": scene_meta.get("is_demo", False),
            "temporal_match": temporal_match,
            "sar_features": features_result["features"],
            "quality": features_result["quality"],
            "available_catalog_scenes": len(catalog),
        }

# Global singleton
sentinel1_service = Sentinel1Service()
