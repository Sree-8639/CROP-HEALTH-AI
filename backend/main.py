import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

from fastapi import FastAPI, HTTPException, Query, status, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from services.geospatial import geocode_query, get_location_details, calculate_aoi
from services.copernicus_auth import (
    copernicus_auth,
    CopernicusAuthError,
    CopernicusCredentialsMissingError,
)
from services.sentinel2_service import (
    sentinel2_service,
    Sentinel2Error,
    NoImageryFoundError,
    InvalidAOIError,
    InvalidDateRangeError,
)
from services.historical_service import (
    historical_monitoring_service,
    HistoricalMonitoringError,
)
from models.ai_models import FieldEvidenceInput, AIFieldReport
from services.ai_analysis_service import ai_analysis_service
from services.sentinel1_service import (
    sentinel1_service,
    Sentinel1Error,
    NoSARDataFoundError,
    InvalidSARGeometryError,
    InvalidSARDatesError,
)
from services.fusion_service import multimodal_fusion_service
from services.stress_detection_service import (
    stress_detection_service,
    StressDetectionError,
    ModelUnavailableError,
)
from services.growth_prediction_service import (
    growth_prediction_service,
    GrowthPredictionError,
)
from ml.growth_prediction.schema import GrowthPredictionRequest
from services.explainability_service import explainability_service
from ml.explainability.schema import ExplainabilityRequest
from services.field_risk_mapping_service import (
    field_risk_mapping_service,
    FieldRiskMappingError,
)
from ml.field_risk.schema import (
    FieldRiskMapRequest,
    PixelInspectionRequest,
)
from services.farmer_dashboard_service import (
    farmer_dashboard_service,
    FieldNotFoundError,
)
from ml.farmer_dashboard.schema import (
    RegisterFieldRequest,
)

app = FastAPI(
    title="Crop Health Monitoring API",
    description="Backend API for Crop Health Monitoring using Satellite Imagery (Phase 17: Advanced Farmer Intelligence Dashboard)",
    version="1.7.0",
)


# Enable CORS for frontend
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
if allowed_origins_env:
    origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]
else:
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AOIRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Center latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Center longitude")
    radius_km: float = Field(..., gt=0.0, le=50.0, description="Radius in kilometers")

class SatelliteSearchRequest(BaseModel):
    geometry: Dict[str, Any] = Field(..., description="GeoJSON Polygon geometry")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    max_cloud_coverage: float = Field(
        default=30.0, ge=0.0, le=100.0, description="Maximum allowed cloud coverage percentage"
    )

class VegetationIndexRequest(BaseModel):
    geometry: Dict[str, Any] = Field(..., description="GeoJSON Polygon geometry")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    max_cloud_coverage: float = Field(
        default=30.0, ge=0.0, le=100.0, description="Maximum allowed cloud coverage percentage"
    )
    aoi_area_km2: Optional[float] = Field(
        default=None, description="Optional total field area in square kilometers"
    )

class CropConditionRequest(VegetationIndexRequest):
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional custom heuristic condition configuration thresholds and weights",
    )

# Alias for backwards compatibility with Phase 4
NDVIAnalysisRequest = VegetationIndexRequest

class HistoricalAnalysisRequest(BaseModel):
    geometry: Optional[Dict[str, Any]] = Field(
        default=None, description="GeoJSON Polygon geometry"
    )
    latitude: Optional[float] = Field(
        default=None, ge=-90.0, le=90.0, description="Center latitude"
    )
    longitude: Optional[float] = Field(
        default=None, ge=-180.0, le=180.0, description="Center longitude"
    )
    radius_km: Optional[float] = Field(
        default=2.0, gt=0.0, le=50.0, description="Radius in kilometers"
    )
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    max_cloud_coverage: float = Field(
        default=30.0, ge=0.0, le=100.0, description="Maximum allowed cloud coverage percentage"
    )
    tolerance: float = Field(
        default=0.03, ge=0.001, le=0.5, description="Configurable numeric trend classification tolerance"
    )
    aoi_area_km2: Optional[float] = Field(
        default=None, description="Total field area in square kilometers"
    )

class HistoricalCompareRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID from a previous history request")
    observation_a_id: str = Field(..., description="First observation ID")
    observation_b_id: str = Field(..., description="Second observation ID")

class Sentinel1SearchRequest(BaseModel):
    geometry: Optional[Dict[str, Any]] = Field(default=None, description="GeoJSON Polygon geometry")
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0, description="Center latitude")
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0, description="Center longitude")
    radius_km: Optional[float] = Field(default=2.0, gt=0.0, le=50.0, description="Radius in kilometers")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    orbit_direction: Optional[str] = Field(default=None, description="Optional 'ascending' or 'descending'")

class Sentinel1AnalysisRequest(BaseModel):
    geometry: Optional[Dict[str, Any]] = Field(default=None, description="GeoJSON Polygon geometry")
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0, description="Center latitude")
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0, description="Center longitude")
    radius_km: Optional[float] = Field(default=2.0, gt=0.0, le=50.0, description="Radius in kilometers")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    s2_date: Optional[str] = Field(default=None, description="Optional Sentinel-2 date for temporal matching")
    aoi_area_km2: Optional[float] = Field(default=None, description="Optional field area in square kilometers")

class MultimodalFusionRequest(BaseModel):
    geometry: Optional[Dict[str, Any]] = Field(default=None, description="GeoJSON Polygon geometry")
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0, description="Center latitude")
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0, description="Center longitude")
    radius_km: Optional[float] = Field(default=2.0, gt=0.0, le=50.0, description="Radius in kilometers")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    max_cloud_coverage: float = Field(default=30.0, ge=0.0, le=100.0)
    temporal_window_days: int = Field(default=3, ge=0, le=30, description="Max temporal gap in days between S1 and S2")
    normalization: str = Field(default="raw", description="Feature normalization: 'raw', 'min-max', or 'z-score'")
    aoi_area_km2: Optional[float] = Field(default=None)

class MultimodalFusionExportRequest(MultimodalFusionRequest):
    export_format: str = Field(default="npz", description="Export format: 'npz', 'json', or 'geotiff'")

class EarlyCropStressDetectionRequest(BaseModel):
    geometry: Optional[Dict[str, Any]] = Field(default=None, description="GeoJSON Polygon geometry")
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0, description="Center latitude")
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0, description="Center longitude")
    radius_km: Optional[float] = Field(default=2.0, gt=0.0, le=50.0, description="Radius in kilometers")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    max_cloud_coverage: float = Field(default=30.0, ge=0.0, le=100.0)
    max_temporal_window_days: int = Field(default=3, ge=1, le=14, description="Max temporal gap in days between S1 and S2")
    spatial_inference: bool = Field(default=True, description="Whether to produce per-pixel spatial stress probability raster")
    aoi_area_km2: Optional[float] = Field(default=None)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Crop Health Monitoring API",
        "phase": "Phase 8: Historical Crop Monitoring & Time-Series Analysis",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "geocode": "/api/v1/geocode?query=...",
            "location": "/api/v1/location?lat=...&lon=...",
            "aoi": "/api/v1/aoi",
            "satellite_search": "POST /api/v1/satellite/search",
            "satellite_preview": "POST /api/v1/satellite/preview",
            "ndvi_analysis": "POST /api/v1/analysis/ndvi",
            "ndmi_analysis": "POST /api/v1/analysis/ndmi",
            "ndre_analysis": "POST /api/v1/analysis/ndre",
            "evi_analysis": "POST /api/v1/analysis/evi",
            "all_indices": "POST /api/v1/analysis/indices",
            "crop_condition": "POST /api/v1/analysis/crop-condition",
            "historical_monitoring": "POST /api/v1/analysis/history",
            "historical_compare": "POST /api/v1/analysis/history/compare",
        },
        "copernicus_status": {
            "configured": copernicus_auth.is_configured(),
            "demo_mode": sentinel2_service.is_demo_mode,
        },
    }

@app.get("/health")
@app.get("/api/health")
def get_health():
    return {
        "status": "ok",
        "phase": 6,
        "platform_version": "v1.8.0",
        "indices_supported": ["NDVI", "NDMI", "NDRE", "EVI"],
        "features": [
            "crop_condition",
            "spatial_zones",
            "stress_patterns",
            "data_quality",
            "multimodal_fusion",
            "stress_detection",
            "growth_prediction",
            "explainability",
            "field_risk_mapping",
            "farmer_intelligence_dashboard",
        ],
        "copernicus_configured": copernicus_auth.is_configured(),
        "demo_mode": sentinel2_service.is_demo_mode,
    }

@app.get("/api/v1")
def get_v1_status():
    return {
        "message": "Crop Health API v1 is running successfully",
        "status": "online",
        "phase": "Phase 6: Crop Condition Intelligence",
        "indices": ["NDVI", "NDMI", "NDRE", "EVI"],
        "spectral_bands": ["B02 (Blue)", "B04 (Red)", "B05 (RedEdge)", "B08 (NIR)", "B11 (SWIR)"],
        "condition_zones": ["No Data (0)", "Very Low (1)", "Low (2)", "Moderate (3)", "Good (4)", "Very Good (5)"],
        "data_source": "Copernicus Data Space Ecosystem / Sentinel Hub",
        "copernicus_configured": copernicus_auth.is_configured(),
        "demo_mode": sentinel2_service.is_demo_mode,
    }

@app.get("/api/v1/geocode")
async def geocode_endpoint(query: str = Query(..., min_length=1, description="Location search query")):
    """
    Geocodes a search string into latitude, longitude, and formatted place information.
    """
    clean_query = query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Location query cannot be empty.")

    try:
        result = await geocode_query(clean_query)
        return result
    except LookupError:
        raise HTTPException(status_code=404, detail="Location not found. Please try another location.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Geocoding service unavailable or failed. Details: {str(exc)}"
        )

@app.get("/api/v1/location")
async def get_location_endpoint(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude"),
):
    """
    Returns location information for the provided latitude and longitude.
    """
    try:
        location_info = await get_location_details(lat, lon)
        return location_info
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to retrieve location details: {str(exc)}"
        )

@app.post("/api/v1/aoi")
def create_aoi_endpoint(payload: AOIRequest):
    """
    Creates an Area of Interest (AOI) with bounding box and GeoJSON polygon geometry.
    """
    try:
        aoi = calculate_aoi(payload.latitude, payload.longitude, payload.radius_km)
        return {
            "status": "success",
            "message": "AOI created successfully",
            "aoi": aoi,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/api/v1/aoi")
def get_aoi_endpoint(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    radius_km: float = Query(..., gt=0.0, le=50.0),
):
    """
    GET alternative for computing AOI geometry.
    """
    try:
        aoi = calculate_aoi(lat, lon, radius_km)
        return {
            "status": "success",
            "message": "AOI created successfully",
            "aoi": aoi,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/api/v1/satellite/search")
async def search_satellite_endpoint(payload: SatelliteSearchRequest):
    """
    Searches for suitable Sentinel-2 Level-2A imagery for the given AOI and date range.
    """
    try:
        search_result = await sentinel2_service.search_imagery(
            geometry=payload.geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            max_cloud_coverage=payload.max_cloud_coverage,
        )
        return search_result
    except CopernicusCredentialsMissingError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Satellite service credentials are not configured. Please configure COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in backend/.env.",
        )
    except CopernicusAuthError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication with Copernicus satellite service failed. Please verify your client credentials.",
        )
    except (InvalidAOIError, InvalidDateRangeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except NoImageryFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No suitable Sentinel-2 imagery was found for this area and date range.",
        )
    except Sentinel2Error as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Satellite service error: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error processing satellite search: {str(exc)}",
        )

@app.post("/api/v1/satellite/preview")
async def preview_satellite_endpoint(payload: SatelliteSearchRequest):
    """
    Requests and returns a true-color (B04-Red, B03-Green, B02-Blue) Sentinel-2 L2A preview image
    for the selected AOI.
    """
    try:
        preview_result = await sentinel2_service.get_preview_image(
            geometry=payload.geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            max_cloud_coverage=payload.max_cloud_coverage,
        )
        return preview_result
    except CopernicusCredentialsMissingError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Satellite service credentials are not configured. Please configure COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in backend/.env.",
        )
    except CopernicusAuthError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication with Copernicus satellite service failed. Please verify your client credentials.",
        )
    except (InvalidAOIError, InvalidDateRangeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except NoImageryFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No suitable Sentinel-2 imagery was found for this area and date range.",
        )
    except Sentinel2Error as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Satellite service error: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error generating satellite preview: {str(exc)}",
        )

@app.post("/api/v1/analysis/ndvi")
async def analyze_ndvi_endpoint(payload: NDVIAnalysisRequest):
    """
    Phase 4: Performs scientific Normalized Difference Vegetation Index (NDVI) analysis
    using Sentinel-2 B04 (Red) and B08 (NIR) spectral bands.
    Returns:
    - scene_id & acquisition date
    - cloud coverage & valid pixel percentage
    - comprehensive NDVI statistics (min, max, mean, median, std dev, vigor breakdown)
    - continuous color-mapped NDVI visualization map
    """
    try:
        ndvi_result = await sentinel2_service.calculate_ndvi_for_aoi(
            geometry=payload.geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            max_cloud_coverage=payload.max_cloud_coverage,
            aoi_area_km2=payload.aoi_area_km2,
        )
        return ndvi_result
    except CopernicusCredentialsMissingError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Satellite service credentials are not configured. Please configure COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in backend/.env.",
        )
    except CopernicusAuthError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication with Copernicus satellite service failed. Please verify your client credentials.",
        )
    except (InvalidAOIError, InvalidDateRangeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except NoImageryFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No suitable Sentinel-2 imagery was found for this area and date range.",
        )
    except Sentinel2Error as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Satellite service error: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error computing NDVI analysis: {str(exc)}",
        )

@app.post("/api/v1/analysis/ndmi")
async def analyze_ndmi_endpoint(payload: VegetationIndexRequest):
    """
    Phase 5: Performs Normalized Difference Moisture Index (NDMI) analysis
    using Sentinel-2 B08 (NIR) and B11 (SWIR) spectral bands.
    NDMI = (B08 - B11) / (B08 + B11)
    Vegetation moisture/water-related spectral indicator.
    """
    try:
        ndmi_result = await sentinel2_service.calculate_ndmi_for_aoi(
            geometry=payload.geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            max_cloud_coverage=payload.max_cloud_coverage,
            aoi_area_km2=payload.aoi_area_km2,
        )
        return ndmi_result
    except CopernicusCredentialsMissingError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Satellite service credentials are not configured. Please configure COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in backend/.env.",
        )
    except CopernicusAuthError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication with Copernicus satellite service failed. Please verify your client credentials.",
        )
    except (InvalidAOIError, InvalidDateRangeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except NoImageryFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No suitable Sentinel-2 imagery was found for this area and date range.",
        )
    except Sentinel2Error as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Satellite service error: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error computing NDMI analysis: {str(exc)}",
        )

@app.post("/api/v1/analysis/ndre")
async def analyze_ndre_endpoint(payload: VegetationIndexRequest):
    """
    Phase 5: Performs Normalized Difference Red Edge Index (NDRE) analysis
    using Sentinel-2 B08 (NIR) and B05 (RedEdge) spectral bands with spatial alignment.
    NDRE = (B08 - B05) / (B08 + B05)
    Red-edge vegetation indicator.
    """
    try:
        ndre_result = await sentinel2_service.calculate_ndre_for_aoi(
            geometry=payload.geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            max_cloud_coverage=payload.max_cloud_coverage,
            aoi_area_km2=payload.aoi_area_km2,
        )
        return ndre_result
    except CopernicusCredentialsMissingError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Satellite service credentials are not configured. Please configure COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in backend/.env.",
        )
    except CopernicusAuthError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication with Copernicus satellite service failed. Please verify your client credentials.",
        )
    except (InvalidAOIError, InvalidDateRangeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except NoImageryFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No suitable Sentinel-2 imagery was found for this area and date range.",
        )
    except Sentinel2Error as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Satellite service error: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error computing NDRE analysis: {str(exc)}",
        )

@app.post("/api/v1/analysis/evi")
async def analyze_evi_endpoint(payload: VegetationIndexRequest):
    """
    Phase 5: Performs Enhanced Vegetation Index (EVI) analysis
    using Sentinel-2 B08 (NIR), B04 (Red), and B02 (Blue) spectral bands.
    EVI = 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)
    Enhanced vegetation index.
    """
    try:
        evi_result = await sentinel2_service.calculate_evi_for_aoi(
            geometry=payload.geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            max_cloud_coverage=payload.max_cloud_coverage,
            aoi_area_km2=payload.aoi_area_km2,
        )
        return evi_result
    except CopernicusCredentialsMissingError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Satellite service credentials are not configured. Please configure COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in backend/.env.",
        )
    except CopernicusAuthError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication with Copernicus satellite service failed. Please verify your client credentials.",
        )
    except (InvalidAOIError, InvalidDateRangeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except NoImageryFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No suitable Sentinel-2 imagery was found for this area and date range.",
        )
    except Sentinel2Error as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Satellite service error: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error computing EVI analysis: {str(exc)}",
        )

@app.post("/api/v1/analysis/indices")
async def analyze_all_indices_endpoint(payload: VegetationIndexRequest):
    """
    Phase 5 Combined Endpoint: Computes NDVI, NDMI, NDRE, and EVI concurrently
    from a single unified spectral band acquisition pass (B02, B04, B05, B08, B11, SCL, dataMask).
    Returns all 4 vegetation indices and common scene metadata without downloading bands multiple times.
    """
    try:
        all_indices_result = await sentinel2_service.calculate_all_indices_for_aoi(
            geometry=payload.geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            max_cloud_coverage=payload.max_cloud_coverage,
            aoi_area_km2=payload.aoi_area_km2,
        )
        return all_indices_result
    except CopernicusCredentialsMissingError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Satellite service credentials are not configured. Please configure COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in backend/.env.",
        )
    except CopernicusAuthError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication with Copernicus satellite service failed. Please verify your client credentials.",
        )
    except (InvalidAOIError, InvalidDateRangeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except NoImageryFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No suitable Sentinel-2 imagery was found for this area and date range.",
        )
    except Sentinel2Error as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Satellite service error: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error computing combined indices analysis: {str(exc)}",
        )

@app.post("/api/v1/analysis/crop-condition")
async def analyze_crop_condition_endpoint(payload: CropConditionRequest):
    """
    Phase 6: Computes heuristic Crop Condition Intelligence combining Sentinel-2
    multispectral indices (NDVI + NDMI + NDRE + EVI), spatial condition zones,
    stress pattern detection, and data quality information.
    """
    try:
        condition_result = await sentinel2_service.calculate_crop_condition_for_aoi(
            geometry=payload.geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            max_cloud_coverage=payload.max_cloud_coverage,
            aoi_area_km2=payload.aoi_area_km2,
            config=payload.config,
        )
        return condition_result
    except CopernicusCredentialsMissingError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Satellite service credentials are not configured. Please configure COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in backend/.env.",
        )
    except CopernicusAuthError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication with Copernicus satellite service failed. Please verify your client credentials.",
        )
    except (InvalidAOIError, InvalidDateRangeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except NoImageryFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Sentinel2Error as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Satellite service error: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error computing crop condition analysis: {str(exc)}",
        )

@app.get("/api/v1/analysis/crop-condition")
async def get_crop_condition_endpoint(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    radius_km: float = Query(2.0, gt=0.0, le=50.0),
    start_date: str = Query(...),
    end_date: str = Query(...),
    max_cloud_coverage: float = Query(30.0, ge=0.0, le=100.0),
):
    """
    GET alternative for computing heuristic Crop Condition Intelligence using lat/lon and radius.
    """
    try:
        from services.geospatial import calculate_aoi
        aoi_data = calculate_aoi(lat, lon, radius_km)
        geometry = aoi_data["geometry"]
        aoi_area = aoi_data["area_sq_km"]

        condition_result = await sentinel2_service.calculate_crop_condition_for_aoi(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
            max_cloud_coverage=max_cloud_coverage,
            aoi_area_km2=aoi_area,
        )
        return condition_result
    except CopernicusCredentialsMissingError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Satellite service credentials are not configured.",
        )
    except CopernicusAuthError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication with Copernicus satellite service failed.",
        )
    except (InvalidAOIError, InvalidDateRangeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except NoImageryFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Sentinel2Error as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Satellite service error: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error computing crop condition analysis: {str(exc)}",
        )

# ==========================================================
# PHASE 8: HISTORICAL CROP MONITORING & TIME-SERIES ANALYSIS
# ==========================================================

@app.post("/api/v1/analysis/history")
async def historical_time_series_endpoint(payload: HistoricalAnalysisRequest):
    """
    Phase 8 Historical Monitoring:
    Retrieves chronological Sentinel-2 observations across a specified date range,
    filters redundant passes, computes NDVI, NDMI, NDRE, and EVI time-series statistics,
    evaluates data quality, and generates directional trend summaries with configurable tolerance.
    """
    try:
        geometry = payload.geometry
        aoi_area = payload.aoi_area_km2

        # If geometry is omitted, calculate from center lat/lon and radius
        if not geometry:
            if payload.latitude is None or payload.longitude is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Either 'geometry' (GeoJSON Polygon) or both 'latitude' and 'longitude' must be provided.",
                )
            from services.geospatial import calculate_aoi
            aoi_data = calculate_aoi(payload.latitude, payload.longitude, payload.radius_km or 2.0)
            geometry = aoi_data["geometry"]
            if not aoi_area:
                aoi_area = aoi_data["area_sq_km"]

        result = await historical_monitoring_service.get_historical_time_series(
            geometry=geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            max_cloud_coverage=payload.max_cloud_coverage,
            aoi_area_km2=aoi_area,
            tolerance=payload.tolerance,
        )
        return result
    except (InvalidAOIError, InvalidDateRangeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except NoImageryFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Sentinel2Error as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Satellite service error: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error retrieving historical time-series: {str(exc)}",
        )

@app.get("/api/v1/analysis/history")
async def get_historical_time_series_get_endpoint(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    radius_km: float = Query(2.0, gt=0.0, le=50.0),
    start_date: str = Query(...),
    end_date: str = Query(...),
    max_cloud_coverage: float = Query(30.0, ge=0.0, le=100.0),
    tolerance: float = Query(0.03, ge=0.001, le=0.5),
):
    """
    GET alternative for Phase 8 Historical Monitoring using query parameters.
    """
    try:
        from services.geospatial import calculate_aoi
        aoi_data = calculate_aoi(lat, lon, radius_km)
        geometry = aoi_data["geometry"]
        aoi_area = aoi_data["area_sq_km"]

        result = await historical_monitoring_service.get_historical_time_series(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
            max_cloud_coverage=max_cloud_coverage,
            aoi_area_km2=aoi_area,
            tolerance=tolerance,
        )
        return result
    except (InvalidAOIError, InvalidDateRangeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except NoImageryFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Sentinel2Error as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Satellite service error: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error retrieving historical time-series: {str(exc)}",
        )

@app.post("/api/v1/analysis/history/compare")
def compare_observations_endpoint(payload: HistoricalCompareRequest):
    """
    Phase 8 Observation Comparison & Change Map:
    Performs pairwise comparison between Observation A and Observation B,
    computing metric differences and generating an interactive spatial NDVI Change Map.
    """
    try:
        comparison_result = historical_monitoring_service.compare_two_observations(
            session_id=payload.session_id,
            obs_a_id=payload.observation_a_id,
            obs_b_id=payload.observation_b_id,
        )
        return comparison_result
    except HistoricalMonitoringError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error comparing observations: {str(exc)}",
        )

@app.post("/api/v1/analysis/ai-report", response_model=AIFieldReport)
async def generate_ai_field_intelligence_endpoint(payload: FieldEvidenceInput):
    """
    Phase 9 AI Field Intelligence & Explainable Satellite Analysis:
    Interprets verified calculated evidence (NDVI, NDMI, NDRE, EVI, Crop Condition, Historical Trends)
    to generate an explainable, evidence-backed Field Intelligence Report.
    Adheres strictly to scientific agronomic guardrails.
    """
    try:
        report = await ai_analysis_service.generate_field_intelligence(payload)
        return report
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error generating AI Field Intelligence: {str(exc)}",
        )

@app.post("/api/v1/satellite/sentinel1/search")
async def search_sentinel1_endpoint(payload: Sentinel1SearchRequest):
    """
    Phase 11: Sentinel-1 C-band SAR Catalog Search:
    Queries available Sentinel-1 GRD acquisitions over the AOI for the specified date window.
    """
    try:
        geometry = payload.geometry
        if not geometry:
            if payload.latitude is not None and payload.longitude is not None:
                aoi = calculate_aoi(payload.latitude, payload.longitude, payload.radius_km or 2.0)
                geometry = aoi["geometry"]
            else:
                raise HTTPException(status_code=400, detail="Either 'geometry' or 'latitude' and 'longitude' must be provided.")

        results = await sentinel1_service.search_imagery(
            geometry=geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            orbit_direction=payload.orbit_direction,
        )
        return {
            "status": "success",
            "total_scenes_found": len(results),
            "scenes": results,
            "date_range": {"start_date": payload.start_date, "end_date": payload.end_date},
        }
    except (InvalidSARGeometryError, InvalidSARDatesError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except NoSARDataFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Sentinel1Error as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error querying Sentinel-1 catalog: {str(exc)}",
        )

@app.post("/api/v1/analysis/sentinel1")
async def analyze_sentinel1_endpoint(payload: Sentinel1AnalysisRequest):
    """
    Phase 11: Sentinel-1 SAR Backscatter Analysis:
    Extracts VV, VH backscatter, computes cross-ratio (VV/VH), descriptive statistics,
    colormap overlays, and performs temporal matching with optical Sentinel-2 data.
    """
    try:
        geometry = payload.geometry
        if not geometry:
            if payload.latitude is not None and payload.longitude is not None:
                aoi = calculate_aoi(payload.latitude, payload.longitude, payload.radius_km or 2.0)
                geometry = aoi["geometry"]
            else:
                raise HTTPException(status_code=400, detail="Either 'geometry' or 'latitude' and 'longitude' must be provided.")

        result = await sentinel1_service.get_sar_analysis(
            geometry=geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            aoi_area_km2=payload.aoi_area_km2,
            s2_date=payload.s2_date,
        )
        return result
    except (InvalidSARGeometryError, InvalidSARDatesError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except NoSARDataFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Sentinel1Error as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error computing Sentinel-1 SAR analysis: {str(exc)}",
        )

@app.post("/api/v1/analysis/fusion/prepare")
async def prepare_multimodal_fusion_endpoint(payload: MultimodalFusionRequest):
    """
    Phase 11 Multimodal Feature Preparation (Optical + SAR):
    Synthesizes and aligns multispectral Sentinel-2 indices with Sentinel-1 C-band SAR backscatter
    into a standardized dataset ready for future fusion models.
    """
    try:
        geometry = payload.geometry
        if not geometry:
            if payload.latitude is not None and payload.longitude is not None:
                aoi = calculate_aoi(payload.latitude, payload.longitude, payload.radius_km or 2.0)
                geometry = aoi["geometry"]
            else:
                raise HTTPException(status_code=400, detail="Either 'geometry' or 'latitude' and 'longitude' must be provided.")

        opt_data = await sentinel2_service.calculate_crop_condition_for_aoi(
            geometry=geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            max_cloud_coverage=payload.max_cloud_coverage,
            aoi_area_km2=payload.aoi_area_km2,
        )

        s2_date = opt_data.get("scene", {}).get("acquisition_date")

        sar_data = await sentinel1_service.get_sar_analysis(
            geometry=geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            aoi_area_km2=payload.aoi_area_km2,
            s2_date=s2_date,
        )

        fusion_dataset = multimodal_fusion_service.prepare_multimodal_dataset(
            optical_data=opt_data,
            sar_data=sar_data,
            temporal_match=sar_data.get("temporal_match"),
        )
        return {
            "status": "success",
            "fusion_dataset": fusion_dataset,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error preparing multimodal fusion dataset: {str(exc)}",
        )

@app.post("/api/v1/analysis/fusion")
async def analyze_multimodal_fusion_endpoint(payload: MultimodalFusionRequest):
    """
    Phase 12 End-to-End Multimodal Satellite Feature-Fusion Endpoint:
    Combines Sentinel-2 optical spectral features (B02-B11, NDVI, NDMI, NDRE, EVI)
    with Sentinel-1 SAR backscatter (VV, VH, VV/VH ratio), executing temporal matching (±3 days),
    spatial reprojection to canonical UTM grid, quality masking (VALID, PARTIAL, INVALID),
    feature quality statistics, and optional normalization.
    """
    try:
        geometry = payload.geometry
        if not geometry:
            if payload.latitude is not None and payload.longitude is not None:
                aoi = calculate_aoi(payload.latitude, payload.longitude, payload.radius_km or 2.0)
                geometry = aoi["geometry"]
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either 'geometry' or 'latitude' and 'longitude' must be provided."
                )

        fusion_record = await multimodal_fusion_service.execute_multimodal_fusion(
            geometry=geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            max_cloud_coverage=payload.max_cloud_coverage,
            temporal_window_days=payload.temporal_window_days,
            normalization=payload.normalization,
            aoi_area_km2=payload.aoi_area_km2,
        )

        # Exclude internal numpy arrays from HTTP JSON response
        clean_response = {
            k: v for k, v in fusion_record.items()
            if not k.startswith("_")
        }
        return clean_response

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error executing multimodal satellite fusion: {str(exc)}",
        )

@app.post("/api/v1/analysis/fusion/export")
async def export_multimodal_fusion_endpoint(payload: MultimodalFusionExportRequest):
    """
    Phase 12 Multimodal Dataset Export Endpoint:
    Exports standardized multimodal training samples as NumPy .npz, JSON, or GeoTIFF.
    Contains 13 core features, quality mask, canonical grid metadata, and labels_placeholder = None.
    """
    try:
        geometry = payload.geometry
        if not geometry:
            if payload.latitude is not None and payload.longitude is not None:
                aoi = calculate_aoi(payload.latitude, payload.longitude, payload.radius_km or 2.0)
                geometry = aoi["geometry"]
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either 'geometry' or 'latitude' and 'longitude' must be provided."
                )

        fusion_record = await multimodal_fusion_service.execute_multimodal_fusion(
            geometry=geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            max_cloud_coverage=payload.max_cloud_coverage,
            temporal_window_days=payload.temporal_window_days,
            normalization=payload.normalization,
            aoi_area_km2=payload.aoi_area_km2,
        )

        export_format = payload.export_format.lower()
        exported_data = multimodal_fusion_service.export_dataset(
            fusion_record, export_format=export_format
        )

        if export_format == "json":
            return exported_data
        elif export_format == "npz":
            return Response(
                content=exported_data,
                media_type="application/octet-stream",
                headers={"Content-Disposition": "attachment; filename=multimodal_fusion_dataset.npz"},
            )
        elif export_format in ("geotiff", "tiff", "tif"):
            return Response(
                content=exported_data,
                media_type="image/tiff",
                headers={"Content-Disposition": "attachment; filename=multimodal_fusion_stack.tif"},
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format '{export_format}'")

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error exporting multimodal fusion dataset: {str(exc)}",
        )

# ==========================================================
# PHASE 13: AI-POWERED EARLY CROP STRESS DETECTION
# ==========================================================

@app.post("/api/v1/analysis/stress-detection")
async def early_crop_stress_detection_endpoint(payload: EarlyCropStressDetectionRequest):
    """
    Phase 13 AI-Powered Early Crop Stress Detection Endpoint:
    Combines Sentinel-1 SAR and Sentinel-2 optical spectral features with multi-temporal
    historical observations to predict potential crop stress signals before severe canopy collapse.
    """
    try:
        geometry = payload.geometry
        if not geometry:
            if payload.latitude is not None and payload.longitude is not None:
                aoi = calculate_aoi(payload.latitude, payload.longitude, payload.radius_km or 2.0)
                geometry = aoi["geometry"]
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either 'geometry' or 'latitude' and 'longitude' must be provided."
                )

        result = await stress_detection_service.detect_crop_stress(
            geometry=geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            max_cloud_coverage=payload.max_cloud_coverage,
            max_temporal_window_days=payload.max_temporal_window_days,
            spatial_inference=payload.spatial_inference,
            aoi_area_km2=payload.aoi_area_km2,
        )
        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error executing early crop stress detection: {str(exc)}",
        )

# ==========================================================
# PHASE 14: AI-POWERED CROP GROWTH & TREND PREDICTION
# ==========================================================

@app.post("/api/v1/analysis/growth-prediction")
async def crop_growth_prediction_endpoint(payload: GrowthPredictionRequest):
    """
    Phase 14 AI-Powered Crop Growth & Trend Prediction Endpoint:
    Forecasts multi-temporal optical (NDVI, NDMI, NDRE, EVI) and SAR (VV, VH)
    canopy trajectories across configurable short, medium, and long horizons.
    Includes chronological walk-forward validation and calibrated prediction intervals.
    """
    try:
        geometry = payload.geometry
        if not geometry:
            if payload.latitude is not None and payload.longitude is not None:
                aoi = calculate_aoi(payload.latitude, payload.longitude, payload.radius_km or 2.0)
                geometry = aoi["geometry"]
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either 'geometry' or 'latitude' and 'longitude' must be provided."
                )

        result = await growth_prediction_service.predict_growth_and_trends(
            geometry=geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            forecast_horizon=payload.forecast_horizon or "medium",
            selected_indices=payload.selected_indices,
            model_type=payload.model_type or "linear_trend",
            max_cloud_coverage=payload.max_cloud_coverage or 35.0,
            aoi_area_km2=payload.aoi_area_km2,
        )
        return result

    except HTTPException:
        raise
    except GrowthPredictionError as gpe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(gpe),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error executing crop growth and trend prediction: {str(exc)}",
        )

# ==========================================================
# PHASE 15: EXPLAINABLE AI, CONFIDENCE & UNCERTAINTY INTELLIGENCE
# ==========================================================

@app.post("/api/v1/analysis/explainability")
async def explainability_analysis_endpoint(payload: ExplainabilityRequest):
    """
    Phase 15 Unified Explainability & Uncertainty Intelligence Endpoint:
    Provides local SHAP / controlled perturbation feature contributions, global feature importance,
    growth forecast candidate selection rationale, 4-tier data quality assessment,
    model reliability/calibration metadata, and empirical prediction intervals.
    """
    try:
        geometry = payload.geometry
        if not geometry:
            if payload.latitude is not None and payload.longitude is not None:
                aoi = calculate_aoi(payload.latitude, payload.longitude, payload.radius_km or 2.0)
                geometry = aoi["geometry"]
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either 'geometry' or 'latitude' and 'longitude' must be provided."
                )

        req_with_geom = ExplainabilityRequest(
            geometry=geometry,
            latitude=payload.latitude,
            longitude=payload.longitude,
            radius_km=payload.radius_km or 2.0,
            start_date=payload.start_date,
            end_date=payload.end_date,
            explanation_type=payload.explanation_type or "all",
            selected_index=payload.selected_index or "NDVI",
            model_version=payload.model_version,
            max_cloud_coverage=payload.max_cloud_coverage or 35.0,
            aoi_area_km2=payload.aoi_area_km2,
        )

        result = await explainability_service.generate_explainability_assessment(req_with_geom)
        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error executing explainability analysis: {str(exc)}",
        )

# ==========================================================
# PHASE 16: FIELD-LEVEL AI RISK MAPPING & SPATIAL INTELLIGENCE
# ==========================================================

@app.post("/api/v1/analysis/field-risk-map")
async def generate_field_risk_map_endpoint(payload: FieldRiskMapRequest):
    """
    Phase 16 Field-Level AI Risk Mapping Endpoint:
    Generates high-resolution spatial overlays for risk classification, vegetation indices (NDVI, NDMI, NDRE, EVI),
    early stress inference, historical change, and data quality on a canonical UTM 10m grid.
    """
    try:
        geometry = payload.geometry
        if not geometry:
            if payload.latitude is not None and payload.longitude is not None:
                aoi = calculate_aoi(payload.latitude, payload.longitude, payload.radius_km or 1.5)
                geometry = aoi["geometry"]
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Either 'geometry' or 'latitude' and 'longitude' must be provided."
                )

        req_with_geom = FieldRiskMapRequest(
            geometry=geometry,
            latitude=payload.latitude,
            longitude=payload.longitude,
            radius_km=payload.radius_km or 1.5,
            start_date=payload.start_date,
            end_date=payload.end_date,
            selected_layer=payload.selected_layer or "risk_classification",
            model_version=payload.model_version,
            max_cloud_coverage=payload.max_cloud_coverage or 35.0,
            aoi_area_km2=payload.aoi_area_km2,
            demo_mode=payload.demo_mode,
        )

        result = await field_risk_mapping_service.generate_field_risk_map(req_with_geom)
        return result

    except HTTPException:
        raise
    except FieldRiskMappingError as frme:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(frme),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error executing field risk mapping: {str(exc)}",
        )

@app.get("/api/v1/analysis/field-risk-map/{analysis_id}")
async def get_field_risk_map_endpoint(analysis_id: str):
    """
    Retrieves complete cached field risk analysis result by analysis ID.
    """
    cached = field_risk_mapping_service.get_cached_analysis(analysis_id)
    if not cached:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Field risk analysis '{analysis_id}' not found or session expired."
        )
    return cached.get("response")

@app.get("/api/v1/analysis/field-risk-map/{analysis_id}/statistics")
async def get_field_risk_statistics_endpoint(analysis_id: str):
    """
    Retrieves field-level summary statistics, area breakdown, and mean index metrics.
    """
    cached = field_risk_mapping_service.get_cached_analysis(analysis_id)
    if not cached:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Field risk analysis '{analysis_id}' not found or session expired."
        )
    stats = cached.get("statistics")
    if hasattr(stats, "model_dump"):
        return stats.model_dump()
    return stats

@app.get("/api/v1/analysis/field-risk-map/{analysis_id}/layers")
async def get_field_risk_layers_endpoint(analysis_id: str):
    """
    Retrieves available spatial layer metadata and raster overlays for the analysis.
    """
    cached = field_risk_mapping_service.get_cached_analysis(analysis_id)
    if not cached:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Field risk analysis '{analysis_id}' not found or session expired."
        )
    response_data = cached.get("response", {})
    return {
        "analysis_id": analysis_id,
        "available_layers": response_data.get("available_layers", response_data.get("layers", {})),
        "available_layer_keys": response_data.get("available_layer_keys", []),
        "canonical_grid": response_data.get("canonical_grid"),
        "observation_date": response_data.get("observation_date", cached.get("obs_date")),
    }

@app.post("/api/v1/analysis/field-risk-map/pixel-inspect")
async def inspect_field_pixel_endpoint(payload: PixelInspectionRequest):
    """
    Inspects a specific geographic coordinate (latitude, longitude) or pixel grid (row, col)
    within an active field risk analysis session.
    """
    try:
        result = field_risk_mapping_service.inspect_pixel(
            analysis_id=payload.analysis_id,
            lat=payload.latitude,
            lon=payload.longitude,
            pixel_row=payload.pixel_row,
            pixel_col=payload.pixel_col,
        )
        return result.model_dump()
    except KeyError as ke:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ke),
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error inspecting field pixel: {str(exc)}",
        )

@app.get("/api/v1/analysis/field-risk-map/{analysis_id}/export")
async def export_field_risk_map_endpoint(
    analysis_id: str,
    format: str = Query("geotiff", pattern="^(geotiff|csv|json|png)$", description="Export format: geotiff, csv, json, or png")
):
    """
    Exports field risk analysis to multi-band GeoTIFF, summary CSV, comprehensive JSON, or map PNG.
    """
    try:
        content_bytes, media_type, filename = field_risk_mapping_service.export_analysis(
            analysis_id=analysis_id,
            export_format=format,
        )
        return Response(
            content=content_bytes,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except KeyError as ke:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ke),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error exporting field risk analysis: {str(exc)}",
        )

# =============================================================================
# Phase 17: Unified Farmer Intelligence Dashboard Endpoints
# =============================================================================

@app.get("/api/v1/farmer-dashboard/overview")
async def get_farmer_dashboard_overview():
    """
    Returns high-level summary cards for all registered fields in the platform.
    """
    try:
        overview = await farmer_dashboard_service.get_overview()
        return overview.model_dump()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving dashboard overview: {str(exc)}",
        )

@app.get("/api/v1/farmer-dashboard/field/{field_id}")
async def get_farmer_dashboard_field(
    field_id: str,
    prefer_production: bool = Query(True, description="Prefer production models if available"),
    force_refresh: bool = Query(False, description="Force recomputation of analysis cache"),
):
    """
    Returns the comprehensive, unified agricultural intelligence dashboard for a single field.
    Aggregates overview, crop condition, vegetation metrics with deltas, early stress intelligence,
    growth forecast with prediction intervals, spatial risk distribution, historical timeline,
    non-prescriptive recommendations, and data reliability metrics.
    """
    try:
        dashboard = await farmer_dashboard_service.get_field_dashboard(
            field_id=field_id,
            prefer_production=prefer_production,
            force_refresh=force_refresh,
        )
        return dashboard.model_dump()
    except FieldNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating field dashboard: {str(exc)}",
        )

@app.get("/api/v1/farmer-dashboard/field/{field_id}/recommendations")
async def get_farmer_field_recommendations(field_id: str):
    """
    Returns evidence-linked, non-prescriptive agronomic recommendations for the specified field.
    """
    try:
        recommendations = await farmer_dashboard_service.get_field_recommendations(field_id=field_id)
        return [r.model_dump() for r in recommendations]
    except FieldNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving field recommendations: {str(exc)}",
        )

@app.get("/api/v1/farmer-dashboard/compare")
async def compare_farmer_fields(
    fields: str = Query(..., description="Comma-separated field IDs, e.g. 'field_1,field_2'"),
):
    """
    Compares two or more agricultural fields with temporal alignment validation.
    Issues warnings if observation dates differ by more than 10 days.
    """
    field_id_list = [f.strip() for f in fields.split(",") if f.strip()]
    if len(field_id_list) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field comparison requires at least 2 comma-separated field IDs.",
        )

    try:
        comparison = await farmer_dashboard_service.compare_fields(field_ids=field_id_list)
        return comparison.model_dump()
    except FieldNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf),
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error comparing fields: {str(exc)}",
        )

@app.post("/api/v1/farmer-dashboard/field")
async def register_farmer_field(payload: RegisterFieldRequest):
    """
    Registers a new field into the platform catalog.
    """
    try:
        field_overview = farmer_dashboard_service.register_field(payload)
        return field_overview.model_dump()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error registering field: {str(exc)}",
        )

# =============================================================================
# Phase 19: Real-World Field Validation & Ground-Truth Workflow Endpoints
# =============================================================================

from models.field_validation import (
    FieldScoutingObservation,
    LabelVerificationStatus,
    ObserverCredentialType,
)
from services.field_validation_service import field_validation_service

class ReviewObservationRequest(BaseModel):
    reviewer_id: str
    reviewer_credential: ObserverCredentialType
    action: str = Field(..., description="APPROVE, REJECT, or REQUEST_REVISION")
    notes: Optional[str] = None

@app.post("/api/v1/field-validation/observations", status_code=status.HTTP_201_CREATED)
async def record_field_observation(observation: FieldScoutingObservation):
    """
    Registers an in-situ agronomic field observation record under strict quality-control rules.
    Requires informed consent, instrument replicate checks, and credentialed observer identity.
    """
    try:
        result = field_validation_service.record_observation(observation)
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error registering field observation: {str(exc)}",
        )

@app.get("/api/v1/field-validation/observations")
async def list_field_observations(
    field_id: Optional[str] = Query(None, description="Filter by field identifier"),
    verification_status: Optional[LabelVerificationStatus] = Query(None, description="Filter by verification status"),
):
    """
    Lists recorded field validation observations with optional field and verification status filtering.
    """
    try:
        observations = field_validation_service.list_observations(
            field_id=field_id,
            verification_status=verification_status,
        )
        return {"status": "success", "count": len(observations), "observations": observations}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing field observations: {str(exc)}",
        )

@app.post("/api/v1/field-validation/observations/{observation_id}/review")
async def review_field_observation(
    observation_id: str,
    payload: ReviewObservationRequest,
):
    """
    Executes a double-blind peer review of an in-situ field observation by a credentialed second agronomist.
    """
    try:
        result = field_validation_service.review_observation(
            observation_id=observation_id,
            reviewer_id=payload.reviewer_id,
            reviewer_credential=payload.reviewer_credential,
            action=payload.action,
            notes=payload.notes,
        )
        return result
    except KeyError as ke:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ke),
        )
    except (ValueError, PermissionError) as pe:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(pe),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reviewing observation: {str(exc)}",
        )

@app.get("/api/v1/field-validation/provenance")
async def get_field_validation_provenance():
    """
    Returns ground-truth dataset provenance, current sample counts, and production training eligibility.
    """
    try:
        return field_validation_service.get_dataset_provenance()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving provenance: {str(exc)}",
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)



