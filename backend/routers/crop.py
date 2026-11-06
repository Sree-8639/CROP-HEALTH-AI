"""
backend/routers/crop.py:
FastAPI router for the Crop Vigour Anomaly & Agriculture Intelligence Module.
Mounted under /api/crop.
Shares the Common Data-Access Layer with Deforestation and Heat modules.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from services.data_access import data_access_layer
from services.fusion_service import (
    MultimodalFusionService,
    MultimodalFusionError,
)
from services.farmer_dashboard_service import (
    farmer_dashboard_service,
    FieldNotFoundError,
)
from services.ndvi_service import (
    calculate_ndvi,
    create_valid_pixel_mask,
    calculate_ndvi_statistics,
    generate_ndvi_colormap,
)
from services.ndmi_service import calculate_ndmi
from services.ndre_service import calculate_ndre
from services.evi_service import calculate_evi

# Base directory for metric JSON lookup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

router = APIRouter(tags=["Crop Vigour Anomaly Module"])
fusion_svc = MultimodalFusionService()


class CropAnalyzeRequest(BaseModel):
    geometry: Dict[str, Any] = Field(..., description="GeoJSON Polygon geometry")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    max_cloud_coverage: float = Field(30.0, ge=0.0, le=100.0, description="Max cloud filter %")
    temporal_window_days: int = Field(3, ge=1, le=14, description="S1-S2 matching window in days")
    normalization: str = Field("raw", description="Feature normalization method ('raw', 'min-max', 'z-score')")


@router.get("/status")
async def get_crop_status():
    """Returns operational status for Crop Module and shared data access layer."""
    return {
        "module": "crop",
        "name": "Crop Vigour Anomaly & Multispectral Intelligence",
        "status": "online",
        "data_access_layer": "connected",
        "demo_mode": data_access_layer.is_demo_mode,
        "shared_features": ["Sentinel-2 MSI", "Sentinel-1 SAR", "SCL Cloud Masking", "Tile Cache"],
    }


@router.post("/analyze")
async def analyze_crop(payload: CropAnalyzeRequest):
    """
    Executes end-to-end multimodal crop analysis using the common data access layer:
    - Sentinel-2 optical bands (B02, B03, B04, B05, B08, B11) & SCL cloud masking
    - Sentinel-1 GRD SAR backscatter (VV, VH, VV-VH dB difference ratio)
    - Vegetation Indices (NDVI, NDMI, NDRE, EVI)
    - Unified 3-tier Quality Mask (VALID, PARTIAL, INVALID)
    - Standardized Provenance Badge
    """
    try:
        fusion_result = await fusion_svc.execute_multimodal_fusion(
            geometry=payload.geometry,
            start_date=payload.start_date,
            end_date=payload.end_date,
            max_cloud_coverage=payload.max_cloud_coverage,
            temporal_window_days=payload.temporal_window_days,
            normalization=payload.normalization,
        )
        return fusion_result
    except MultimodalFusionError as mfe:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(mfe))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Crop analysis execution failed: {str(exc)}",
        )


@router.post("/fusion")
async def fuse_crop_data(payload: CropAnalyzeRequest):
    """Alias for /api/crop/analyze conforming to multimodal fusion specification."""
    return await analyze_crop(payload)


@router.get("/overview")
async def get_crop_overview():
    """Returns overview cards of registered agricultural parcels."""
    try:
        overview = await farmer_dashboard_service.get_overview()
        return overview.model_dump()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving crop overview: {str(exc)}",
        )


@router.get("/field/{field_id}")
async def get_crop_field(
    field_id: str,
    prefer_production: bool = Query(True),
    force_refresh: bool = Query(False),
):
    """Returns comprehensive single-field agricultural dashboard."""
    try:
        dashboard = await farmer_dashboard_service.get_field_dashboard(
            field_id=field_id,
            prefer_production=prefer_production,
            force_refresh=force_refresh,
        )
        return dashboard.model_dump()
    except FieldNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating crop field dashboard: {str(exc)}",
        )


@router.get("/field/{field_id}/recommendations")
async def get_crop_field_recommendations(field_id: str):
    """Returns agronomic non-prescriptive scouting recommendations."""
    try:
        recs = await farmer_dashboard_service.get_field_recommendations(field_id=field_id)
        return [r.model_dump() for r in recs]
    except FieldNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving field recommendations: {str(exc)}",
        )


@router.get("/compare")
async def compare_crop_fields(
    fields: Optional[str] = Query(None, description="Comma-separated field IDs"),
    field_ids: Optional[str] = Query(None, description="Alias for fields"),
):
    """Compares two or more agricultural fields with temporal alignment validation."""
    query_str = fields or field_ids
    if not query_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field comparison requires at least 2 comma-separated field IDs in 'fields' query parameter.",
        )
    field_id_list = [f.strip() for f in query_str.split(",") if f.strip()]
    if len(field_id_list) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field comparison requires at least 2 comma-separated field IDs.",
        )
    try:
        comparison = await farmer_dashboard_service.compare_fields(field_ids=field_id_list)
        return comparison.model_dump()
    except FieldNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error comparing fields: {str(exc)}",
        )


@router.get("/metrics")
async def get_crop_metrics():
    """Returns empirical benchmark metrics computed by experiments from results/crop_metrics.json."""
    candidates = [
        os.path.join(BASE_DIR, "results", "crop_metrics.json"),
        os.path.join(BASE_DIR, "backend", "results", "crop_metrics.json"),
        os.path.join(BASE_DIR, "frontend", "public", "crop_metrics.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Crop metrics result file not found. Run experiments/run_crop_experiments.py to generate.",
    )

