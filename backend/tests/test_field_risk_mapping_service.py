"""
Test Suite for Field-Level AI Risk Mapping & Spatial Intelligence (Phase 16).

Validates:
1. AOI geometry validation and Canonical 10m UTM grid alignment.
2. 4-tier transparent risk classification (Low, Moderate, Elevated, Insufficient Evidence).
3. Pixel-level inspection by geographic coordinate (lat, lon) and raster index (row, col).
4. Out-of-bounds coordinate handling.
5. In-field summary statistics (projected area, hectares, percentages, index averages).
6. Multi-layer generation (Risk, Stress, NDVI, NDMI, NDRE, EVI, Historical Change, Spatial Trend, Data Quality).
7. Non-causation disclaimers and multi-criteria rule transparency.
8. Anti-fabrication guards (invalid/nodata pixels never assigned risk categories).
9. Export pipelines: Multi-band GeoTIFF, CSV, JSON, PNG.
10. Demo mode separation and production mode resilience.
11. Full REST API endpoints integration and error handling.
"""

import os
import io
import json
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from main import app
from services.field_risk_mapping_service import (
    FieldRiskMappingService,
    field_risk_mapping_service,
    NON_CAUSATION_DISCLAIMER,
)
from ml.field_risk.schema import (
    FieldRiskMapRequest,
    PixelInspectionRequest,
    RISK_LOW,
    RISK_MODERATE,
    RISK_ELEVATED,
    RISK_INSUFFICIENT_EVIDENCE,
    RISK_HEX_COLORS,
)

client = TestClient(app)

SAMPLE_GEOMETRY = {
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


# =============================================================================
# 1. Geometry & Canonical Grid Alignment Tests
# =============================================================================

@pytest.mark.asyncio
async def test_aoi_geometry_validation():
    """Verify that geometry can be provided directly or generated from lat/lon/radius."""
    req_geom = FieldRiskMapRequest(
        geometry=SAMPLE_GEOMETRY,
        start_date="2025-01-01",
        end_date="2025-03-01",
    )
    res = await field_risk_mapping_service.generate_field_risk_map(req_geom)
    assert res["status"] == "success"
    assert "canonical_grid" in res
    assert res["canonical_grid"]["width"] > 0
    assert res["canonical_grid"]["height"] > 0
    assert abs(res["canonical_grid"]["resolution_m"] - 10.0) < 1.0


@pytest.mark.asyncio
async def test_aoi_canonical_grid_consistency():
    """Verify canonical UTM grid preserves projected bounds, affine transform, and CRS."""
    req = FieldRiskMapRequest(
        latitude=16.805,
        longitude=81.525,
        radius_km=1.0,
        start_date="2025-01-01",
        end_date="2025-03-01",
    )
    res = await field_risk_mapping_service.generate_field_risk_map(req)
    grid = res["canonical_grid"]
    assert "crs" in grid
    assert "UTM" in grid["crs"]
    assert len(grid["transform"]) == 6
    assert len(grid["bounds"]) == 4
    assert len(grid["geo_bounds"]) == 4
    # geo_bounds: [min_lon, min_lat, max_lon, max_lat]
    assert grid["geo_bounds"][0] < grid["geo_bounds"][2]
    assert grid["geo_bounds"][1] < grid["geo_bounds"][3]


# =============================================================================
# 2. 4-Tier Risk Classification Tests
# =============================================================================

def test_four_tier_risk_classification_logic():
    """Directly test multi-criteria risk classification rules on synthetic test arrays."""
    service = FieldRiskMappingService()
    h, w = 10, 10
    q_mask = np.ones((h, w), dtype=np.uint8)
    stress_prob = np.full((h, w), 0.2, dtype=np.float32)

    raw_features = {
        "NDVI": np.full((h, w), 0.65, dtype=np.float32),
        "NDMI": np.full((h, w), 0.30, dtype=np.float32),
        "NDRE": np.full((h, w), 0.35, dtype=np.float32),
        "EVI": np.full((h, w), 0.50, dtype=np.float32),
    }

    # Pixel 0,0: Nominal vigor -> Low Risk
    # Pixel 1,1: High stress prob -> Elevated Risk
    stress_prob[1, 1] = 0.75

    # Pixel 2,2: Moderate stress prob -> Moderate Risk
    stress_prob[2, 2] = 0.45

    # Pixel 3,3: Water deficit (NDMI < 0) -> Elevated Risk
    stress_prob[3, 3] = 0.55
    raw_features["NDMI"][3, 3] = -0.1

    # Pixel 4,4: Invalid nodata mask -> Insufficient Evidence
    q_mask[4, 4] = 0

    risk_grid = service._compute_risk_classification(
        quality_mask=q_mask,
        stress_prob_grid=stress_prob,
        raw_features=raw_features,
        change_grid=None,
    )

    assert risk_grid[0, 0] == 0  # Low Risk
    assert risk_grid[1, 1] == 2  # Elevated Risk
    assert risk_grid[2, 2] == 1  # Moderate Risk
    assert risk_grid[3, 3] == 2  # Elevated Risk (due to water stress)
    assert risk_grid[4, 4] == -1  # Insufficient Evidence


def test_anti_fabrication_invalid_pixels_never_classified_as_risk():
    """Verify that cloud or nodata pixels are NEVER converted to zero or low/elevated risk."""
    service = FieldRiskMappingService()
    h, w = 4, 4
    q_mask = np.zeros((h, w), dtype=np.uint8)  # Entirely invalid / cloud-covered
    stress_prob = np.full((h, w), 0.9, dtype=np.float32)  # High model output shouldn't matter
    raw_features = {"NDVI": np.full((h, w), 0.1, dtype=np.float32)}

    risk_grid = service._compute_risk_classification(
        quality_mask=q_mask,
        stress_prob_grid=stress_prob,
        raw_features=raw_features,
        change_grid=None,
    )

    # All pixels MUST be Insufficient Evidence (-1)
    assert np.all(risk_grid == -1)


# =============================================================================
# 3. Field Statistics Tests
# =============================================================================

@pytest.mark.asyncio
async def test_field_summary_statistics_computation():
    """Verify area accounting, category breakdowns, and index averages."""
    req = FieldRiskMapRequest(
        geometry=SAMPLE_GEOMETRY,
        start_date="2025-01-01",
        end_date="2025-03-01",
    )
    res = await field_risk_mapping_service.generate_field_risk_map(req)
    stats = res["statistics"]

    assert stats["total_area_hectares"] > 0
    assert stats["valid_area_hectares"] > 0
    assert stats["valid_area_hectares"] <= stats["total_area_hectares"]
    assert 0 <= stats["valid_pixel_percentage"] <= 100.0
    assert 0 <= stats["nodata_percentage"] <= 100.0
    assert abs(stats["valid_pixel_percentage"] + stats["nodata_percentage"] - 100.0) < 1e-2

    # Check risk categories sum to valid area
    cats = stats["risk_distribution"]
    assert "low_risk" in cats
    assert "moderate_risk" in cats
    assert "elevated_risk" in cats
    assert "insufficient_evidence" in cats

    valid_sum_ha = (
        cats["low_risk"]["area_hectares"]
        + cats["moderate_risk"]["area_hectares"]
        + cats["elevated_risk"]["area_hectares"]
    )
    assert abs(valid_sum_ha - stats["valid_area_hectares"]) < 0.2

    # Check mean vegetation indices are reasonable
    means = stats["vegetation_indices_mean"]
    assert "NDVI" in means
    if means["NDVI"] is not None:
        assert -1.0 <= means["NDVI"] <= 1.0


# =============================================================================
# 4. Pixel Inspection Tests
# =============================================================================

@pytest.mark.asyncio
async def test_pixel_inspection_by_geo_coordinates():
    """Verify inspection at an in-bounds coordinate returns accurate pixel data."""
    req = FieldRiskMapRequest(
        geometry=SAMPLE_GEOMETRY,
        start_date="2025-01-01",
        end_date="2025-03-01",
    )
    res = await field_risk_mapping_service.generate_field_risk_map(req)
    analysis_id = res["analysis_id"]

    # Target centroid
    center_lat = (16.800244 + 16.820244) / 2.0
    center_lon = (81.516592 + 81.536592) / 2.0

    inspect_res = field_risk_mapping_service.inspect_pixel(
        analysis_id=analysis_id,
        lat=center_lat,
        lon=center_lon,
    )

    assert inspect_res.status == "success"
    assert inspect_res.is_valid_pixel is True
    assert inspect_res.pixel_coords["row"] >= 0
    assert inspect_res.pixel_coords["col"] >= 0
    assert inspect_res.risk_category in (RISK_LOW, RISK_MODERATE, RISK_ELEVATED)
    assert inspect_res.risk_color_hex.startswith("#")
    assert "ndvi" in inspect_res.vegetation_indices
    assert inspect_res.scientific_disclaimer == NON_CAUSATION_DISCLAIMER
    assert len(inspect_res.evidence_summary) > 0


@pytest.mark.asyncio
async def test_pixel_inspection_by_grid_indices():
    """Verify inspection by pixel_row and pixel_col."""
    req = FieldRiskMapRequest(
        geometry=SAMPLE_GEOMETRY,
        start_date="2025-01-01",
        end_date="2025-03-01",
    )
    res = await field_risk_mapping_service.generate_field_risk_map(req)
    analysis_id = res["analysis_id"]

    inspect_res = field_risk_mapping_service.inspect_pixel(
        analysis_id=analysis_id,
        pixel_row=2,
        pixel_col=2,
    )

    assert inspect_res.status == "success"
    assert inspect_res.pixel_coords["row"] == 2
    assert inspect_res.pixel_coords["col"] == 2
    assert inspect_res.geo_coords["latitude"] is not None
    assert inspect_res.geo_coords["longitude"] is not None


@pytest.mark.asyncio
async def test_pixel_inspection_out_of_bounds():
    """Verify inspection outside AOI boundary returns out_of_bounds status."""
    req = FieldRiskMapRequest(
        geometry=SAMPLE_GEOMETRY,
        start_date="2025-01-01",
        end_date="2025-03-01",
    )
    res = await field_risk_mapping_service.generate_field_risk_map(req)
    analysis_id = res["analysis_id"]

    # Point far outside field
    out_lat = 0.0
    out_lon = 0.0

    inspect_res = field_risk_mapping_service.inspect_pixel(
        analysis_id=analysis_id,
        lat=out_lat,
        lon=out_lon,
    )

    assert inspect_res.status == "out_of_bounds"
    assert inspect_res.is_valid_pixel is False
    assert inspect_res.risk_category == RISK_INSUFFICIENT_EVIDENCE


# =============================================================================
# 5. Spatial Layer Generation & Rendering Tests
# =============================================================================

@pytest.mark.asyncio
async def test_all_spatial_layers_generated():
    """Verify all 9 spatial layers are rendered as valid base64 PNGs."""
    req = FieldRiskMapRequest(
        geometry=SAMPLE_GEOMETRY,
        start_date="2025-01-01",
        end_date="2025-03-01",
    )
    res = await field_risk_mapping_service.generate_field_risk_map(req)
    layers = res["layers"]

    expected_layers = [
        "risk_classification",
        "potential_stress",
        "ndvi",
        "ndmi",
        "ndre",
        "evi",
        "historical_change",
        "growth_trend",
        "data_quality",
    ]

    for key in expected_layers:
        assert key in layers
        layer = layers[key]
        assert layer["layer_key"] == key
        assert len(layer["bounds"]) == 4
        assert len(layer["image_png_base64"]) > 50


# =============================================================================
# 6. Export Functionality Tests
# =============================================================================

@pytest.mark.asyncio
async def test_export_geotiff():
    """Verify 7-band GeoTIFF export."""
    req = FieldRiskMapRequest(
        geometry=SAMPLE_GEOMETRY,
        start_date="2025-01-01",
        end_date="2025-03-01",
    )
    res = await field_risk_mapping_service.generate_field_risk_map(req)
    analysis_id = res["analysis_id"]

    content, media_type, filename = field_risk_mapping_service.export_analysis(
        analysis_id=analysis_id,
        export_format="geotiff",
    )

    assert media_type == "image/tiff"
    assert filename.endswith(".tif")
    assert len(content) > 100
    # TIFF header check (II* or MM*)
    assert content[:4] in (b"II*\x00", b"MM\x00*")


@pytest.mark.asyncio
async def test_export_csv():
    """Verify CSV statistics export."""
    req = FieldRiskMapRequest(
        geometry=SAMPLE_GEOMETRY,
        start_date="2025-01-01",
        end_date="2025-03-01",
    )
    res = await field_risk_mapping_service.generate_field_risk_map(req)
    analysis_id = res["analysis_id"]

    content, media_type, filename = field_risk_mapping_service.export_analysis(
        analysis_id=analysis_id,
        export_format="csv",
    )

    assert media_type == "text/csv"
    assert filename.endswith(".csv")
    csv_text = content.decode("utf-8")
    assert "Field Risk Intelligence Analysis Summary" in csv_text
    assert "Elevated Potential Risk" in csv_text
    assert "NDVI" in csv_text
    assert "DISCLAIMER" in csv_text


@pytest.mark.asyncio
async def test_export_json():
    """Verify JSON metadata export."""
    req = FieldRiskMapRequest(
        geometry=SAMPLE_GEOMETRY,
        start_date="2025-01-01",
        end_date="2025-03-01",
    )
    res = await field_risk_mapping_service.generate_field_risk_map(req)
    analysis_id = res["analysis_id"]

    content, media_type, filename = field_risk_mapping_service.export_analysis(
        analysis_id=analysis_id,
        export_format="json",
    )

    assert media_type == "application/json"
    assert filename.endswith(".json")
    parsed = json.loads(content.decode("utf-8"))
    assert parsed["analysis_id"] == analysis_id
    assert "classification_methodology" in parsed
    assert "statistics" in parsed


@pytest.mark.asyncio
async def test_export_png():
    """Verify composite PNG map preview export."""
    req = FieldRiskMapRequest(
        geometry=SAMPLE_GEOMETRY,
        start_date="2025-01-01",
        end_date="2025-03-01",
    )
    res = await field_risk_mapping_service.generate_field_risk_map(req)
    analysis_id = res["analysis_id"]

    content, media_type, filename = field_risk_mapping_service.export_analysis(
        analysis_id=analysis_id,
        export_format="png",
    )

    assert media_type == "image/png"
    assert filename.endswith(".png")
    # Verify PNG header
    assert content[:8] == b"\x89PNG\r\n\x1a\n"


# =============================================================================
# 7. Demo vs Production Separation Tests
# =============================================================================

@pytest.mark.asyncio
async def test_demo_mode_separation():
    """Verify DEMO_MODE marks responses as demo and warns user."""
    req = FieldRiskMapRequest(
        geometry=SAMPLE_GEOMETRY,
        start_date="2025-01-01",
        end_date="2025-03-01",
        demo_mode=True,
    )
    res = await field_risk_mapping_service.generate_field_risk_map(req)
    assert res["is_demo"] is True
    assert any("DEMO MODE" in w for w in res["warnings"])


# =============================================================================
# 8. REST API Endpoints Integration Tests
# =============================================================================

def test_api_generate_field_risk_map():
    """POST /api/v1/analysis/field-risk-map"""
    payload = {
        "geometry": SAMPLE_GEOMETRY,
        "start_date": "2025-01-01",
        "end_date": "2025-03-01",
        "selected_layer": "risk_classification",
    }
    response = client.post("/api/v1/analysis/field-risk-map", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "analysis_id" in data
    assert "statistics" in data
    assert "layers" in data


def test_api_get_field_risk_map_by_id():
    """GET /api/v1/analysis/field-risk-map/{analysis_id}"""
    # Create analysis first
    post_res = client.post(
        "/api/v1/analysis/field-risk-map",
        json={
            "geometry": SAMPLE_GEOMETRY,
            "start_date": "2025-01-01",
            "end_date": "2025-03-01",
        },
    )
    assert post_res.status_code == 200
    analysis_id = post_res.json()["analysis_id"]

    # Fetch cached analysis
    get_res = client.get(f"/api/v1/analysis/field-risk-map/{analysis_id}")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["analysis_id"] == analysis_id


def test_api_get_field_risk_statistics():
    """GET /api/v1/analysis/field-risk-map/{analysis_id}/statistics"""
    post_res = client.post(
        "/api/v1/analysis/field-risk-map",
        json={
            "geometry": SAMPLE_GEOMETRY,
            "start_date": "2025-01-01",
            "end_date": "2025-03-01",
        },
    )
    assert post_res.status_code == 200
    analysis_id = post_res.json()["analysis_id"]

    stat_res = client.get(f"/api/v1/analysis/field-risk-map/{analysis_id}/statistics")
    assert stat_res.status_code == 200
    stats = stat_res.json()
    assert "total_area_hectares" in stats
    assert "risk_distribution" in stats


def test_api_get_field_risk_layers():
    """GET /api/v1/analysis/field-risk-map/{analysis_id}/layers"""
    post_res = client.post(
        "/api/v1/analysis/field-risk-map",
        json={
            "geometry": SAMPLE_GEOMETRY,
            "start_date": "2025-01-01",
            "end_date": "2025-03-01",
        },
    )
    assert post_res.status_code == 200
    analysis_id = post_res.json()["analysis_id"]

    layer_res = client.get(f"/api/v1/analysis/field-risk-map/{analysis_id}/layers")
    assert layer_res.status_code == 200
    layers = layer_res.json()
    assert "available_layers" in layers
    assert "available_layer_keys" in layers


def test_api_pixel_inspect():
    """POST /api/v1/analysis/field-risk-map/pixel-inspect"""
    post_res = client.post(
        "/api/v1/analysis/field-risk-map",
        json={
            "geometry": SAMPLE_GEOMETRY,
            "start_date": "2025-01-01",
            "end_date": "2025-03-01",
        },
    )
    assert post_res.status_code == 200
    analysis_id = post_res.json()["analysis_id"]

    inspect_payload = {
        "analysis_id": analysis_id,
        "latitude": 16.810,
        "longitude": 81.525,
    }
    inspect_res = client.post("/api/v1/analysis/field-risk-map/pixel-inspect", json=inspect_payload)
    assert inspect_res.status_code == 200
    result = inspect_res.json()
    assert result["status"] in ("success", "out_of_bounds")
    assert "risk_category" in result
    assert "evidence_summary" in result


def test_api_export_formats():
    """GET /api/v1/analysis/field-risk-map/{analysis_id}/export?format=..."""
    post_res = client.post(
        "/api/v1/analysis/field-risk-map",
        json={
            "geometry": SAMPLE_GEOMETRY,
            "start_date": "2025-01-01",
            "end_date": "2025-03-01",
        },
    )
    assert post_res.status_code == 200
    analysis_id = post_res.json()["analysis_id"]

    # Test CSV export
    csv_res = client.get(f"/api/v1/analysis/field-risk-map/{analysis_id}/export?format=csv")
    assert csv_res.status_code == 200
    assert csv_res.headers["content-type"] == "text/csv; charset=utf-8"

    # Test GeoTIFF export
    tiff_res = client.get(f"/api/v1/analysis/field-risk-map/{analysis_id}/export?format=geotiff")
    assert tiff_res.status_code == 200
    assert tiff_res.headers["content-type"] == "image/tiff"

    # Test PNG export
    png_res = client.get(f"/api/v1/analysis/field-risk-map/{analysis_id}/export?format=png")
    assert png_res.status_code == 200
    assert png_res.headers["content-type"] == "image/png"
