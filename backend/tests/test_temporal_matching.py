"""
Unit tests for the Temporal Matching Engine and SAR Decibel Difference Ratio.
Tests:
- ±3-day temporal window enforcement (0, 1, 2, 3 days pass; 4+ days fail)
- Nearest Sentinel-1 acquisition selection
- Tie-breaking on observation quality
- Decibel ratio (VV_dB - VH_dB) vs linear power ratio
- Provenance badge structure and values
"""

import pytest
import numpy as np

from services.fusion_service import MultimodalFusionService
from services.sentinel1_service import Sentinel1Service
from services.provenance import build_provenance_badge


def test_temporal_window_enforcement_pass():
    """Validates that S1 candidates within ±3 days match successfully."""
    fusion_svc = MultimodalFusionService()
    s2_date = "2024-01-10T05:20:00Z"

    # Candidate 2 days away
    catalog = [
        {"scene_id": "S1A_01", "acquisition_date": "2024-01-08T00:38:00Z", "valid_percentage": 95.0},
        {"scene_id": "S1A_02", "acquisition_date": "2024-01-13T00:38:00Z", "valid_percentage": 98.0},
    ]

    result = fusion_svc.match_temporal_observations(
        sentinel2_date=s2_date,
        sentinel1_scenes=catalog,
        temporal_window_days=3,
    )

    assert result["matched"] is True
    assert result["sar_available"] is True
    # 2024-01-08 is 2 days away, 2024-01-13 is 3 days away -> nearest is 2024-01-08
    assert result["sentinel1_date"] == "2024-01-08"
    assert result["temporal_gap_days"] == 2
    assert result["temporal_match_quality"] == "GOOD"
    assert result["matched_scene"]["scene_id"] == "S1A_01"


def test_temporal_window_enforcement_fail():
    """Validates that S1 candidates strictly outside ±3 days are rejected without fabrication."""
    fusion_svc = MultimodalFusionService()
    s2_date = "2024-01-10T05:20:00Z"

    # Candidates 4 days and 16 days away
    catalog = [
        {"scene_id": "S1A_04", "acquisition_date": "2024-01-06T00:38:00Z", "valid_percentage": 95.0},
        {"scene_id": "S1A_16", "acquisition_date": "2024-01-26T00:38:00Z", "valid_percentage": 98.0},
    ]

    result = fusion_svc.match_temporal_observations(
        sentinel2_date=s2_date,
        sentinel1_scenes=catalog,
        temporal_window_days=3,
    )

    assert result["matched"] is False
    assert result["sar_available"] is False
    assert result["sentinel1_date"] is None
    assert result["temporal_gap_days"] is None
    assert result["temporal_match_quality"] == "UNAVAILABLE"
    assert result["matched_scene"] is None


def test_temporal_nearest_and_tie_breaker():
    """Tests selection of nearest candidate and tie-breaking by quality score."""
    fusion_svc = MultimodalFusionService()
    s2_date = "2024-01-10T05:20:00Z"

    # Two scenes with exact same gap (1 day: 01-09 vs 01-11)
    catalog = [
        {"scene_id": "S1A_DAY_BEFORE", "acquisition_date": "2024-01-09T00:38:00Z", "valid_percentage": 82.0},
        {"scene_id": "S1A_DAY_AFTER", "acquisition_date": "2024-01-11T00:38:00Z", "valid_percentage": 99.0},
    ]

    result = fusion_svc.match_temporal_observations(
        sentinel2_date=s2_date,
        sentinel1_scenes=catalog,
        temporal_window_days=3,
    )

    assert result["matched"] is True
    assert result["temporal_gap_days"] == 1
    # Tie-breaker should pick S1A_DAY_AFTER due to 99.0% vs 82.0% quality
    assert result["matched_scene"]["scene_id"] == "S1A_DAY_AFTER"


def test_sar_decibel_difference_ratio():
    """
    Validates that the SAR ratio is computed as the decibel difference (VV_dB - VH_dB)
    and that linear power ratio is also calculated and statistically consistent.
    """
    s1_svc = Sentinel1Service()

    # Create synthetic linear backscatter:
    # VV_dB = -10 dB -> linear = 10^(-10/10) = 0.10
    # VH_dB = -18 dB -> linear = 10^(-18/10) = 0.0158489
    # Expected VV_dB - VH_dB = -10 - (-18) = +8.0 dB
    # Expected Linear Ratio = 0.10 / 0.0158489 = 6.30957 -> 10*log10(6.30957) = 8.0 dB
    vv_linear = np.full((10, 10), 0.10, dtype=np.float32)
    vh_linear = np.full((10, 10), 10.0 ** (-1.8), dtype=np.float32)
    mask = np.ones((10, 10), dtype=bool)

    features = s1_svc.compute_sar_features(vv_linear, vh_linear, mask)

    vv_stats = features["features"]["vv"]["statistics"]
    vh_stats = features["features"]["vh"]["statistics"]
    ratio_stats = features["features"]["vv_vh_ratio"]["statistics"]
    ratio_lin_stats = features["features"]["vv_vh_ratio"]["linear_statistics"]

    assert np.isclose(vv_stats["mean"], -10.0, atol=0.01)
    assert np.isclose(vh_stats["mean"], -18.0, atol=0.01)
    assert np.isclose(ratio_stats["mean"], 8.0, atol=0.01)
    assert np.isclose(ratio_lin_stats["mean"], 6.31, atol=0.02)
    assert features["features"]["vv_vh_ratio"]["unit"] == "dB difference"


def test_provenance_badge_structure():
    """Validates standardized provenance badge generation for live vs demo modes."""
    live_badge = build_provenance_badge(
        is_demo=False,
        sensor="Copernicus Sentinel-2 MSI Level-2A",
        optical_date="2024-01-05T05:11:51Z",
        sar_date="2024-01-08T00:38:15Z",
        temporal_gap_days=3,
        cloud_coverage_percentage=0.74,
        valid_pixel_percentage=98.5,
        scene_id="S2A_MSIL2A_20240105T051151",
    )

    assert live_badge["is_demo"] is False
    assert live_badge["mode"] == "LIVE"
    assert live_badge["badge_label"] == "VERIFIED LIVE COPERNICUS"
    assert live_badge["badge_color"] == "emerald"
    assert live_badge["cloud_coverage_percentage"] == 0.74
    assert live_badge["valid_pixel_percentage"] == 98.5
    assert live_badge["acquisition_dates"]["optical_date"] == "2024-01-05T05:11:51Z"
    assert live_badge["acquisition_dates"]["sar_date"] == "2024-01-08T00:38:15Z"
    assert live_badge["acquisition_dates"]["temporal_gap_days"] == 3

    demo_badge = build_provenance_badge(
        is_demo=True,
        sensor="Copernicus Sentinel-2 MSI Level-2A",
        cloud_coverage_percentage=5.0,
        valid_pixel_percentage=100.0,
    )

    assert demo_badge["is_demo"] is True
    assert demo_badge["mode"] == "DEMO"
    assert demo_badge["badge_label"] == "DEMO SIMULATION"
    assert demo_badge["badge_color"] == "amber"

