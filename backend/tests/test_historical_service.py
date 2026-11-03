import pytest
from fastapi.testclient import TestClient
import numpy as np

from main import app
from services.historical_service import (
    historical_monitoring_service,
    HistoricalMonitoringService,
    HistoricalMonitoringError,
)

client = TestClient(app)

SAMPLE_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [
            [78.4800, 17.3800],
            [78.4900, 17.3800],
            [78.4900, 17.3900],
            [78.4800, 17.3900],
            [78.4800, 17.3800],
        ]
    ],
}

def test_classify_trend():
    service = HistoricalMonitoringService()
    # Tolerance default 0.03
    assert service.classify_trend(0.05, tolerance=0.03) == "Increasing"
    assert service.classify_trend(-0.04, tolerance=0.03) == "Decreasing"
    assert service.classify_trend(0.02, tolerance=0.03) == "Stable"
    assert service.classify_trend(-0.02, tolerance=0.03) == "Stable"
    assert service.classify_trend(0.00, tolerance=0.03) == "Stable"

    # Custom tolerance
    assert service.classify_trend(0.04, tolerance=0.05) == "Stable"
    assert service.classify_trend(0.06, tolerance=0.05) == "Increasing"

def test_assess_quality():
    service = HistoricalMonitoringService()
    assert service.assess_quality(95.0, 5.0) == "Excellent"
    assert service.assess_quality(80.0, 15.0) == "Good"
    assert service.assess_quality(60.0, 25.0) == "Fair"
    assert service.assess_quality(40.0, 10.0) == "Limited valid observations"

def test_temporal_sampling_filter():
    service = HistoricalMonitoringService()
    raw_scenes = [
        {"id": "S1", "properties": {"datetime": "2026-07-01T10:00:00Z", "eo:cloud_cover": 18.0}},
        {"id": "S2", "properties": {"datetime": "2026-07-02T10:00:00Z", "eo:cloud_cover": 4.0}},  # Cluster 1 winner
        {"id": "S3", "properties": {"datetime": "2026-07-15T10:00:00Z", "eo:cloud_cover": 12.0}},
        {"id": "S4", "properties": {"datetime": "2026-07-16T10:00:00Z", "eo:cloud_cover": 2.5}},  # Cluster 2 winner
        {"id": "S5", "properties": {"datetime": "2026-08-01T10:00:00Z", "eo:cloud_cover": 8.0}},   # Cluster 3 winner
    ]
    filtered = service.temporal_sampling_filter(raw_scenes, min_cluster_days=3)
    assert len(filtered) == 3
    # Check that lowest cloud scenes were selected
    ids = [s["id"] for s in filtered]
    assert ids == ["S2", "S4", "S5"]

def test_deterministic_demo_history():
    service = HistoricalMonitoringService()
    start_date = "2026-06-01"
    end_date = "2026-09-01"

    run1 = service.generate_deterministic_demo_observations(start_date, end_date)
    run2 = service.generate_deterministic_demo_observations(start_date, end_date)

    assert run1["status"] == "success"
    assert run1["is_demo"] is True
    assert run1["observation_count"] == run2["observation_count"]
    assert len(run1["observations"]) >= 4

    # Verify identical values between runs
    for o1, o2 in zip(run1["observations"], run2["observations"]):
        assert o1["date"] == o2["date"]
        assert o1["cloud_coverage"] == o2["cloud_coverage"]
        assert o1["ndvi"]["mean"] == o2["ndvi"]["mean"]
        assert o1["ndmi"]["mean"] == o2["ndmi"]["mean"]
        assert o1["ndre"]["mean"] == o2["ndre"]["mean"]
        assert o1["evi"]["mean"] == o2["evi"]["mean"]
        assert o1["preview_map"].startswith("data:image/png;base64,")

    # Verify chronological ordering
    dates = [o["date"] for o in run1["observations"]]
    assert dates == sorted(dates)

def test_trend_summary_calculation():
    service = HistoricalMonitoringService()
    observations = [
        {
            "date": "2026-06-01",
            "ndvi": {"mean": 0.40},
            "ndmi": {"mean": 0.10},
            "ndre": {"mean": 0.20},
            "evi": {"mean": 0.30},
        },
        {
            "date": "2026-07-01",
            "ndvi": {"mean": 0.55},
            "ndmi": {"mean": 0.22},
            "ndre": {"mean": 0.31},
            "evi": {"mean": 0.42},
        },
    ]
    summary = service.calculate_trend_summary(observations, tolerance=0.03)
    assert summary["observation_count"] == 2
    assert summary["indices"]["ndvi"]["trend"] == "Increasing"
    assert summary["indices"]["ndvi"]["absolute_change"] == pytest.approx(0.15, 0.001)
    assert summary["indices"]["ndvi"]["percentage_change"] == pytest.approx(37.5, 0.1)
    assert "disclaimer" in summary
    assert "scientific_interpretation" in summary

def test_pairwise_comparison_and_change_map():
    service = HistoricalMonitoringService()
    start_date = "2026-05-01"
    end_date = "2026-08-01"
    demo_result = service.generate_deterministic_demo_observations(start_date, end_date)
    session_id = demo_result["session_id"]
    obs_list = demo_result["observations"]

    obs_a_id = obs_list[0]["observation_id"]
    obs_b_id = obs_list[-1]["observation_id"]

    comparison = service.compare_two_observations(session_id, obs_a_id, obs_b_id)
    assert comparison["status"] == "success"
    assert "deltas" in comparison
    assert "change_map" in comparison
    assert comparison["change_map"].startswith("data:image/png;base64,")
    assert "change_breakdown" in comparison
    breakdown = comparison["change_breakdown"]
    total_pct = breakdown["lower_signal_pct"] + breakdown["minimal_change_pct"] + breakdown["higher_signal_pct"]
    assert pytest.approx(total_pct, 1.0) == 100.0

def test_api_historical_endpoints():
    payload = {
        "geometry": SAMPLE_GEOMETRY,
        "start_date": "2026-06-01",
        "end_date": "2026-09-01",
        "max_cloud_coverage": 30.0,
        "tolerance": 0.03,
    }
    response = client.post("/api/v1/analysis/history", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "observations" in data
    assert len(data["observations"]) > 0
    assert "trend_summary" in data
    session_id = data["session_id"]

    # Test Compare endpoint
    obs_a = data["observations"][0]["observation_id"]
    obs_b = data["observations"][-1]["observation_id"]
    comp_res = client.post(
        "/api/v1/analysis/history/compare",
        json={
            "session_id": session_id,
            "observation_a_id": obs_a,
            "observation_b_id": obs_b,
        },
    )
    assert comp_res.status_code == 200
    comp_data = comp_res.json()
    assert comp_data["status"] == "success"
    assert "change_map" in comp_data
    assert "deltas" in comp_data
