"""
backend/services/provenance.py:
Standardized provenance badge builder across all Crop Health AI satellite workflows.
Provides clear attribution:
- Live vs. Demo Mode status
- Constellation / Sensor origin
- Acquisition dates & temporal matching interval
- Cloud coverage & valid pixel percentage
- Scene IDs and processing levels
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone


def build_provenance_badge(
    is_demo: bool,
    sensor: str = "Copernicus Earth Observation",
    optical_date: Optional[str] = None,
    sar_date: Optional[str] = None,
    temporal_gap_days: Optional[int] = None,
    cloud_coverage_percentage: Optional[float] = None,
    valid_pixel_percentage: Optional[float] = None,
    scene_id: Optional[str] = None,
    orbit_direction: Optional[str] = None,
    relative_orbit: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Constructs a standardized provenance payload suitable for frontend badges and publication tables.
    """
    mode = "DEMO" if is_demo else "LIVE"
    now_utc = datetime.now(timezone.utc).isoformat()

    acquisition_dates = {}
    if optical_date:
        acquisition_dates["optical_date"] = optical_date
    if sar_date:
        acquisition_dates["sar_date"] = sar_date
    if temporal_gap_days is not None:
        acquisition_dates["temporal_gap_days"] = temporal_gap_days

    badge = {
        "is_demo": bool(is_demo),
        "mode": mode,
        "badge_label": f"DEMO SIMULATION" if is_demo else "VERIFIED LIVE COPERNICUS",
        "badge_color": "amber" if is_demo else "emerald",
        "data_source": f"{sensor} ({'Simulated Demo' if is_demo else 'Live Stream'})",
        "sensor": sensor,
        "scene_id": scene_id or ("DEMO_SYNTHETIC_SCENE" if is_demo else "LIVE_ACQUISITION"),
        "acquisition_dates": acquisition_dates,
        "cloud_coverage_percentage": round(float(cloud_coverage_percentage), 2) if cloud_coverage_percentage is not None else None,
        "valid_pixel_percentage": round(float(valid_pixel_percentage), 2) if valid_pixel_percentage is not None else None,
        "orbit_direction": orbit_direction,
        "relative_orbit": relative_orbit,
        "timestamp_utc": now_utc,
    }

    if extra:
        badge["details"] = extra

    return badge

