"""
ml/crop/chirps_data.py:
Downloads and manages real CHIRPS precipitation data for Tadepalligudem AOI
(Climate Hazards Group InfraRed Precipitation with Station data v2.0).
Provides real daily precipitation time series and cumulative water deficits
for label-free agronomic validation.
"""

import os
import json
import time
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional
import numpy as np

CHIRPS_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
CHIRPS_FILE = os.path.join(CHIRPS_CACHE_DIR, "chirps_tadepalligudem_rainfall.json")


def fetch_or_load_real_chirps_data(
    start_date: str = "2023-06-01",
    end_date: str = "2024-02-29",
    bbox: Optional[List[float]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Retrieves real CHIRPS daily precipitation data for the Tadepalligudem AOI.
    Checks local cache first; if absent, fetches directly via the official ClimateSERV API.
    """
    bbox = bbox or [81.503, 16.793, 81.543, 16.833]
    os.makedirs(CHIRPS_CACHE_DIR, exist_ok=True)

    if not force_refresh and os.path.exists(CHIRPS_FILE):
        try:
            with open(CHIRPS_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if cached.get("data") and len(cached["data"]) > 0:
                    return cached
        except Exception:
            pass

    # Geometry for Tadepalligudem AOI
    min_lon, min_lat, max_lon, max_lat = bbox
    geom = {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]]
    }

    # Format dates as MM/DD/YYYY for ClimateSERV API
    s_dt = start_date.split("-")
    e_dt = end_date.split("-")
    begintime = f"{s_dt[1]}/{s_dt[2]}/{s_dt[0]}"
    endtime = f"{e_dt[1]}/{e_dt[2]}/{e_dt[0]}"

    params = {
        "datatype": 0,  # 0 = CHIRPS
        "begintime": begintime,
        "endtime": endtime,
        "intervaltype": 0,
        "operationtype": 5,  # 5 = Average over polygon
        "geometry": json.dumps(geom),
    }

    print(f"[CHIRPS] Querying ClimateSERV API for real precipitation ({start_date} to {end_date})...")
    url = "https://climateserv.servirglobal.net/api/submitDataRequest/?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CropHealthAI/2.0)"})

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            submit_res = json.loads(resp.read().decode("utf-8"))
            job_id = submit_res[0] if isinstance(submit_res, list) else submit_res.get("uniqueid")

        # Poll for completion (typically takes ~2-6 seconds)
        for _ in range(15):
            time.sleep(2)
            prog_url = f"https://climateserv.servirglobal.net/api/getDataRequestProgress/?id={job_id}"
            prog_req = urllib.request.Request(prog_url, headers={"User-Agent": "Mozilla/5.0 (CropHealthAI/2.0)"})
            with urllib.request.urlopen(prog_req, timeout=10) as pr:
                p_val = json.loads(pr.read().decode("utf-8"))
                if p_val[0] == 100.0:
                    break

        data_url = f"https://climateserv.servirglobal.net/api/getDataFromRequest/?id={job_id}"
        data_req = urllib.request.Request(data_url, headers={"User-Agent": "Mozilla/5.0 (CropHealthAI/2.0)"})
        with urllib.request.urlopen(data_req, timeout=15) as dr:
            raw_payload = json.loads(dr.read().decode("utf-8"))

        daily_records = raw_payload.get("data", [])
        total_precip_mm = sum(float(r["value"]["avg"]) for r in daily_records if r.get("value", {}).get("avg") is not None)

        record = {
            "dataset": "CHIRPS v2.0 (Climate Hazards Center, UC Santa Barbara)",
            "aoi_name": "Tadepalligudem, West Godavari, Andhra Pradesh",
            "bbox": bbox,
            "temporal_range": {"start_date": start_date, "end_date": end_date},
            "total_days": len(daily_records),
            "total_precipitation_mm": round(total_precip_mm, 2),
            "data": daily_records,
        }

        with open(CHIRPS_FILE, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        print(f"[CHIRPS] Successfully downloaded & cached {len(daily_records)} real days of rainfall (Total: {total_precip_mm:.1f} mm).")
        return record

    except Exception as e:
        print(f"[CHIRPS Warning] API query error: {e}. Falling back to cached baseline if present.")
        if os.path.exists(CHIRPS_FILE):
            with open(CHIRPS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        raise e

