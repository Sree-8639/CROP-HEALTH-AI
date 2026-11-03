"""
Data pipeline for multimodal crop stress detection: feature extraction,
spatial-temporal splitting, class imbalance handling, and dataset preparation.
"""

import math
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from ml.stress_detection.schema import (
    ALL_MODEL_FEATURE_NAMES,
    OPTICAL_FEATURES,
    SAR_FEATURES,
    TEMPORAL_FEATURES,
    QUALITY_FEATURES,
    LABEL_MAPPING,
    REVERSE_LABEL_MAPPING,
)

def safe_float(val: Any, default: float = np.nan) -> float:
    """Safely converts any value to float, handling None, dicts, strings, and nan."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default


def get_stat_val(obs: Dict[str, Any], key_upper: str, key_lower: str) -> float:
    """Helper to retrieve index mean value safely from various nested representations."""
    if not isinstance(obs, dict):
        return np.nan

    v = obs.get(key_lower)
    if isinstance(v, dict):
        mean_val = v.get("mean")
        if mean_val is not None:
            return safe_float(mean_val)
    elif v is not None and not isinstance(v, (list, tuple)):
        return safe_float(v)

    indices = obs.get("indices")
    if isinstance(indices, dict):
        ind_v = indices.get(key_lower)
        if isinstance(ind_v, dict):
            stat = ind_v.get("statistics")
            if isinstance(stat, dict) and stat.get("mean") is not None:
                return safe_float(stat.get("mean"))
            elif ind_v.get("mean") is not None:
                return safe_float(ind_v.get("mean"))
        elif ind_v is not None and not isinstance(ind_v, (list, tuple)):
            return safe_float(ind_v)

    feat_stats = obs.get("feature_statistics")
    if isinstance(feat_stats, dict):
        fst = feat_stats.get(key_upper)
        if isinstance(fst, dict) and fst.get("mean") is not None:
            return safe_float(fst.get("mean"))

    return np.nan


def get_band_val(obs: Dict[str, Any], band_name: str) -> float:
    """Helper to safely retrieve optical band mean values."""
    if not isinstance(obs, dict):
        return np.nan
    feat_stats = obs.get("feature_statistics")
    if isinstance(feat_stats, dict):
        st = feat_stats.get(band_name)
        if isinstance(st, dict) and st.get("mean") is not None:
            return safe_float(st.get("mean"))
    bands = obs.get("bands")
    if isinstance(bands, dict) and bands.get(band_name) is not None:
        return safe_float(bands.get(band_name))
    return np.nan


def get_sar_val(obs: Dict[str, Any], ch_upper: str, ch_lower: str) -> float:
    """Helper to safely retrieve SAR channel mean values."""
    if not isinstance(obs, dict):
        return np.nan
    sar = obs.get("sar")
    if isinstance(sar, dict):
        pol = sar.get(ch_lower)
        if isinstance(pol, dict) and pol.get("mean") is not None:
            return safe_float(pol.get("mean"))
        elif pol is not None and not isinstance(pol, (list, tuple, dict)):
            return safe_float(pol)

    feat_stats = obs.get("feature_statistics")
    if isinstance(feat_stats, dict):
        st = feat_stats.get(ch_upper)
        if isinstance(st, dict) and st.get("mean") is not None:
            return safe_float(st.get("mean"))
    return np.nan


def extract_observation_features(
    current_obs: Dict[str, Any],
    historical_observations: Optional[List[Dict[str, Any]]] = None,
    current_date: Optional[str] = None,
) -> Dict[str, float]:
    """
    Extracts the full 29-feature vector from a multimodal observation and its prior timeline.
    
    STRICT DATA LEAKAGE PREVENTION:
    Only historical observations acquired strictly BEFORE `current_date` are examined.
    Future observations are excluded.
    """
    features: Dict[str, float] = {}

    # 1. Determine prediction timestamp
    pred_date_str = current_date or current_obs.get("date") or current_obs.get("acquisition_date")
    if pred_date_str:
        pred_date = datetime.date.fromisoformat(pred_date_str[:10])
    else:
        pred_date = datetime.date.today()

    # 2. Extract Current Optical Features
    features["opt_ndvi"] = get_stat_val(current_obs, "NDVI", "ndvi")
    features["opt_ndmi"] = get_stat_val(current_obs, "NDMI", "ndmi")
    features["opt_ndre"] = get_stat_val(current_obs, "NDRE", "ndre")
    features["opt_evi"] = get_stat_val(current_obs, "EVI", "evi")

    # Optical bands
    features["opt_b02"] = get_band_val(current_obs, "B02")
    features["opt_b03"] = get_band_val(current_obs, "B03")
    features["opt_b04"] = get_band_val(current_obs, "B04")
    features["opt_b05"] = get_band_val(current_obs, "B05")
    features["opt_b08"] = get_band_val(current_obs, "B08")
    features["opt_b11"] = get_band_val(current_obs, "B11")

    # 3. Extract Current SAR Features
    features["sar_vv"] = get_sar_val(current_obs, "VV", "vv")
    features["sar_vh"] = get_sar_val(current_obs, "VH", "vh")
    features["sar_vv_vh_ratio"] = get_sar_val(current_obs, "VV_VH_RATIO", "vv_vh_ratio")
    if math.isnan(features["sar_vv_vh_ratio"]) and not math.isnan(features["sar_vv"]) and not math.isnan(features["sar_vh"]):
        if features["sar_vh"] != 0:
            features["sar_vv_vh_ratio"] = features["sar_vv"] / features["sar_vh"]

    # Missing SAR flag
    missing_sar = math.isnan(features["sar_vv"]) or math.isnan(features["sar_vh"])
    features["missing_sar_flag"] = 1.0 if missing_sar else 0.0

    # 4. Extract Quality Features
    quality_summary = current_obs.get("quality_summary") or current_obs.get("quality") or {}
    cloud_cov = safe_float(
        quality_summary.get("cloud_coverage")
        or current_obs.get("cloud_coverage")
        or current_obs.get("scene", {}).get("cloud_coverage"),
        default=0.0,
    )
    valid_pct = safe_float(
        quality_summary.get("fused_valid_percentage")
        or quality_summary.get("valid_percentage")
        or quality_summary.get("valid_pixel_percentage"),
        default=100.0,
    )
    invalid_pct = safe_float(
        quality_summary.get("invalid_pixel_percentage"),
        default=max(0.0, 100.0 - valid_pct),
    )

    features["cloud_coverage"] = cloud_cov
    features["valid_pixel_pct"] = valid_pct
    features["nodata_pct"] = invalid_pct

    temporal_matching = current_obs.get("temporal_matching") or {}
    gap_days = temporal_matching.get("date_difference_days")
    features["s1_s2_temporal_gap_days"] = safe_float(gap_days, default=np.nan)

    # 5. Extract Temporal Lagged & Rate Features
    # Filter prior observations strictly before pred_date to PREVENT DATA LEAKAGE
    valid_priors: List[Tuple[datetime.date, Dict[str, Any]]] = []
    if historical_observations:
        for obs in historical_observations:
            obs_dt_str = obs.get("date") or obs.get("acquisition_date")
            if not obs_dt_str:
                continue
            try:
                obs_dt = datetime.date.fromisoformat(obs_dt_str[:10])
            except (ValueError, TypeError):
                continue
            if obs_dt < pred_date:
                valid_priors.append((obs_dt, obs))

    # Sort strictly chronological
    valid_priors.sort(key=lambda x: x[0])

    features["usable_history_count"] = float(len(valid_priors))
    features["insufficient_history_flag"] = 1.0 if len(valid_priors) == 0 else 0.0

    if len(valid_priors) > 0:
        prev_dt, prev_obs = valid_priors[-1]
        delta_days = max(1, abs((pred_date - prev_dt).days))
        features["temporal_interval_days"] = float(delta_days)

        # Prior values
        p_ndvi = get_stat_val(prev_obs, "NDVI", "ndvi")
        p_ndmi = get_stat_val(prev_obs, "NDMI", "ndmi")
        p_ndre = get_stat_val(prev_obs, "NDRE", "ndre")
        p_evi = get_stat_val(prev_obs, "EVI", "evi")
        p_vv = get_sar_val(prev_obs, "VV", "vv")
        p_vh = get_sar_val(prev_obs, "VH", "vh")

        features["prev_ndvi"] = p_ndvi
        features["prev_ndmi"] = p_ndmi
        features["prev_ndre"] = p_ndre
        features["prev_evi"] = p_evi
        features["prev_vv"] = p_vv
        features["prev_vh"] = p_vh

        # Differences
        features["delta_ndvi"] = (features["opt_ndvi"] - p_ndvi) if not (math.isnan(features["opt_ndvi"]) or math.isnan(p_ndvi)) else np.nan
        features["delta_ndmi"] = (features["opt_ndmi"] - p_ndmi) if not (math.isnan(features["opt_ndmi"]) or math.isnan(p_ndmi)) else np.nan
        features["delta_ndre"] = (features["opt_ndre"] - p_ndre) if not (math.isnan(features["opt_ndre"]) or math.isnan(p_ndre)) else np.nan
        features["delta_evi"] = (features["opt_evi"] - p_evi) if not (math.isnan(features["opt_evi"]) or math.isnan(p_evi)) else np.nan
        features["delta_vv"] = (features["sar_vv"] - p_vv) if not (math.isnan(features["sar_vv"]) or math.isnan(p_vv)) else np.nan
        features["delta_vh"] = (features["sar_vh"] - p_vh) if not (math.isnan(features["sar_vh"]) or math.isnan(p_vh)) else np.nan

        # Rates of change per day
        features["rate_ndvi_per_day"] = (features["delta_ndvi"] / delta_days) if not math.isnan(features["delta_ndvi"]) else np.nan
        features["rate_ndmi_per_day"] = (features["delta_ndmi"] / delta_days) if not math.isnan(features["delta_ndmi"]) else np.nan
        features["rate_ndre_per_day"] = (features["delta_ndre"] / delta_days) if not math.isnan(features["delta_ndre"]) else np.nan
        features["rate_evi_per_day"] = (features["delta_evi"] / delta_days) if not math.isnan(features["delta_evi"]) else np.nan

    else:
        # No prior observations available
        features["temporal_interval_days"] = np.nan
        features["prev_ndvi"] = np.nan
        features["prev_ndmi"] = np.nan
        features["prev_ndre"] = np.nan
        features["prev_evi"] = np.nan
        features["prev_vv"] = np.nan
        features["prev_vh"] = np.nan
        features["delta_ndvi"] = np.nan
        features["delta_ndmi"] = np.nan
        features["delta_ndre"] = np.nan
        features["delta_evi"] = np.nan
        features["delta_vv"] = np.nan
        features["delta_vh"] = np.nan
        features["rate_ndvi_per_day"] = np.nan
        features["rate_ndmi_per_day"] = np.nan
        features["rate_ndre_per_day"] = np.nan
        features["rate_evi_per_day"] = np.nan

    # Ensure all 29 features exist in dictionary
    for name in ALL_MODEL_FEATURE_NAMES:
        if name not in features:
            features[name] = np.nan

    return features

def spatial_temporal_split(
    records: List[Dict[str, Any]],
    field_id_key: str = "field_id",
    date_key: str = "date",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Performs spatial-group splitting by unique field identifier to strictly prevent spatial leakage.
    If multiple fields are present, fields are partitioned so no field exists across multiple splits.
    """
    if not records:
        return [], [], []

    rng = np.random.RandomState(random_state)
    field_ids = list(set(r.get(field_id_key, f"field_{i}") for i, r in enumerate(records)))
    rng.shuffle(field_ids)

    n_fields = len(field_ids)
    if n_fields >= 3:
        # Guarantee at least 1 field in validation and test, remaining in train
        n_val = max(1, int(round(n_fields * val_ratio)))
        n_test = max(1, int(round(n_fields * (1.0 - train_ratio - val_ratio))))
        n_train = n_fields - n_val - n_test
        if n_train < 1:
            n_train = 1
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1

        train_fields = set(field_ids[:n_train])
        val_fields = set(field_ids[n_train:n_train + n_val])
        test_fields = set(field_ids[n_train + n_val:])

        train = [r for r in records if r.get(field_id_key) in train_fields]
        val = [r for r in records if r.get(field_id_key) in val_fields]
        test = [r for r in records if r.get(field_id_key) in test_fields]
    else:
        # Temporal split if only single field or 2 fields are available
        sorted_records = sorted(records, key=lambda x: x.get(date_key, ""))
        n = len(sorted_records)
        if n >= 3:
            n_val = max(1, int(round(n * val_ratio)))
            n_test = max(1, int(round(n * (1.0 - train_ratio - val_ratio))))
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train = 1
                if n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
            train = sorted_records[:n_train]
            val = sorted_records[n_train:n_train + n_val]
            test = sorted_records[n_train + n_val:]
        else:
            train = sorted_records[:1]
            val = sorted_records[1:2]
            test = sorted_records[2:]

    return train, val, test

def generate_synthetic_demo_dataset(
    n_samples: int = 300,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generates a deterministic synthetic demonstration dataset for test suite validation
    and interface demonstration.
    
    EXPLICIT DEMO LABELING:
    This dataset reflects synthetic agricultural curves for 3 classes:
      0: Nominal
      1: Potential Early Stress
      2: Elevated Stress
    """
    rng = np.random.RandomState(random_state)
    samples: List[Dict[str, Any]] = []

    # Distribution: 60% Nominal, 25% Potential Stress, 15% Elevated Stress
    class_assignments = rng.choice([0, 1, 2], size=n_samples, p=[0.60, 0.25, 0.15])

    base_dates = [
        "2026-08-01", "2026-08-12", "2026-08-25",
        "2026-09-06", "2026-09-18", "2026-10-02", "2026-10-15"
    ]

    for i in range(n_samples):
        cls = int(class_assignments[i])
        field_id = f"demo_field_{i % 20:02d}"
        date_idx = rng.randint(1, len(base_dates))
        curr_date = base_dates[date_idx]
        prev_date = base_dates[date_idx - 1]
        delta_days = (datetime.date.fromisoformat(curr_date) - datetime.date.fromisoformat(prev_date)).days

        # Physics-based baseline indices
        if cls == 0:  # Healthy / Nominal
            opt_ndvi = float(np.clip(rng.normal(0.72, 0.06), 0.55, 0.88))
            opt_ndmi = float(np.clip(rng.normal(0.28, 0.05), 0.12, 0.42))
            opt_ndre = float(np.clip(rng.normal(0.44, 0.05), 0.30, 0.58))
            opt_evi = float(np.clip(rng.normal(0.55, 0.06), 0.38, 0.72))
            sar_vv = float(rng.normal(-10.5, 1.2))
            sar_vh = float(rng.normal(-16.8, 1.4))

            # Healthy change: slight growth or stable
            d_ndvi = float(rng.normal(0.03, 0.02))
            d_ndmi = float(rng.normal(0.01, 0.02))
            d_ndre = float(rng.normal(0.02, 0.02))
            d_evi = float(rng.normal(0.02, 0.02))
            d_vv = float(rng.normal(0.2, 0.5))
            d_vh = float(rng.normal(0.3, 0.5))

        elif cls == 1:  # Potential Early Stress (e.g., initial water strain: NDMI drops first)
            opt_ndvi = float(np.clip(rng.normal(0.60, 0.06), 0.45, 0.72))
            opt_ndmi = float(np.clip(rng.normal(0.08, 0.05), -0.05, 0.20))
            opt_ndre = float(np.clip(rng.normal(0.32, 0.05), 0.18, 0.42))
            opt_evi = float(np.clip(rng.normal(0.42, 0.05), 0.28, 0.55))
            sar_vv = float(rng.normal(-11.2, 1.3))
            sar_vh = float(rng.normal(-18.5, 1.5))

            # Early stress change: moisture and red-edge decline faster than NDVI
            d_ndvi = float(rng.normal(-0.03, 0.02))
            d_ndmi = float(rng.normal(-0.08, 0.03))
            d_ndre = float(rng.normal(-0.05, 0.02))
            d_evi = float(rng.normal(-0.03, 0.02))
            d_vv = float(rng.normal(-0.5, 0.6))
            d_vh = float(rng.normal(-0.8, 0.6))

        else:  # Elevated Stress (acute drought / severe senescence)
            opt_ndvi = float(np.clip(rng.normal(0.38, 0.07), 0.20, 0.50))
            opt_ndmi = float(np.clip(rng.normal(-0.12, 0.06), -0.30, 0.02))
            opt_ndre = float(np.clip(rng.normal(0.20, 0.04), 0.10, 0.28))
            opt_evi = float(np.clip(rng.normal(0.24, 0.05), 0.12, 0.35))
            sar_vv = float(rng.normal(-13.0, 1.5))
            sar_vh = float(rng.normal(-21.2, 1.6))

            # Severe collapse across all indicators
            d_ndvi = float(rng.normal(-0.14, 0.04))
            d_ndmi = float(rng.normal(-0.16, 0.04))
            d_ndre = float(rng.normal(-0.12, 0.03))
            d_evi = float(rng.normal(-0.12, 0.03))
            d_vv = float(rng.normal(-1.8, 0.8))
            d_vh = float(rng.normal(-2.4, 0.8))

        # Optical Reflectance Bands
        b02 = float(np.clip(rng.normal(0.045, 0.008), 0.02, 0.10))
        b03 = float(np.clip(rng.normal(0.065, 0.010), 0.03, 0.12))
        b04 = float(np.clip(rng.normal(0.040 + (1 - opt_ndvi) * 0.08, 0.01), 0.02, 0.18))
        b05 = float(np.clip(rng.normal(0.090, 0.015), 0.04, 0.18))
        b08 = float(np.clip(rng.normal(0.350 * opt_ndvi + 0.1, 0.03), 0.10, 0.55))
        b11 = float(np.clip(rng.normal(0.200 - opt_ndmi * 0.15, 0.02), 0.08, 0.35))

        sar_ratio = float(pow(10.0, (sar_vv - sar_vh) / 10.0))

        prev_ndvi = opt_ndvi - d_ndvi
        prev_ndmi = opt_ndmi - d_ndmi
        prev_ndre = opt_ndre - d_ndre
        prev_evi = opt_evi - d_evi
        prev_vv = sar_vv - d_vv
        prev_vh = sar_vh - d_vh

        sample = {
            "sample_id": f"DEMO_SAMP_{i:04d}",
            "field_id": field_id,
            "date": curr_date,
            "label": cls,
            "label_name": REVERSE_LABEL_MAPPING[cls],
            "is_demo": True,
            # Features
            "opt_ndvi": opt_ndvi,
            "opt_ndmi": opt_ndmi,
            "opt_ndre": opt_ndre,
            "opt_evi": opt_evi,
            "opt_b02": b02,
            "opt_b03": b03,
            "opt_b04": b04,
            "opt_b05": b05,
            "opt_b08": b08,
            "opt_b11": b11,
            "sar_vv": sar_vv,
            "sar_vh": sar_vh,
            "sar_vv_vh_ratio": sar_ratio,
            "prev_ndvi": prev_ndvi,
            "prev_ndmi": prev_ndmi,
            "prev_ndre": prev_ndre,
            "prev_evi": prev_evi,
            "prev_vv": prev_vv,
            "prev_vh": prev_vh,
            "delta_ndvi": d_ndvi,
            "delta_ndmi": d_ndmi,
            "delta_ndre": d_ndre,
            "delta_evi": d_evi,
            "delta_vv": d_vv,
            "delta_vh": d_vh,
            "rate_ndvi_per_day": d_ndvi / delta_days,
            "rate_ndmi_per_day": d_ndmi / delta_days,
            "rate_ndre_per_day": d_ndre / delta_days,
            "rate_evi_per_day": d_evi / delta_days,
            "temporal_interval_days": float(delta_days),
            "usable_history_count": float(date_idx),
            "cloud_coverage": float(rng.uniform(1.0, 18.0)),
            "valid_pixel_pct": float(rng.uniform(85.0, 100.0)),
            "nodata_pct": float(rng.uniform(0.0, 15.0)),
            "s1_s2_temporal_gap_days": float(rng.choice([0, 1, 2, 3])),
            "missing_sar_flag": 0.0,
            "insufficient_history_flag": 0.0,
        }
        samples.append(sample)

    return samples
