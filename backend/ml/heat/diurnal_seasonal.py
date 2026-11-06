"""
ml/heat/diurnal_seasonal.py:
Implements improved thermal features:
1. Day vs Night LST comparison (Diurnal Temperature Range and Thermal Inertia)
2. Seasonal comparison (Summer Pre-Monsoon vs Winter Post-Monsoon)
3. Multi-criteria Urban Heat Island (UHI) Hotspot Score
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


def compute_diurnal_thermal_analysis(
    lst_day: np.ndarray,
    lst_night: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes diurnal thermal metrics:
      - Diurnal Temperature Range (DTR): LST_day - LST_night
      - Nocturnal Heat Retention Index: Night LST normalized by regional background
    """
    dtr = lst_day - lst_night
    mean_night = float(np.nanmean(lst_night))
    night_retention = lst_night - mean_night

    return {
        "diurnal_temperature_range_map": dtr.astype(np.float32),
        "nocturnal_heat_retention_map": night_retention.astype(np.float32),
        "mean_dtr_celsius": round(float(np.nanmean(dtr)), 2),
        "max_dtr_celsius": round(float(np.nanmax(dtr)), 2),
        "min_dtr_celsius": round(float(np.nanmin(dtr)), 2),
        "mean_day_lst": round(float(np.nanmean(lst_day)), 2),
        "mean_night_lst": round(mean_night, 2),
    }


def compute_seasonal_thermal_analysis(
    lst_summer: np.ndarray,
    lst_winter: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes seasonal thermal contrast (Summer April/May vs Winter Dec/Jan):
      - Seasonal LST Delta: LST_summer - LST_winter
    """
    seasonal_delta = lst_summer - lst_winter

    return {
        "seasonal_delta_map": seasonal_delta.astype(np.float32),
        "mean_summer_lst": round(float(np.nanmean(lst_summer)), 2),
        "mean_winter_lst": round(float(np.nanmean(lst_winter)), 2),
        "mean_seasonal_amplitude": round(float(np.nanmean(seasonal_delta)), 2),
        "max_seasonal_amplitude": round(float(np.nanmax(seasonal_delta)), 2),
    }


def compute_uhi_hotspot_score(
    lst_10m: np.ndarray,
    ndvi_10m: np.ndarray,
    ndbi_10m: np.ndarray,
    mndwi_10m: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculates multi-criteria Urban Heat Island (UHI) Hotspot Score:
      - Thermal Z-Score: relative to natural vegetated cooling refuges (NDVI > 0.50)
      - Impervious Amplification: high NDBI built-up density
      - Canopy Deficit Weighting: max(0, 0.45 - NDVI)
    
    Categorizes pixels into:
      - Severe Hotspot (Score >= 2.5)
      - Moderate Hotspot (1.5 <= Score < 2.5)
      - Mild Thermal Elevation (0.5 <= Score < 1.5)
      - Cool Thermal Refuge (Score < 0.5)
    """
    H, W = lst_10m.shape
    land_mask = mndwi_10m < 0.35

    # Determine natural vegetation baseline (Simhachalam & Kailasagiri forest canopy)
    natural_refuge_mask = (ndvi_10m > 0.50) & land_mask
    if np.sum(natural_refuge_mask) > 100:
        refuge_mean = float(np.mean(lst_10m[natural_refuge_mask]))
        refuge_std = float(np.std(lst_10m[natural_refuge_mask]))
        refuge_std = max(refuge_std, 0.5)
    else:
        refuge_mean = float(np.percentile(lst_10m[land_mask], 15))
        refuge_std = float(np.std(lst_10m[land_mask])) * 0.5

    # 1. Thermal Anomaly Z-Score
    thermal_z = (lst_10m - refuge_mean) / refuge_std

    # 2. Canopy Deficit & Built-up Amplification
    canopy_deficit = np.clip(0.45 - ndvi_10m, 0.0, 0.60) / 0.60
    built_factor = np.clip(ndbi_10m + 0.20, 0.0, 0.80) / 0.80

    # Composite Hotspot Score (0.0 to 4.5+)
    hotspot_score = thermal_z * 0.70 + (canopy_deficit + built_factor) * 0.30
    # Zero out open water bodies
    hotspot_score[~land_mask] = 0.0

    # Zone Classification
    severe_mask = (hotspot_score >= 2.5) & land_mask
    moderate_mask = (hotspot_score >= 1.5) & (hotspot_score < 2.5) & land_mask
    mild_mask = (hotspot_score >= 0.5) & (hotspot_score < 1.5) & land_mask
    refuge_mask = (hotspot_score < 0.5) & land_mask

    total_land_pixels = int(np.sum(land_mask))
    pixel_area_ha = (10.0 * 10.0) / 10000.0 # 0.01 ha per pixel

    zones_breakdown = {
        "severe_hotspot": {
            "label": "Severe Urban Heat Island Hotspot (Score >= 2.5)",
            "pixel_count": int(np.sum(severe_mask)),
            "area_hectares": round(int(np.sum(severe_mask)) * pixel_area_ha, 1),
            "percentage": round((np.sum(severe_mask) / total_land_pixels) * 100, 2),
            "color": "#e11d48", # Crimson
            "action_priority": "Immediate High-Priority Interventions Needed",
        },
        "moderate_hotspot": {
            "label": "Moderate Heat Island (1.5 <= Score < 2.5)",
            "pixel_count": int(np.sum(moderate_mask)),
            "area_hectares": round(int(np.sum(moderate_mask)) * pixel_area_ha, 1),
            "percentage": round((np.sum(moderate_mask) / total_land_pixels) * 100, 2),
            "color": "#f97316", # Orange
            "action_priority": "Medium-Term Canopy Expansion",
        },
        "mild_elevation": {
            "label": "Mild Thermal Elevation (0.5 <= Score < 1.5)",
            "pixel_count": int(np.sum(mild_mask)),
            "area_hectares": round(int(np.sum(mild_mask)) * pixel_area_ha, 1),
            "percentage": round((np.sum(mild_mask) / total_land_pixels) * 100, 2),
            "color": "#eab308", # Amber
            "action_priority": "Canopy Maintenance & Preservation",
        },
        "cool_refuge": {
            "label": "Cool Thermal Refuge (Score < 0.5)",
            "pixel_count": int(np.sum(refuge_mask)),
            "area_hectares": round(int(np.sum(refuge_mask)) * pixel_area_ha, 1),
            "percentage": round((np.sum(refuge_mask) / total_land_pixels) * 100, 2),
            "color": "#10b981", # Emerald
            "action_priority": "Protected Urban Green Space",
        },
    }

    return {
        "hotspot_score_map": hotspot_score.astype(np.float32),
        "refuge_baseline_lst_celsius": round(refuge_mean, 2),
        "mean_hotspot_score": round(float(np.mean(hotspot_score[land_mask])), 2),
        "max_hotspot_score": round(float(np.max(hotspot_score[land_mask])), 2),
        "zones_breakdown": zones_breakdown,
        "total_land_hectares": round(total_land_pixels * pixel_area_ha, 1),
    }
