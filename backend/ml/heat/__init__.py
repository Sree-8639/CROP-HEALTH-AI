"""
ml/heat/__init__.py:
Package exports for Urban Heat Island, Landsat ST_B10 downscaling,
and tree-planting priority delineation.
"""

from ml.heat.landsat_lst import (
    dn_to_lst_celsius,
    lst_celsius_to_dn,
    LandsatLSTTile,
)
from ml.heat.sentinel_predictors import (
    Sentinel2PredictorSuite,
    calculate_ndvi,
    calculate_ndbi,
    calculate_mndwi,
    calculate_broadband_albedo,
    aggregate_10m_to_30m,
)
from ml.heat.downscaler_rf import (
    RandomForestLSTDownscaler,
)
from ml.heat.diurnal_seasonal import (
    compute_diurnal_thermal_analysis,
    compute_seasonal_thermal_analysis,
    compute_uhi_hotspot_score,
)
from ml.heat.tree_planting_priority import (
    compute_tree_planting_priority_index,
    delineate_tree_planting_priority_zones,
)

__all__ = [
    "dn_to_lst_celsius",
    "lst_celsius_to_dn",
    "LandsatLSTTile",
    "Sentinel2PredictorSuite",
    "calculate_ndvi",
    "calculate_ndbi",
    "calculate_mndwi",
    "calculate_broadband_albedo",
    "aggregate_10m_to_30m",
    "RandomForestLSTDownscaler",
    "compute_diurnal_thermal_analysis",
    "compute_seasonal_thermal_analysis",
    "compute_uhi_hotspot_score",
    "compute_tree_planting_priority_index",
    "delineate_tree_planting_priority_zones",
]
