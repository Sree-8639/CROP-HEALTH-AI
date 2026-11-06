"""
Crop Health AI — Ground Truth & Target Definition Workflow (Phase 21.3)

Maintains rigorous scientific separation between:
1. Satellite-Index Reference Targets (for time-series forecasting): Future genuine
   satellite observations from an independent subsequent acquisition ($t+h$).
2. Agronomic Biological Ground Truth (for future classification/yield tasks): Certified
   in-situ field surveys, SPAD readings, and official harvest records.
"""

from typing import Dict, Any, List, Optional, Tuple
import datetime
import math


class CircularValidationError(ValueError):
    """Raised when ground truth is generated from the same features used as input."""
    pass


class TargetHorizonGenerator:
    """
    Constructs future satellite index forecasting targets from chronological observation sequences.
    Strictly prevents circular thresholding and data leakage.
    """

    def __init__(self, target_index: str = "ndvi", min_horizon_days: int = 5, max_horizon_days: int = 180):
        self.target_index = target_index.lower()
        self.min_horizon_days = min_horizon_days
        self.max_horizon_days = max_horizon_days

    def generate_supervised_pairs(
        self,
        observations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Takes a chronological list of observation records for a single field and generates
        valid (features_at_t, target_at_t_plus_h) pairs.
        
        Invariant:
        target must be calculated from an observation whose acquisition_date > current observation_date.
        """
        if len(observations) < 2:
            return []

        # Sort chronologically by date
        sorted_obs = sorted(observations, key=lambda x: x["acquisition_date"])
        supervised_records = []

        for i in range(len(sorted_obs) - 1):
            current_obs = sorted_obs[i]
            current_date = datetime.date.fromisoformat(current_obs["acquisition_date"][:10])

            # Find matching future observation within valid horizon
            matched_future = None
            horizon_days = None

            for j in range(i + 1, len(sorted_obs)):
                future_obs = sorted_obs[j]
                future_date = datetime.date.fromisoformat(future_obs["acquisition_date"][:10])
                delta_days = (future_date - current_date).days

                if self.min_horizon_days <= delta_days <= self.max_horizon_days:
                    # Valid future observation found
                    future_target_val = future_obs.get(self.target_index)
                    if future_target_val is not None and not math.isnan(future_target_val):
                        matched_future = future_obs
                        horizon_days = delta_days
                        break

            if matched_future is not None and horizon_days is not None:
                record = {
                    "base_observation": current_obs,
                    "target_date": matched_future["acquisition_date"][:10],
                    "target_horizon_days": horizon_days,
                    "target_product_identifier": matched_future["product_identifier"],
                    "target_value": float(matched_future[self.target_index]),
                    "target_index_name": self.target_index.upper(),
                    "is_synthetic_target": bool(matched_future.get("is_demo", False)),
                }
                supervised_records.append(record)

        return supervised_records


class AgronomicGroundTruthRegistry:
    """
    Registry for future independent agronomic field measurements (scouting, SPAD, LAI, yields).
    Enforces that unmeasured labels remain explicitly missing.
    """

    ALLOWED_SOURCES = {
        "IN_SITU_RESEARCH_STATION",
        "KRISHI_VIGYAN_KENDRA_SURVEY",
        "RADIANT_MLHUB",
        "CROPHARVEST",
        "AGRICULTURE_DEPT_OFFICIAL_STATISTICS",
    }

    def __init__(self):
        self._records: Dict[str, Dict[str, Any]] = {}

    def register_ground_truth(
        self,
        record_id: str,
        field_id: str,
        observation_date: str,
        source: str,
        crop_stage: Optional[str] = None,
        spad_chlorophyll: Optional[float] = None,
        leaf_area_index: Optional[float] = None,
        measured_biomass_g_m2: Optional[float] = None,
        verified_yield_kg_ha: Optional[float] = None,
        agronomist_stress_rating: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Registers a certified ground truth observation."""
        if source not in self.ALLOWED_SOURCES:
            raise ValueError(f"Invalid ground truth source '{source}'. Must be one of {self.ALLOWED_SOURCES}")

        entry = {
            "record_id": record_id,
            "field_id": field_id,
            "observation_date": observation_date,
            "source": source,
            "crop_stage": crop_stage,
            "spad_chlorophyll": spad_chlorophyll,
            "leaf_area_index": leaf_area_index,
            "measured_biomass_g_m2": measured_biomass_g_m2,
            "verified_yield_kg_ha": verified_yield_kg_ha,
            "agronomist_stress_rating": agronomist_stress_rating,
            "metadata": metadata or {},
            "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._records[record_id] = entry
        return entry

    def get_ground_truth(self, field_id: str, observation_date: str) -> Optional[Dict[str, Any]]:
        """Returns matching ground truth for a field and date, or None if unmeasured."""
        for rec in self._records.values():
            if rec["field_id"] == field_id and rec["observation_date"] == observation_date:
                return rec
        return None

    def count(self) -> int:
        return len(self._records)
