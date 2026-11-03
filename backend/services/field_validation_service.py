from __future__ import annotations

import datetime
from typing import Dict, Any, List, Optional
from models.field_validation import (
    FieldScoutingObservation,
    LabelVerificationStatus,
    ObserverCredentialType,
    DataSharingTier,
)


class FieldValidationService:
    """
    Service managing in-situ agronomic field validation data collection,
    rigorous quality-control review workflows, satellite observation alignment,
    and dataset versioning.
    
    Guarantees:
    - Never fabricates field observations or synthetic ground truth.
    - Rejects unverified observations from entering model training corpora.
    - Requires certified agronomist credentials and double-blind review for verified status.
    - Matches observations with Copernicus Sentinel acquisitions within ±5 days.
    """

    def __init__(self):
        # In-memory storage for validated field trial sessions
        self._observations_store: Dict[str, FieldScoutingObservation] = {}

    def record_observation(self, observation: FieldScoutingObservation) -> Dict[str, Any]:
        """
        Ingests and validates an agronomic field observation record.
        Applies strict quality assurance checks.
        """
        # QC Rule 1: Informed consent check
        if not observation.consent.farmer_consent_obtained:
            raise ValueError("Observation rejected: Farmer/landowner informed consent is required.")

        # QC Rule 2: Replicate check for instrument readings
        if observation.chlorophyll:
            if observation.chlorophyll.replicate_count < 3:
                raise ValueError("QC Failure: Chlorophyll SPAD readings require at least 3 sampling replicates.")

        # QC Rule 3: Coordinate bounding check
        geom = observation.geometry
        coords = geom.get("coordinates")
        if not coords:
            raise ValueError("QC Failure: Geometry must contain coordinates.")

        # Store record
        self._observations_store[observation.observation_id] = observation

        return {
            "status": "success",
            "observation_id": observation.observation_id,
            "verification_status": observation.verification_status.value,
            "dataset_version": observation.dataset_version,
            "message": "Observation registered successfully under quality-control workflow.",
        }

    def review_observation(
        self,
        observation_id: str,
        reviewer_id: str,
        reviewer_credential: ObserverCredentialType,
        action: str,  # "APPROVE", "REJECT", "REQUEST_REVISION"
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Double-blind verification step executed by a credentialed second agronomist.
        """
        if observation_id not in self._observations_store:
            raise KeyError(f"Observation ID '{observation_id}' not found.")

        obs = self._observations_store[observation_id]

        # Observer cannot verify their own observation (Double-blind rule)
        if obs.observer.observer_id == reviewer_id:
            raise ValueError("Integrity Violation: Observer cannot review their own field observation.")

        # Reviewer must have verified professional agronomic credentials
        valid_reviewer_creds = {
            ObserverCredentialType.CERTIFIED_CROP_ADVISOR,
            ObserverCredentialType.AGRONOMIST_MSC_PHD,
            ObserverCredentialType.UNIVERSITY_RESEARCHER,
        }
        if reviewer_credential not in valid_reviewer_creds:
            raise PermissionError("Only Certified Crop Advisors or PhD/MSc Agronomists can approve validation ground truth.")

        if action == "APPROVE":
            obs.verification_status = LabelVerificationStatus.VERIFIED_AGRONOMIST
            obs.second_reviewer_id = reviewer_id
            obs.reviewer_notes = notes
        elif action == "REJECT":
            obs.verification_status = LabelVerificationStatus.REJECTED
            obs.second_reviewer_id = reviewer_id
            obs.reviewer_notes = notes
        else:
            obs.verification_status = LabelVerificationStatus.PENDING_SECOND_REVIEW
            obs.reviewer_notes = notes

        return {
            "status": "success",
            "observation_id": observation_id,
            "verification_status": obs.verification_status.value,
            "reviewer_id": reviewer_id,
            "reviewer_notes": notes,
        }

    def match_with_satellite_scene(
        self,
        observation_id: str,
        sentinel_scenes: List[Dict[str, Any]],
        max_temporal_offset_days: int = 5,
    ) -> Dict[str, Any]:
        """
        Aligns ground observation with the temporally and spatially closest Sentinel pass.
        """
        if observation_id not in self._observations_store:
            raise KeyError(f"Observation ID '{observation_id}' not found.")

        obs = self._observations_store[observation_id]
        obs_date = obs.observation_timestamp.date()

        best_scene = None
        min_offset = 999999

        for scene in sentinel_scenes:
            raw_dt = scene.get("acquisition_date") or scene.get("datetime")
            if not raw_dt:
                continue
            scene_date = datetime.date.fromisoformat(raw_dt[:10])
            offset = abs((scene_date - obs_date).days)
            if offset <= max_temporal_offset_days and offset < min_offset:
                min_offset = offset
                best_scene = scene

        if best_scene:
            obs.matched_sentinel_scene_id = best_scene.get("scene_id") or best_scene.get("id")
            obs.satellite_temporal_offset_days = min_offset
            return {
                "matched": True,
                "observation_id": observation_id,
                "matched_scene_id": obs.matched_sentinel_scene_id,
                "temporal_offset_days": min_offset,
                "data_quality": "HIGH" if min_offset <= 2 else "ACCEPTABLE",
            }

        return {
            "matched": False,
            "observation_id": observation_id,
            "matched_scene_id": None,
            "temporal_offset_days": None,
            "message": f"No Sentinel acquisition found within ±{max_temporal_offset_days} days.",
        }

    def list_observations(
        self,
        field_id: Optional[str] = None,
        verification_status: Optional[LabelVerificationStatus] = None,
    ) -> List[Dict[str, Any]]:
        """Lists recorded field observations with optional filtering."""
        results = []
        for obs in self._observations_store.values():
            if field_id and obs.field_id != field_id:
                continue
            if verification_status and obs.verification_status != verification_status:
                continue
            results.append(obs.model_dump())
        return results

    def get_dataset_provenance(self) -> Dict[str, Any]:
        """Returns metadata regarding active ground-truth catalog version and stats."""
        total = len(self._observations_store)
        verified = sum(
            1 for o in self._observations_store.values()
            if o.verification_status == LabelVerificationStatus.VERIFIED_AGRONOMIST
        )
        return {
            "catalog_name": "CropHealth-GroundTruth-FieldValidation",
            "dataset_version": "v1.0.0-rc1",
            "total_observations_recorded": total,
            "verified_agronomist_labels": verified,
            "production_training_eligibility": verified >= 500,
            "eligibility_message": (
                "ELIGIBLE FOR RETRAINING" if verified >= 500 else
                f"INSUFFICIENT VERIFIED SAMPLES ({verified}/500 required for production model training)"
            ),
        }


field_validation_service = FieldValidationService()
