"""
Crop Health AI — Research Data Acquisition Metadata Validator (Phase 21.2)

Ensures that every ingested satellite observation meets strict scientific requirements:
1. Persistent Field ID and valid AOI geometry.
2. Valid ISO 8601 acquisition timestamp.
3. Satellite platform and genuine product identifier.
4. Correct processing level (L2A for MSI, GRD for C-SAR).
5. Non-interpolation verification: missing data must be explicitly None or NaN, never fabricated.
6. Provenance documentation.
"""

from typing import Dict, Any, List, Optional
import datetime
import math


class MetadataValidationError(ValueError):
    """Raised when an observation record fails scientific metadata validation."""
    pass


class SatelliteMetadataValidator:
    """Validates metadata integrity for research satellite observations."""

    VALID_PLATFORMS = {
        "Sentinel-2A",
        "Sentinel-2B",
        "Sentinel-1A",
        "Sentinel-1B",
        "Sentinel-2",
        "Sentinel-1",
    }

    VALID_PROCESSING_LEVELS = {"L2A", "L1C", "GRD", "SLC"}

    @classmethod
    def validate_observation_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates a single observation record dictionary.
        Raises MetadataValidationError if any invariant is violated.
        Returns the validated record.
        """
        # 1. Field Identifier
        field_id = record.get("field_id")
        if not field_id or not isinstance(field_id, str):
            raise MetadataValidationError("Record must contain a non-empty string 'field_id'.")

        # 2. AOI Geometry
        aoi = record.get("aoi_geometry")
        if not aoi or not isinstance(aoi, dict):
            raise MetadataValidationError("Record must contain a valid 'aoi_geometry' dictionary.")
        lat = aoi.get("latitude")
        lon = aoi.get("longitude")
        if lat is None or lon is None or not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            raise MetadataValidationError(f"Invalid AOI coordinates: lat={lat}, lon={lon}.")

        # 3. Acquisition Date
        acq_date_str = record.get("acquisition_date")
        if not acq_date_str:
            raise MetadataValidationError("Record must contain 'acquisition_date'.")
        try:
            acq_date = datetime.date.fromisoformat(acq_date_str[:10])
        except Exception as exc:
            raise MetadataValidationError(f"Invalid acquisition_date format '{acq_date_str}': {exc}")

        # Ensure not in the future
        if acq_date > datetime.date.today():
            raise MetadataValidationError(f"Acquisition date {acq_date} cannot be in the future.")

        # 4. Satellite Platform & Processing Level
        platform = record.get("satellite_platform")
        if platform not in cls.VALID_PLATFORMS:
            raise MetadataValidationError(
                f"Invalid satellite_platform '{platform}'. Allowed: {cls.VALID_PLATFORMS}"
            )

        proc_level = record.get("processing_level")
        if proc_level not in cls.VALID_PROCESSING_LEVELS:
            raise MetadataValidationError(
                f"Invalid processing_level '{proc_level}'. Allowed: {cls.VALID_PROCESSING_LEVELS}"
            )

        # 5. Product Identifier & Provenance
        product_id = record.get("product_identifier")
        if not product_id or not isinstance(product_id, str):
            raise MetadataValidationError("Record must contain non-empty 'product_identifier'.")

        provenance = record.get("data_provenance")
        if not provenance or not isinstance(provenance, str):
            raise MetadataValidationError("Record must specify 'data_provenance'.")

        # 6. Quality Metrics & Pixel Ratios
        valid_pixel_pct = record.get("valid_pixel_percentage")
        if valid_pixel_pct is not None:
            if not (0.0 <= valid_pixel_pct <= 100.0):
                raise MetadataValidationError(
                    f"valid_pixel_percentage must be between 0 and 100, got {valid_pixel_pct}."
                )

        cloud_cov = record.get("cloud_coverage")
        if cloud_cov is not None:
            if not (0.0 <= cloud_cov <= 100.0):
                raise MetadataValidationError(
                    f"cloud_coverage must be between 0 and 100, got {cloud_cov}."
                )

        # 7. Non-Interpolation Guarantee
        # If an observation is flagged as missing optical or SAR, the corresponding values MUST be None or NaN
        if record.get("is_missing_optical", False):
            optical_features = ["ndvi", "ndmi", "ndre", "evi"]
            for feat in optical_features:
                val = record.get(feat)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    raise MetadataValidationError(
                        f"Non-interpolation rule violated: 'is_missing_optical' is True, but '{feat}' has value {val}."
                    )

        if record.get("is_missing_sar", False):
            sar_features = ["sar_vv", "sar_vh", "sar_vv_vh_ratio"]
            for feat in sar_features:
                val = record.get(feat)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    raise MetadataValidationError(
                        f"Non-interpolation rule violated: 'is_missing_sar' is True, but '{feat}' has value {val}."
                    )

        return record

    @classmethod
    def validate_batch(cls, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates a sequence of observation records.
        Returns summary metrics and any validation errors encountered.
        """
        valid_records = []
        errors = []
        for idx, rec in enumerate(records):
            try:
                validated = cls.validate_observation_record(rec)
                valid_records.append(validated)
            except MetadataValidationError as err:
                errors.append({"index": idx, "field_id": rec.get("field_id"), "error": str(err)})

        return {
            "total_records": len(records),
            "valid_records_count": len(valid_records),
            "error_count": len(errors),
            "is_valid": len(errors) == 0,
            "errors": errors,
        }

