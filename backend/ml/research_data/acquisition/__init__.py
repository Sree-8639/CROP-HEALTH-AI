"""
Research data acquisition package.
"""

from ml.research_data.acquisition.metadata_validator import (
    SatelliteMetadataValidator,
    MetadataValidationError,
)
from ml.research_data.acquisition.collector import ResearchObservationCollector

__all__ = [
    "SatelliteMetadataValidator",
    "MetadataValidationError",
    "ResearchObservationCollector",
]

