"""
ml/forest/__init__.py:
Package exports for forest change detection, Hansen GFC benchmark,
Siamese U-Net, and spatial evaluation.
"""

from ml.forest.hansen_dataset import (
    HansenReferenceTile,
    PatchForestDataset,
    build_spatial_split_forest_datasets,
)
from ml.forest.baselines import (
    SpectralDifferencingBaseline,
    TorresEarlyConcatUNet,
    train_unet_model,
    evaluate_torch_model,
)
from ml.forest.siamese_unet import (
    MultimodalSiameseUNet,
    train_siamese_model,
    evaluate_siamese_model,
)
from ml.forest.evaluation import (
    raster_mask_to_geojson_polygons,
    generate_visual_overlays_base64,
)

__all__ = [
    "HansenReferenceTile",
    "PatchForestDataset",
    "build_spatial_split_forest_datasets",
    "SpectralDifferencingBaseline",
    "TorresEarlyConcatUNet",
    "train_unet_model",
    "evaluate_torch_model",
    "MultimodalSiameseUNet",
    "train_siamese_model",
    "evaluate_siamese_model",
    "raster_mask_to_geojson_polygons",
    "generate_visual_overlays_base64",
]
