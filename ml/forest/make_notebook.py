"""
make_notebook.py:
Generates the comprehensive training notebook ml/forest/forest_deforestation_training.ipynb.
"""

import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Forest Canopy Disturbance & Change Detection Training Pipeline\n",
                "### Benchmarking Sentinel-2 Optical & Sentinel-1 SAR Fusion Against Global Forest Change\n",
                "\n",
                "This notebook provides a complete, reproducible workflow for:\n",
                "1. **Reference Labels Ingestion & Disclosure**: Hansen Global Forest Change (GFC v1.11, 30m resolution) and user ground-truth labels.\n",
                "2. **Patch Dataset Builder**: Extraction of 256x256 multimodal patches across strict non-overlapping **SPATIAL splits**.\n",
                "3. **Baselines**:\n",
                "   - Spectral Differencing (Delta-NBR and Delta-NDVI thresholding)\n",
                "   - Early-Concatenation U-Net (as benchmarked by *Torres et al., 2021* for Sentinel-2 optical)\n",
                "4. **Improved Model**: Multimodal Siamese U-Net fusing Sentinel-1 C-band SAR with Sentinel-2 optical imagery for cloud resilience.\n",
                "5. **Scientific Evaluation**: Precision, Recall, F1, and IoU on the held-out test region and an **Out-of-Region Indian Forest AOI** (Hasdeo Arand, Chhattisgarh).\n",
                "6. **Vector Polygonization**: GeoJSON extraction with exact hectare area measurements.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 1. Ground-Truth Labels & Scientific Disclosures\n",
                "\n",
                "### Hansen Global Forest Change (GFC) Reference Labels (30m)\n",
                "- **Source**: Hansen et al. (2013), updated annually via Landsat 7/8/9 observations.\n",
                "- **Nominal Resolution**: 30 meters Ground Sample Distance (GSD).\n",
                "- **Known Label Noise**:\n",
                "  - **Temporal Attribution Lag**: Canopy loss occurring late in the calendar year is often attributed to the following year.\n",
                "  - **Selective Logging Omission**: Thinning or selective felling under partial canopy is often undetected by 30m Landsat.\n",
                "  - **Rotational Crop/Plantation Errors**: Bamboo, eucalyptus, and oil palm harvests create false-positive deforestation alerts.\n",
                "  - **Grid Resampling Uncertainty**: Resampling 30m labels to 10m Sentinel-2 grid introduces edge boundary blur.\n",
                "\n",
                "### User-Provided Ground-Truth Labels Format\n",
                "To evaluate models against surveyor field records or drone ground-truth, supply a **GeoJSON** or **CSV** file with:\n",
                "- `polygon_id`: Unique identifier (string or integer)\n",
                "- `geometry`: GeoJSON Polygon or WKT polygon in WGS84 (`EPSG:4326`)\n",
                "- `disturbance_date`: Date of canopy loss event (`YYYY-MM-DD`)\n",
                "- `disturbance_type`: `clearing`, `fire`, `selective_logging`, or `plantation_harvest`\n",
                "- `verification_source`: `field_inspection`, `drone_orthomosaic`, or `high_res_planet`\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import numpy as np\n",
                "import torch\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Set repository path\n",
                "BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), \"..\", \"..\"))\n",
                "if BASE_DIR not in sys.path:\n",
                "    sys.path.insert(0, BASE_DIR)\n",
                "\n",
                "from ml.forest.hansen_dataset import build_spatial_split_forest_datasets, HansenReferenceTile\n",
                "from ml.forest.baselines import SpectralDifferencingBaseline, TorresEarlyConcatUNet, train_unet_model, evaluate_torch_model\n",
                "from ml.forest.siamese_unet import MultimodalSiameseUNet, train_siamese_model, evaluate_siamese_model\n",
                "from ml.forest.evaluation import raster_mask_to_geojson_polygons, generate_visual_overlays_base64\n",
                "\n",
                "print(f\"PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 2. Dataset Builder: 256x256 Patches with Spatial Partitioning\n",
                "\n",
                "We enforce a strict **spatial block split** rather than a random patch split to prevent spatial autocorrelation leakage:\n",
                "- **Train Region**: Northern and Western forest sectors\n",
                "- **Validation Region**: Central transitional sector\n",
                "- **Test Region**: South-Eastern forest sector\n",
                "- **Out-of-Region Indian AOI**: Hasdeo Arand, Chhattisgarh (Central Indian dry deciduous teak/sal biome)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Build spatial split dataset\n",
                "dataset_bundle = build_spatial_split_forest_datasets(\n",
                "    tile_size=512,\n",
                "    patch_size=256,\n",
                "    stride=256,\n",
                "    year_before=2020,\n",
                "    year_after=2021,\n",
                "    random_seed=42,\n",
                ")\n",
                "\n",
                "train_ds = dataset_bundle[\"train_dataset\"]\n",
                "val_ds = dataset_bundle[\"val_dataset\"]\n",
                "test_ds = dataset_bundle[\"test_dataset\"]\n",
                "indian_ds = dataset_bundle[\"indian_aoi_dataset\"]\n",
                "meta = dataset_bundle[\"spatial_metadata\"]\n",
                "\n",
                "print(f\"Spatial Train Patches: {len(train_ds)}\")\n",
                "print(f\"Spatial Val Patches:   {len(val_ds)}\")\n",
                "print(f\"Spatial Test Patches:  {len(test_ds)}\")\n",
                "print(f\"Indian AOI Patches:    {len(indian_ds)} ({meta['indian_aoi_details']['name']})\")\n",
                "\n",
                "# Inspect tensor dimensions\n",
                "x0_sample, x1_sample, y_sample = test_ds[0]\n",
                "print(f\"T0 Tensor Shape: {x0_sample.shape} (7 S2 bands + 3 S1 SAR bands)\")\n",
                "print(f\"T1 Tensor Shape: {x1_sample.shape}\")\n",
                "print(f\"Loss Mask Shape: {y_sample.shape} (Hansen GFC 2020->2021 loss)\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 3. Baseline 1: Spectral Differencing (Delta-NBR & Delta-NDVI)\n",
                "\n",
                "Computes normalized burn ratio and vegetation index differencing:\n",
                "$$\\Delta\\text{NBR} = \\text{NBR}_{T_0} - \\text{NBR}_{T_1}$$\n",
                "A reduction in canopy foliage causes positive Delta-NBR. Optimal threshold is selected on validation patches.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Delta-NBR differencing\n",
                "nbr_baseline = SpectralDifferencingBaseline(index_name=\"delta_nbr\")\n",
                "best_nbr_th = nbr_baseline.fit(val_ds)\n",
                "nbr_test_metrics = nbr_baseline.evaluate(test_ds)\n",
                "nbr_indian_metrics = nbr_baseline.evaluate(indian_ds)\n",
                "\n",
                "print(f\"Delta-NBR Optimal Threshold: {best_nbr_th:.3f}\")\n",
                "print(f\"Primary Test  -> Precision: {nbr_test_metrics['precision']:.4f}, Recall: {nbr_test_metrics['recall']:.4f}, F1: {nbr_test_metrics['f1']:.4f}, IoU: {nbr_test_metrics['iou']:.4f}\")\n",
                "print(f\"Indian AOI    -> Precision: {nbr_indian_metrics['precision']:.4f}, Recall: {nbr_indian_metrics['recall']:.4f}, F1: {nbr_indian_metrics['f1']:.4f}, IoU: {nbr_indian_metrics['iou']:.4f}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 4. Baseline 2: Early-Concatenation U-Net (Torres et al., 2021)\n",
                "\n",
                "Concatenates optical channels from T0 and T1 at the input layer (14 channels total) and trains standard encoder-decoder U-Net with cross-entropy and dice loss.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class S2OnlyPatchDataset(torch.utils.data.Dataset):\n",
                "    def __init__(self, ds):\n",
                "        self.ds = ds\n",
                "    def __len__(self):\n",
                "        return len(self.ds)\n",
                "    def __getitem__(self, idx):\n",
                "        x0, x1, y = self.ds[idx]\n",
                "        return x0[:7], x1[:7], y\n",
                "\n",
                "s2_train = S2OnlyPatchDataset(train_ds)\n",
                "s2_val = S2OnlyPatchDataset(val_ds)\n",
                "s2_test = S2OnlyPatchDataset(test_ds)\n",
                "s2_indian = S2OnlyPatchDataset(indian_ds)\n",
                "\n",
                "torres_unet = TorresEarlyConcatUNet(in_channels=14, out_channels=1, base_features=32)\n",
                "unet_history = train_unet_model(torres_unet, s2_train, s2_val, epochs=5, batch_size=4)\n",
                "unet_test_metrics = evaluate_torch_model(torres_unet, s2_test, model_name=\"Torres et al. (2021) Early-Concat U-Net\")\n",
                "unet_indian_metrics = evaluate_torch_model(torres_unet, s2_indian, model_name=\"Torres et al. (2021) Early-Concat U-Net\")\n",
                "\n",
                "print(f\"Torres U-Net Test F1: {unet_test_metrics['f1']:.4f}, IoU: {unet_test_metrics['iou']:.4f}\")\n",
                "print(f\"Torres U-Net Indian AOI F1: {unet_indian_metrics['f1']:.4f}, IoU: {unet_indian_metrics['iou']:.4f}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 5. Improved Model: Multimodal Siamese U-Net (Sentinel-1 SAR + Sentinel-2 Optical)\n",
                "\n",
                "Twin encoder branches extract deep spatial representations from T0 and T1 with shared weights. Feature differences feed decoder skip connections. Sentinel-1 C-SAR backscatter ensures robust canopy volume detection even through cloud cover.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "siamese_model = MultimodalSiameseUNet(in_channels=10, out_channels=1, base_filters=32)\n",
                "siamese_history = train_siamese_model(siamese_model, train_ds, val_ds, epochs=5, batch_size=4)\n",
                "siamese_test_metrics = evaluate_siamese_model(siamese_model, test_ds, model_name=\"Multimodal Siamese U-Net\")\n",
                "siamese_indian_metrics = evaluate_siamese_model(siamese_model, indian_ds, model_name=\"Multimodal Siamese U-Net\")\n",
                "\n",
                "print(f\"Siamese U-Net Test F1: {siamese_test_metrics['f1']:.4f}, IoU: {siamese_test_metrics['iou']:.4f}\")\n",
                "print(f\"Siamese U-Net Indian AOI F1: {siamese_indian_metrics['f1']:.4f}, IoU: {siamese_indian_metrics['iou']:.4f}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 6. Vector Polygonization & Hectare Accounting\n",
                "\n",
                "Converts binary change probability masks into valid GeoJSON polygons with exact surface area in hectares (at 10m GSD, 1 pixel = 0.01 ha):\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Extract sample loss predictions on test tile\n",
                "x0_t, x1_t, y_t = test_ds[0]\n",
                "with torch.no_grad():\n",
                "    prob_map = siamese_model(x0_t.unsqueeze(0), x1_t.unsqueeze(0)).squeeze().numpy()\n",
                "\n",
                "binary_mask = (prob_map >= 0.50).astype(np.uint8)\n",
                "test_bbox = [82.55, 22.75, 82.75, 22.95]\n",
                "\n",
                "geojson_output = raster_mask_to_geojson_polygons(\n",
                "    binary_mask=binary_mask,\n",
                "    confidence_map=prob_map,\n",
                "    bbox=test_bbox,\n",
                "    min_pixels=6,\n",
                "    pixel_size_meters=10.0,\n",
                ")\n",
                "\n",
                "print(f\"Total Loss Polygons Delineated: {geojson_output['total_polygons']}\")\n",
                "print(f\"Total Deforestation Area:       {geojson_output['total_hectares_lost']} Hectares\")\n",
                "if geojson_output['geojson']['features']:\n",
                "    first_poly = geojson_output['geojson']['features'][0]\n",
                "    print(f\"Sample Feature Properties: {first_poly['properties']}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 7. Interactive Before/After Visualizations\n",
                "\n",
                "Visualizing true-color Sentinel-2 composites before (T0) and after (T1), along with the Siamese U-Net loss polygon overlay.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)\n",
                "\n",
                "rgb0 = np.clip(np.stack([x0_t[2], x0_t[1], x0_t[0]], axis=-1) * 3.0, 0, 1)\n",
                "rgb1 = np.clip(np.stack([x1_t[2], x1_t[1], x1_t[0]], axis=-1) * 3.0, 0, 1)\n",
                "\n",
                "axes[0].imshow(rgb0)\n",
                "axes[0].set_title(\"Before Disturbance (T0: 2020) True-Color\")\n",
                "axes[0].axis(\"off\")\n",
                "\n",
                "axes[1].imshow(rgb1)\n",
                "axes[1].set_title(\"After Disturbance (T1: 2021) True-Color\")\n",
                "axes[1].axis(\"off\")\n",
                "\n",
                "axes[2].imshow(rgb1)\n",
                "axes[2].imshow(binary_mask, cmap=\"Reds\", alpha=0.55)\n",
                "axes[2].set_title(\"Siamese U-Net Delineated Loss (Red)\")\n",
                "axes[2].axis(\"off\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join(os.path.dirname(__file__), "forest_deforestation_training.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"SUCCESS: Created {out_path}")

