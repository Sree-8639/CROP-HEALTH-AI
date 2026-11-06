"""
ml/forest/pipeline.py:
End-to-end execution pipeline for real forest change detection:
- Real Hansen GFC v1.11 reference dataset (treecover2000, lossyear) paired with real CDSE Sentinel-2 and Sentinel-1
- Strict non-overlapping SPATIAL train/val/test split
- 4 Real Forest Regions across India and multiple observation years:
    1. Eastern Ghats Forest Reserve (Maredumilli / Papikonda Corridor, AP) - 2020-2021
    2. Hasdeo Arand Forest (Parsa East & Kete Basan Sector, Chhattisgarh) - 2020-2021
    3. Western Ghats / Wayanad-Nilgiris Montane Corridor (Kerala/Tamil Nadu) - 2021-2022
    4. Dandeli-Anshi / Kali Tiger Reserve Corridor (Western Ghats, Karnataka) - 2021-2022
- Cloudy-scene test evaluation subset (evaluating Sentinel-1 C-SAR microwave cloud penetration)
- Baselines:
    1. Spectral Differencing (Delta-NBR)
    2. Spectral Differencing (Delta-NDVI)
    3. SAR Differencing (Delta-VH backscatter drop in dB)
    4. Multimodal Differencing (Delta-NBR + Delta-VH)
    5. Torres et al. (2021) Early-Concat U-Net (with class-balanced loss)
- Improved: Multimodal Siamese U-Net (Sentinel-2 + Sentinel-1 with class-balanced loss)
- Hyperparameter tuning (decision thresholds): strictly on VALIDATION split only
- Scientific metrics: Precision, Recall, F1, IoU, and Hectares Lost
- Export to results/forest_metrics.json (with data_source = "REAL")
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.forest.hansen_dataset import (
    build_spatial_split_forest_datasets,
    PatchForestDataset,
)
from ml.forest.baselines import (
    SpectralDifferencingBaseline,
    SARDifferencingBaseline,
    MultimodalDifferencingBaseline,
    TorresEarlyConcatUNet,
    train_unet_model,
    optimize_threshold_on_val,
    evaluate_torch_model,
)
from ml.forest.siamese_unet import (
    MultimodalSiameseUNet,
    train_siamese_model,
    optimize_siamese_threshold_on_val,
    evaluate_siamese_model,
)
from ml.forest.evaluation import (
    raster_mask_to_geojson_polygons,
    generate_visual_overlays_base64,
)


def run_forest_pipeline(
    year_before: int = 2020,
    year_after: int = 2021,
    output_path: str = "results/forest_metrics.json",
    random_seed: int = 42,
    epochs_unet: int = 10,
    epochs_siamese: int = 12,
) -> Dict[str, Any]:
    print("=" * 80)
    print("RUNNING REAL MULTI-REGION FOREST DISTURBANCE PIPELINE")
    print("Reference Dataset: Hansen Global Forest Change (v1.11, Landsat 30m)")
    print("Imagery Sources:   Copernicus CDSE Sentinel-2 L2A + Sentinel-1 C-SAR GRD")
    print(f"Analysis Interval: {year_before} (T0) -> {year_after} (T1)")
    print("=" * 80)

    # 1. Dataset Generation & Spatial Splitting (4 Real Regions)
    print("\n[1/6] Loading real multimodal patch datasets with strict SPATIAL split across 4 regions...")
    dataset_bundle = build_spatial_split_forest_datasets(
        tile_size=512,
        patch_size=256,
        stride=64,
        year_before=year_before,
        year_after=year_after,
        random_seed=random_seed,
    )
    train_ds = dataset_bundle["train_dataset"]
    val_ds = dataset_bundle["val_dataset"]
    test_ds = dataset_bundle["test_dataset"]
    cloudy_test_ds = dataset_bundle["cloudy_test_dataset"]
    indian_ds = dataset_bundle["indian_aoi_dataset"]
    wayanad_ds = dataset_bundle["wayanad_dataset"]
    dandeli_ds = dataset_bundle["dandeli_dataset"]
    meta = dataset_bundle["spatial_metadata"]

    print(f"      Train Patches:       {len(train_ds)} (Region 1: Northern & Western Extent)")
    print(f"      Val Patches:         {len(val_ds)} (Region 1: North-Eastern Sector)")
    print(f"      Test Patches:        {len(test_ds)} (Region 1: South-Eastern Extent)")
    print(f"      Cloudy Test Patches: {len(cloudy_test_ds)} (Region 1: Optical Cloud Occlusion)")
    print(f"      Region 2 (Hasdeo):   {len(indian_ds)} patches ({meta['indian_aoi_details']['name']})")
    print(f"      Region 3 (Wayanad):  {len(wayanad_ds)} patches ({meta['wayanad_aoi_details']['name']})")
    print(f"      Region 4 (Dandeli):  {len(dandeli_ds)} patches ({meta['dandeli_aoi_details']['name']})")

    # 2. Baseline 1: Spectral Differencing (Delta-NBR & Delta-NDVI)
    print("\n[2/6] Evaluating Baseline 1: Spectral Differencing (Delta-NBR & Delta-NDVI)...")
    nbr_baseline = SpectralDifferencingBaseline(index_name="delta_nbr")
    nbr_thresh = nbr_baseline.fit(val_ds)
    nbr_test_eval = nbr_baseline.evaluate(test_ds)
    nbr_cloudy_eval = nbr_baseline.evaluate(cloudy_test_ds)
    nbr_indian_eval = nbr_baseline.evaluate(indian_ds)
    nbr_wayanad_eval = nbr_baseline.evaluate(wayanad_ds)
    nbr_dandeli_eval = nbr_baseline.evaluate(dandeli_ds)
    print(f"      Delta-NBR Optimal Val Threshold: {nbr_thresh:.3f}")
    print(f"      Delta-NBR Test Set   -> Prec: {nbr_test_eval['precision']:.4f}, Rec: {nbr_test_eval['recall']:.4f}, F1: {nbr_test_eval['f1']:.4f}, IoU: {nbr_test_eval['iou']:.4f}")
    print(f"      Delta-NBR Cloudy Set -> Prec: {nbr_cloudy_eval['precision']:.4f}, Rec: {nbr_cloudy_eval['recall']:.4f}, F1: {nbr_cloudy_eval['f1']:.4f}, IoU: {nbr_cloudy_eval['iou']:.4f}")

    ndvi_baseline = SpectralDifferencingBaseline(index_name="delta_ndvi")
    ndvi_thresh = ndvi_baseline.fit(val_ds)
    ndvi_test_eval = ndvi_baseline.evaluate(test_ds)
    ndvi_cloudy_eval = ndvi_baseline.evaluate(cloudy_test_ds)
    ndvi_indian_eval = ndvi_baseline.evaluate(indian_ds)
    ndvi_wayanad_eval = ndvi_baseline.evaluate(wayanad_ds)
    ndvi_dandeli_eval = ndvi_baseline.evaluate(dandeli_ds)
    print(f"      Delta-NDVI Test Set   -> Prec: {ndvi_test_eval['precision']:.4f}, Rec: {ndvi_test_eval['recall']:.4f}, F1: {ndvi_test_eval['f1']:.4f}, IoU: {ndvi_test_eval['iou']:.4f}")

    # 3. Baseline 2: SAR Differencing (Delta-VH) & Multimodal Differencing (Delta-NBR + Delta-VH)
    print("\n[3/6] Evaluating Baseline 2: SAR Differencing (Delta-VH) & Multimodal Differencing...")
    sar_baseline = SARDifferencingBaseline(vh_channel_idx=8)
    sar_thresh = sar_baseline.fit(val_ds)
    sar_test_eval = sar_baseline.evaluate(test_ds)
    sar_cloudy_eval = sar_baseline.evaluate(cloudy_test_ds)
    sar_indian_eval = sar_baseline.evaluate(indian_ds)
    sar_wayanad_eval = sar_baseline.evaluate(wayanad_ds)
    sar_dandeli_eval = sar_baseline.evaluate(dandeli_ds)
    print(f"      Delta-VH Optimal Val Threshold: {sar_thresh:.3f}")
    print(f"      Delta-VH Test Set   -> Prec: {sar_test_eval['precision']:.4f}, Rec: {sar_test_eval['recall']:.4f}, F1: {sar_test_eval['f1']:.4f}, IoU: {sar_test_eval['iou']:.4f}")
    print(f"      Delta-VH Cloudy Set -> Prec: {sar_cloudy_eval['precision']:.4f}, Rec: {sar_cloudy_eval['recall']:.4f}, F1: {sar_cloudy_eval['f1']:.4f}, IoU: {sar_cloudy_eval['iou']:.4f}")

    multi_diff_baseline = MultimodalDifferencingBaseline(nbr_idx=5, vh_idx=8)
    m_tn, m_tv = multi_diff_baseline.fit(val_ds)
    multi_test_eval = multi_diff_baseline.evaluate(test_ds)
    multi_cloudy_eval = multi_diff_baseline.evaluate(cloudy_test_ds)
    multi_indian_eval = multi_diff_baseline.evaluate(indian_ds)
    multi_wayanad_eval = multi_diff_baseline.evaluate(wayanad_ds)
    multi_dandeli_eval = multi_diff_baseline.evaluate(dandeli_ds)
    print(f"      Multimodal Differencing (NBR+VH) -> Clear F1: {multi_test_eval['f1']:.4f}, Cloudy F1: {multi_cloudy_eval['f1']:.4f}")

    # 4. Baseline 3: Torres et al. (2021) Early Concatenation U-Net (Class-Balanced Loss)
    print("\n[4/6] Training Baseline 3: Torres et al. (2021) Early-Concat U-Net (Class-Balanced Loss)...")
    class S2OnlyDataset(torch.utils.data.Dataset):
        def __init__(self, base_ds):
            self.base_ds = base_ds
        def __len__(self):
            return len(self.base_ds)
        def __getitem__(self, idx):
            x0, x1, y = self.base_ds[idx]
            return x0[:7], x1[:7], y

    s2_train = S2OnlyDataset(train_ds)
    s2_val = S2OnlyDataset(val_ds)
    s2_test = S2OnlyDataset(test_ds)
    s2_cloudy = S2OnlyDataset(cloudy_test_ds)
    s2_indian = S2OnlyDataset(indian_ds)
    s2_wayanad = S2OnlyDataset(wayanad_ds)
    s2_dandeli = S2OnlyDataset(dandeli_ds)

    torres_unet = TorresEarlyConcatUNet(in_channels=14, out_channels=1, base_features=32)
    unet_train_info = train_unet_model(torres_unet, s2_train, s2_val, epochs=epochs_unet, batch_size=4, pos_weight=15.0)
    torres_opt_th = optimize_threshold_on_val(torres_unet, s2_val)
    print(f"      Torres U-Net Optimal Val Threshold: {torres_opt_th:.3f}")

    unet_test_eval = evaluate_torch_model(torres_unet, s2_test, model_name="Torres et al. (2021) Early-Concat U-Net (S2 Only)", threshold=torres_opt_th)
    unet_cloudy_eval = evaluate_torch_model(torres_unet, s2_cloudy, model_name="Torres et al. (2021) Early-Concat U-Net (S2 Only)", threshold=torres_opt_th)
    unet_indian_eval = evaluate_torch_model(torres_unet, s2_indian, model_name="Torres et al. (2021) Early-Concat U-Net (S2 Only)", threshold=torres_opt_th)
    unet_wayanad_eval = evaluate_torch_model(torres_unet, s2_wayanad, model_name="Torres et al. (2021) Early-Concat U-Net (S2 Only)", threshold=torres_opt_th)
    unet_dandeli_eval = evaluate_torch_model(torres_unet, s2_dandeli, model_name="Torres et al. (2021) Early-Concat U-Net (S2 Only)", threshold=torres_opt_th)
    print(f"      Torres U-Net Test Set   -> Prec: {unet_test_eval['precision']:.4f}, Rec: {unet_test_eval['recall']:.4f}, F1: {unet_test_eval['f1']:.4f}, IoU: {unet_test_eval['iou']:.4f}")
    print(f"      Torres U-Net Cloudy Set -> Prec: {unet_cloudy_eval['precision']:.4f}, Rec: {unet_cloudy_eval['recall']:.4f}, F1: {unet_cloudy_eval['f1']:.4f}, IoU: {unet_cloudy_eval['iou']:.4f}")

    # 5. Improved Model: Multimodal Siamese U-Net (Class-Balanced Loss + Validation Threshold Search)
    print("\n[5/6] Training Improved Model: Multimodal Siamese U-Net (Class-Balanced Loss)...")
    siamese_fused = MultimodalSiameseUNet(in_channels=10, out_channels=1, base_filters=32)
    siamese_train_info = train_siamese_model(siamese_fused, train_ds, val_ds, epochs=epochs_siamese, batch_size=4, pos_weight=15.0)
    siamese_opt_th = optimize_siamese_threshold_on_val(siamese_fused, val_ds)
    print(f"      Siamese U-Net Optimal Val Threshold: {siamese_opt_th:.3f}")

    siamese_test_eval = evaluate_siamese_model(siamese_fused, test_ds, model_name="Multimodal Siamese U-Net (S1+S2 Fused)", threshold=siamese_opt_th)
    siamese_cloudy_eval = evaluate_siamese_model(siamese_fused, cloudy_test_ds, model_name="Multimodal Siamese U-Net (S1+S2 Fused)", threshold=siamese_opt_th)
    siamese_indian_eval = evaluate_siamese_model(siamese_fused, indian_ds, model_name="Multimodal Siamese U-Net (S1+S2 Fused)", threshold=siamese_opt_th)
    siamese_wayanad_eval = evaluate_siamese_model(siamese_fused, wayanad_ds, model_name="Multimodal Siamese U-Net (S1+S2 Fused)", threshold=siamese_opt_th)
    siamese_dandeli_eval = evaluate_siamese_model(siamese_fused, dandeli_ds, model_name="Multimodal Siamese U-Net (S1+S2 Fused)", threshold=siamese_opt_th)
    print(f"      Siamese U-Net Test Set   -> Prec: {siamese_test_eval['precision']:.4f}, Rec: {siamese_test_eval['recall']:.4f}, F1: {siamese_test_eval['f1']:.4f}, IoU: {siamese_test_eval['iou']:.4f}")
    print(f"      Siamese U-Net Cloudy Set -> Prec: {siamese_cloudy_eval['precision']:.4f}, Rec: {siamese_cloudy_eval['recall']:.4f}, F1: {siamese_cloudy_eval['f1']:.4f}, IoU: {siamese_cloudy_eval['iou']:.4f}")

    # Save weights
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "siamese_s1_s2_weights.pt")
    torch.save(siamese_fused.state_dict(), weights_path)

    # 6. Case Study & Polygonization
    print("\n[6/6] Generating GeoJSON Polygons and Vector Summaries for Hasdeo Arand...")
    indian_full_t0 = dataset_bundle["full_indian_tile"]["t0"]
    indian_full_t1 = dataset_bundle["full_indian_tile"]["t1"]
    indian_true_mask = dataset_bundle["full_indian_tile"]["mask"]

    siamese_fused.eval()
    pred_prob_map = np.zeros((512, 512), dtype=np.float32)
    count_map = np.zeros((512, 512), dtype=np.float32)

    with torch.no_grad():
        for r in range(0, 512 - 256 + 1, 64):
            for c in range(0, 512 - 256 + 1, 64):
                p0 = torch.from_numpy(indian_full_t0[:, r:r+256, c:c+256]).unsqueeze(0).float()
                p1 = torch.from_numpy(indian_full_t1[:, r:r+256, c:c+256]).unsqueeze(0).float()
                prob = siamese_fused(p0, p1).squeeze().numpy()
                pred_prob_map[r:r+256, c:c+256] += prob
                count_map[r:r+256, c:c+256] += 1.0

    count_map = np.maximum(count_map, 1.0)
    pred_prob_map /= count_map
    binary_loss_pred = (pred_prob_map >= siamese_opt_th).astype(np.uint8)

    indian_bbox = meta["indian_aoi_details"]["bbox"]
    vector_results = raster_mask_to_geojson_polygons(
        binary_mask=binary_loss_pred,
        confidence_map=pred_prob_map,
        bbox=indian_bbox,
        min_pixels=6,
        pixel_size_meters=10.0,
    )
    print(f"      Hasdeo PEKB Deforestation Detected: {vector_results['total_hectares_lost']} Hectares ({vector_results['total_polygons']} Polygons)")

    visual_overlays = generate_visual_overlays_base64(
        t0_tensor=indian_full_t0,
        t1_tensor=indian_full_t1,
        loss_mask=binary_loss_pred,
        confidence_map=pred_prob_map,
    )

    raw_ranking = [
        {"model": "Multimodal Siamese U-Net (S1+S2 Fused)", "f1_score": siamese_test_eval["f1"], "iou": siamese_test_eval["iou"], "precision": siamese_test_eval["precision"], "recall": siamese_test_eval["recall"], "cloudy_f1": siamese_cloudy_eval["f1"]},
        {"model": "Multimodal Differencing (Delta-NBR + Delta-VH)", "f1_score": multi_test_eval["f1"], "iou": multi_test_eval["iou"], "precision": multi_test_eval["precision"], "recall": multi_test_eval["recall"], "cloudy_f1": multi_cloudy_eval["f1"]},
        {"model": "SAR Differencing (Delta-VH dB)", "f1_score": sar_test_eval["f1"], "iou": sar_test_eval["iou"], "precision": sar_test_eval["precision"], "recall": sar_test_eval["recall"], "cloudy_f1": sar_cloudy_eval["f1"]},
        {"model": "Torres et al. (2021) Early-Concat U-Net (S2 Only)", "f1_score": unet_test_eval["f1"], "iou": unet_test_eval["iou"], "precision": unet_test_eval["precision"], "recall": unet_test_eval["recall"], "cloudy_f1": unet_cloudy_eval["f1"]},
        {"model": "Spectral Differencing (Delta-NBR)", "f1_score": nbr_test_eval["f1"], "iou": nbr_test_eval["iou"], "precision": nbr_test_eval["precision"], "recall": nbr_test_eval["recall"], "cloudy_f1": nbr_cloudy_eval["f1"]},
        {"model": "Spectral Differencing (Delta-NDVI)", "f1_score": ndvi_test_eval["f1"], "iou": ndvi_test_eval["iou"], "precision": ndvi_test_eval["precision"], "recall": ndvi_test_eval["recall"], "cloudy_f1": ndvi_cloudy_eval["f1"]},
    ]
    raw_ranking.sort(key=lambda x: x["f1_score"], reverse=True)
    metrics_ranking = [{"rank": idx + 1, **item} for idx, item in enumerate(raw_ranking)]

    output_dict = {
        "project": "Crop Health AI — Forest Disturbance & Canopy Change Monitoring",
        "data_source": "REAL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_dataset": {
            "name": "Hansen Global Forest Change (GFC v1.11)",
            "nominal_resolution": "30 meters (Landsat 7/8 ETM+/OLI)",
            "target_resolution": "10 meters (Sentinel-2 Optical & Sentinel-1 C-SAR)",
            "total_regions_evaluated": 4,
            "ground_truth_policy": "Strictly independent Hansen GFC v1.11 labels (treecover2000 >= 30, lossyear matching acquisition dates); NEVER derived from NBR/NDVI",
            "scientific_disclosure": "Hansen Global Forest Change labels are Landsat-derived at 30 m nominal resolution, with known label noise. Threshold search is strictly isolated to the validation set.",
        },
        "spatial_evaluation_protocol": {
            "methodology": "Strict non-overlapping geographic spatial partition. Hyperparameters (decision thresholds) optimized strictly on the validation set.",
            "patch_dimensions": "256x256 pixels at 10m Ground Sample Distance (GSD)",
            "spatial_metadata": meta,
        },
        "model_performance_test_region": {
            "primary_test_region": {
                "spectral_differencing_nbr": nbr_test_eval,
                "spectral_differencing_ndvi": ndvi_test_eval,
                "sar_differencing_vh": sar_test_eval,
                "multimodal_differencing_nbr_vh": multi_test_eval,
                "torres_unet_early_concat_s2": unet_test_eval,
                "siamese_unet_multimodal_s1_s2": siamese_test_eval,
            },
            "cloudy_scene_test_subset": {
                "spectral_differencing_nbr": nbr_cloudy_eval,
                "spectral_differencing_ndvi": ndvi_cloudy_eval,
                "sar_differencing_vh": sar_cloudy_eval,
                "multimodal_differencing_nbr_vh": multi_cloudy_eval,
                "torres_unet_early_concat_s2": unet_cloudy_eval,
                "siamese_unet_multimodal_s1_s2": siamese_cloudy_eval,
                "sar_resilience_analysis": {
                    "siamese_f1_retention_cloudy_pct": round((siamese_cloudy_eval["f1"] / max(siamese_test_eval["f1"], 0.001)) * 100, 2),
                    "sar_differencing_f1_retention_cloudy_pct": round((sar_cloudy_eval["f1"] / max(sar_test_eval["f1"], 0.001)) * 100, 2),
                    "torres_f1_retention_cloudy_pct": round((unet_cloudy_eval["f1"] / max(unet_test_eval["f1"], 0.001)) * 100, 2),
                },
            },
            "metrics_ranking": metrics_ranking,
        },
        "multi_region_generalisation_evaluation": {
            "region_2_hasdeo_arand": {
                "details": meta["indian_aoi_details"],
                "results": {
                    "spectral_differencing_nbr": nbr_indian_eval,
                    "spectral_differencing_ndvi": ndvi_indian_eval,
                    "sar_differencing_vh": sar_indian_eval,
                    "multimodal_differencing_nbr_vh": multi_indian_eval,
                    "torres_unet_early_concat_s2": unet_indian_eval,
                    "siamese_unet_multimodal_s1_s2": siamese_indian_eval,
                },
            },
            "region_3_wayanad_nilgiris": {
                "details": meta.get("wayanad_aoi_details", {}),
                "results": {
                    "spectral_differencing_nbr": nbr_wayanad_eval,
                    "spectral_differencing_ndvi": ndvi_wayanad_eval,
                    "sar_differencing_vh": sar_wayanad_eval,
                    "multimodal_differencing_nbr_vh": multi_wayanad_eval,
                    "torres_unet_early_concat_s2": unet_wayanad_eval,
                    "siamese_unet_multimodal_s1_s2": siamese_wayanad_eval,
                },
            },
            "region_4_dandeli_kali": {
                "details": meta.get("dandeli_aoi_details", {}),
                "results": {
                    "spectral_differencing_nbr": nbr_dandeli_eval,
                    "spectral_differencing_ndvi": ndvi_dandeli_eval,
                    "sar_differencing_vh": sar_dandeli_eval,
                    "multimodal_differencing_nbr_vh": multi_dandeli_eval,
                    "torres_unet_early_concat_s2": unet_dandeli_eval,
                    "siamese_unet_multimodal_s1_s2": siamese_dandeli_eval,
                },
            },
        },
        "out_of_region_generalisation_test": {
            "aoi_name": meta["indian_aoi_details"]["name"],
            "state_country": meta["indian_aoi_details"]["state"],
            "geographic_coordinates": meta["indian_aoi_details"]["bbox"],
            "forest_ecosystem": meta["indian_aoi_details"]["forest_type"],
            "threat_profile": meta["indian_aoi_details"]["threat_profile"],
            "evaluation_results": {
                "spectral_differencing_nbr": nbr_indian_eval,
                "spectral_differencing_ndvi": ndvi_indian_eval,
                "torres_unet_early_concat_s2": unet_indian_eval,
                "siamese_unet_multimodal_s1_s2": siamese_indian_eval,
            },
            "domain_shift_analysis": {
                "test_region_f1": siamese_test_eval["f1"],
                "indian_aoi_f1": siamese_indian_eval["f1"],
                "f1_retention_pct": round((siamese_indian_eval["f1"] / (siamese_test_eval["f1"] + 1e-6)) * 100, 2),
            },
            "vector_polygon_summary": {
                "total_polygons_delineated": vector_results["total_polygons"],
                "total_hectares_lost": vector_results["total_hectares_lost"],
                "total_loss_pixels_10m": vector_results["total_loss_pixels"],
            },
        },
        "visual_overlays": visual_overlays,
        "sample_geojson": vector_results["geojson"],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)

    backend_results_path = os.path.join(BASE_DIR, "backend", "results", "forest_metrics.json")
    frontend_public_path = os.path.join(BASE_DIR, "frontend", "public", "forest_metrics.json")
    
    os.makedirs(os.path.dirname(backend_results_path), exist_ok=True)
    with open(backend_results_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)
        
    os.makedirs(os.path.dirname(frontend_public_path), exist_ok=True)
    with open(frontend_public_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)

    print("\nSUCCESS: All forest disturbance benchmark metrics written to:")
    print(f" -> {output_path}")
    print(f" -> {backend_results_path}")
    print(f" -> {frontend_public_path}")

    return output_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Forest Disturbance & Change Detection Pipeline")
    parser.add_argument("--before", type=int, default=2020, help="Baseline observation year (T0)")
    parser.add_argument("--after", type=int, default=2021, help="Post-disturbance observation year (T1)")
    parser.add_argument("--output", type=str, default="results/forest_metrics.json", help="Path to export results JSON")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    args = parser.parse_args()

    run_forest_pipeline(
        year_before=args.before,
        year_after=args.after,
        output_path=args.output,
        epochs_siamese=args.epochs,
        epochs_unet=max(6, args.epochs - 3),
    )
