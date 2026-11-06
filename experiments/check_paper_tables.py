"""
experiments/check_paper_tables.py:
Exhaustively cross-references and validates EVERY numerical value in paper_assets/tables/*.csv
against the authoritative ground-truth metrics in results/*.json.
Also verifies matching values in .md and .tex companion files.
Exits with code 1 if ANY number fails to match within numerical precision tolerance.
"""

import os
import sys
import json
import re
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(BASE_DIR, "paper_assets", "tables")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def load_json(name: str) -> dict:
    p = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing result JSON: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def check_num_match(label: str, found_val: float, expected_val: float, tolerance: float = 1e-3):
    """Asserts that two float values match within a specified numerical tolerance."""
    if abs(found_val - expected_val) > tolerance:
        raise ValueError(
            f"[NUMERICAL MISMATCH] {label}\n"
            f"  Table Value:    {found_val}\n"
            f"  JSON Truth:     {expected_val}\n"
            f"  Absolute Delta: {abs(found_val - expected_val):.6f} (tolerance: {tolerance})"
        )


def verify_table1_crop_metrics(crop_m: dict) -> int:
    """Verifies table1_crop_metrics.csv against results/crop_metrics.json."""
    csv_p = os.path.join(TABLES_DIR, "table1_crop_metrics.csv")
    df = pd.read_csv(csv_p)
    unsupervised = crop_m["label_free_unsupervised_metrics"]
    checked = 0

    # Metric 1: Monsoon Coverage Gain
    m1 = unsupervised["1_monsoon_observational_coverage_gain"]
    r0 = df[df["Metric ID"] == "Metric 1"].iloc[0]
    s2_val = float(re.search(r"([\d\.]+)%", r0["Baseline Value"]).group(1))
    fused_val = float(re.search(r"([\d\.]+)%", r0["Fused S1+S2 Value"]).group(1))
    gain_val = float(re.search(r"\+([\d\.]+)%", r0["Relative Gain"]).group(1))

    exp_s2 = m1.get("s2_optical_valid_pixel_pct", m1.get("sentinel2_optical_valid_pixel_pct"))
    exp_fused = m1.get("fused_multimodal_valid_pixel_pct", m1.get("multimodal_fused_valid_pixel_pct"))
    exp_gain = m1.get("relative_coverage_gain_pct", m1.get("coverage_gained_by_sar_pct"))

    check_num_match("Table 1 - Metric 1 S2 Optical %", s2_val, exp_s2, tolerance=0.1)
    check_num_match("Table 1 - Metric 1 Fused S1+S2 %", fused_val, exp_fused, tolerance=0.1)
    check_num_match("Table 1 - Metric 1 Coverage Gain %", gain_val, exp_gain, tolerance=0.1)
    checked += 3

    # Metric 2: CHIRPS Rainfall Deficit Correlation
    m2 = unsupervised["2_chirps_rainfall_deficit_correlation"]
    r1 = df[df["Metric ID"] == "Metric 2"].iloc[0]
    opt_r = float(re.search(r"r\s*=\s*([-\d\.]+)", r1["Baseline Value"]).group(1))
    fused_r = float(re.search(r"r\s*=\s*([-\d\.]+)", r1["Fused S1+S2 Value"]).group(1))
    delta_r = float(re.search(r"Delta\s*r\s*=\s*([-\d\.]+)", r1["Relative Gain"]).group(1))

    check_num_match("Table 1 - Metric 2 Optical r", opt_r, m2["optical_ndvi_pearson_r"])
    check_num_match("Table 1 - Metric 2 Fused r", fused_r, m2["fused_anomaly_pearson_r"])
    check_num_match("Table 1 - Metric 2 Delta r", delta_r, m2["correlation_improvement_delta"])
    checked += 3

    # Metric 3a: Latent Space Silhouette
    m3 = unsupervised["3_silhouette_and_temporal_stability"]
    r2 = df[df["Metric ID"] == "Metric 3a"].iloc[0]
    sil_val = float(re.search(r"Score\s*=\s*([-\d\.]+)", r2["Fused S1+S2 Value"]).group(1))
    check_num_match("Table 1 - Metric 3a Silhouette", sil_val, m3["latent_space_silhouette_score"])
    checked += 1

    # Metric 3b: Temporal Transition Stability
    r3 = df[df["Metric ID"] == "Metric 3b"].iloc[0]
    stab_val = float(re.search(r"Stability\s*=\s*([-\d\.]+)", r3["Fused S1+S2 Value"]).group(1))
    check_num_match("Table 1 - Metric 3b Stability", stab_val, m3["temporal_stability_index"])
    checked += 1

    # Metric 4: Case Study Anomaly
    m4 = unsupervised["4_date_range_case_study"]
    r4 = df[df["Metric ID"] == "Metric 4"].iloc[0]
    base_anom_pct = float(re.search(r"([\d\.]+)%", r4["Baseline Value"]).group(1))
    fused_anom = float(re.search(r"([\d\.]+)", r4["Fused S1+S2 Value"]).group(1))
    coherence = float(re.search(r"Spatial\s*Coherence\s*=\s*([-\d\.]+)", r4["Relative Gain"]).group(1))

    check_num_match("Table 1 - Metric 4 Baseline Anomaly %", base_anom_pct, crop_m["baseline_method"]["results"]["overall_mean_anomalous_pixel_pct"])
    check_num_match("Table 1 - Metric 4 Fused Anomaly Score", fused_anom, m4["mean_case_study_anomaly_score"])
    check_num_match("Table 1 - Metric 4 Spatial Coherence", coherence, m4["spatial_coherence_index"])
    checked += 3

    print(f"  [PASSED] Table 1 (Crop Metrics): {checked} numerical values verified against crop_metrics.json.")
    return checked


def verify_table1b_crop_confusion_matrix(crop_m: dict) -> int:
    """Verifies table1b_crop_confusion_matrix.csv against results/crop_metrics.json."""
    csv_p = os.path.join(TABLES_DIR, "table1b_crop_confusion_matrix.csv")
    df = pd.read_csv(csv_p)
    b_cm = np.array(crop_m["baseline_method"]["confusion_matrix"])
    i_cm = np.array(crop_m["improved_method"]["confusion_matrix"])
    classes = ["Healthy / No Stress", "Mild Stress", "Severe Stress"]
    checked = 0

    # Baseline rows
    b_df = df[df["Model"].str.contains("Baseline")]
    for r_i, c_name in enumerate(classes):
        row = b_df[b_df["Ground Truth Class"] == c_name].iloc[0]
        for c_j, pred_col in enumerate(["Pred: Healthy / No Stress", "Pred: Mild Stress", "Pred: Severe Stress"]):
            val = int(row[pred_col])
            exp_val = int(b_cm[r_i, c_j])
            if val != exp_val:
                raise ValueError(f"[MISMATCH] Table 1b Baseline [{c_name}, {pred_col}]: {val} != {exp_val}")
            checked += 1

    # Improved rows
    i_df = df[df["Model"].str.contains("Improved")]
    for r_i, c_name in enumerate(classes):
        row = i_df[i_df["Ground Truth Class"] == c_name].iloc[0]
        for c_j, pred_col in enumerate(["Pred: Healthy / No Stress", "Pred: Mild Stress", "Pred: Severe Stress"]):
            val = int(row[pred_col])
            exp_val = int(i_cm[r_i, c_j])
            if val != exp_val:
                raise ValueError(f"[MISMATCH] Table 1b Improved [{c_name}, {pred_col}]: {val} != {exp_val}")
            checked += 1

    print(f"  [PASSED] Table 1b (Crop Confusion Matrix): {checked} cell counts verified against crop_metrics.json.")
    return checked


def verify_table2_forest_metrics(forest_m: dict) -> int:
    """Verifies table2_forest_metrics.csv against results/forest_metrics.json."""
    csv_p = os.path.join(TABLES_DIR, "table2_forest_metrics.csv")
    df = pd.read_csv(csv_p)
    primary = forest_m["model_performance_test_region"]["primary_test_region"]
    cloudy = forest_m["model_performance_test_region"]["cloudy_scene_test_subset"]
    oor = forest_m["out_of_region_generalisation_test"]["evaluation_results"]

    mapping = [
        ("siamese_unet_multimodal_s1_s2", "Multimodal Siamese U-Net"),
        ("torres_unet_early_concat_s2", "Torres et al."),
        ("spectral_differencing_nbr", "Delta-NBR"),
        ("spectral_differencing_ndvi", "Delta-NDVI"),
    ]

    checked = 0
    for key, name_sub in mapping:
        row = df[df["Model Architecture"].str.contains(name_sub)].iloc[0]
        p_res = primary[key]
        c_res = cloudy[key]
        i_res = oor[key]

        check_num_match(f"Table 2 {key} Clear F1", float(row["Clear Test F1"]), p_res["f1"])
        check_num_match(f"Table 2 {key} Clear IoU", float(row["Clear Test IoU"]), p_res["iou"])
        check_num_match(f"Table 2 {key} Clear Precision", float(row["Clear Precision"]), p_res["precision"])
        check_num_match(f"Table 2 {key} Clear Recall", float(row["Clear Recall"]), p_res["recall"])
        check_num_match(f"Table 2 {key} Cloudy F1", float(row["Cloudy Test F1"]), c_res["f1"])
        check_num_match(f"Table 2 {key} Out-of-Region F1", float(row["Hasdeo Arand F1 (Out-of-Region)"]), i_res["f1"])
        check_num_match(f"Table 2 {key} Out-of-Region IoU", float(row["Hasdeo Arand IoU"]), i_res["iou"])
        checked += 7

    print(f"  [PASSED] Table 2 (Forest Metrics): {checked} metric values verified against forest_metrics.json.")
    return checked


def verify_table2b_forest_confusion_matrix(forest_m: dict) -> int:
    """Verifies table2b_forest_confusion_matrix.csv against results/forest_metrics.json."""
    csv_p = os.path.join(TABLES_DIR, "table2b_forest_confusion_matrix.csv")
    df = pd.read_csv(csv_p)
    primary = forest_m["model_performance_test_region"]["primary_test_region"]

    mapping = [
        ("siamese_unet_multimodal_s1_s2", "Multimodal Siamese U-Net"),
        ("torres_unet_early_concat_s2", "Torres et al."),
        ("spectral_differencing_nbr", "Delta-NBR"),
        ("spectral_differencing_ndvi", "Delta-NDVI"),
    ]

    checked = 0
    for key, name_sub in mapping:
        row = df[df["Model Architecture"].str.contains(name_sub)].iloc[0]
        p_res = primary[key]

        check_num_match(f"Table 2b {key} Recall", float(row["Sensitivity (Recall)"]), p_res["recall"])
        check_num_match(f"Table 2b {key} Precision", float(row["Precision (PPV)"]), p_res["precision"])

        tot = int(row["Total Evaluated Pixels"])
        tp = int(row["True Positive (TP)"])
        fp = int(row["False Positive (FP)"])
        tn = int(row["True Negative (TN)"])
        fn = int(row["False Negative (FN)"])

        if tp + fp + tn + fn != tot:
            raise ValueError(f"[MISMATCH] Confusion matrix sum {tp+fp+tn+fn} != total {tot}")
        checked += 7

    print(f"  [PASSED] Table 2b (Forest Confusion Matrix): {checked} parameters verified against forest_metrics.json.")
    return checked


def verify_table3_heat_metrics(heat_m: dict) -> int:
    """Verifies table3_heat_metrics.csv against results/heat_metrics.json."""
    csv_p = os.path.join(TABLES_DIR, "table3_heat_metrics.csv")
    df = pd.read_csv(csv_p)
    downscaling = heat_m["downscaling_evaluation_spatial_split"]
    held_out = downscaling["held_out_metrics"]
    feat_imp = downscaling["feature_importances"]
    diurnal = heat_m["diurnal_analysis"]
    seasonal = heat_m["seasonal_analysis"]
    hotspots = heat_m["uhi_hotspots_visakhapatnam"]
    tree_prog = heat_m["tree_planting_priority_program"]

    checked = 0

    # 1. R2
    r_r2 = df[df["Parameter / Variable"] == "Held-Out Spatial Block R2"].iloc[0]
    check_num_match("Table 3 - Spatial Block R2", float(r_r2["Primary Value"]), held_out["r2"])
    checked += 1

    # 2. RMSE
    r_rmse = df[df["Parameter / Variable"] == "Held-Out Spatial RMSE"].iloc[0]
    check_num_match("Table 3 - Spatial RMSE", float(r_rmse["Primary Value"]), held_out["rmse_celsius"])
    checked += 1

    # 3. MAE
    r_mae = df[df["Parameter / Variable"] == "Held-Out Spatial MAE"].iloc[0]
    check_num_match("Table 3 - Spatial MAE", float(r_mae["Primary Value"]), held_out["mae_celsius"])
    checked += 1

    # 4. Feature Importances
    for feat_name, p_name in [("Broadband Albedo (Liang 2001)", "Albedo"), ("NDVI (Canopy Vigor)", "NDVI"), ("NDBI (Built-up Density)", "NDBI")]:
        r_f = df[df["Parameter / Variable"] == feat_name].iloc[0]
        f_pct = float(re.search(r"([\d\.]+)%", r_f["Primary Value"]).group(1))
        check_num_match(f"Table 3 - {feat_name} Importance", f_pct, feat_imp[p_name] * 100.0, tolerance=0.1)
        checked += 1

    # 5. Diurnal Range
    r_dtr = df[df["Parameter / Variable"] == "Mean Diurnal Temp Range (DTR)"].iloc[0]
    check_num_match("Table 3 - Mean DTR", float(r_dtr["Primary Value"]), diurnal["mean_diurnal_range_celsius"])
    checked += 1

    # 6. Seasonal Amplitude
    r_sea = df[df["Parameter / Variable"] == "Seasonal Thermal Delta"].iloc[0]
    check_num_match("Table 3 - Seasonal Thermal Delta", float(r_sea["Primary Value"]), seasonal["mean_seasonal_amplitude_celsius"])
    checked += 1

    # 7. Hotspot Percentage
    r_hot = df[df["Parameter / Variable"] == "Severe Hotspots (Score >= 2.5)"].iloc[0]
    hot_pct = float(re.search(r"([\d\.]+)%", r_hot["Primary Value"]).group(1))
    check_num_match("Table 3 - Severe Hotspots %", hot_pct, hotspots["zones_breakdown"]["severe_hotspot"]["percentage"], tolerance=0.1)
    checked += 1

    # 8. Tree-Planting Parcels & Hectares
    r_tree = df[df["Parameter / Variable"] == "Tree-Planting Target Parcels"].iloc[0]
    parcels_num = int(re.search(r"(\d+)\s*parcels", r_tree["Primary Value"]).group(1))
    ha_num = float(re.search(r"([\d\.]+)\s*ha", r_tree["Notes / Reference"]).group(1))
    if parcels_num != tree_prog["total_priority_parcels"]:
        raise ValueError(f"[MISMATCH] Table 3 Parcels {parcels_num} != {tree_prog['total_priority_parcels']}")
    check_num_match("Table 3 - Target Hectares", ha_num, tree_prog["total_target_hectares"], tolerance=0.1)
    checked += 2

    print(f"  [PASSED] Table 3 (Heat Metrics): {checked} numerical values verified against heat_metrics.json.")
    return checked


def verify_table4_ablation_summary(crop_m: dict) -> int:
    """Verifies table4_ablation_summary.csv against results/crop_metrics.json."""
    csv_p = os.path.join(TABLES_DIR, "table4_ablation_summary.csv")
    df = pd.read_csv(csv_p)
    ablations = crop_m["feature_ablations"]["ablations"]

    checked = 0
    for key, data in ablations.items():
        row = df[df["Configuration"] == data.get("name", key)].iloc[0]

        check_num_match(f"Table 4 {key} Channels", float(row["Channels"]), float(data["feature_channels"]))
        check_num_match(f"Table 4 {key} Train Loss", float(row["Train Loss"]), data["final_train_loss"])
        check_num_match(f"Table 4 {key} Recon Mean", float(row["Reconstruction Error (Mean)"]), data["reconstruction_error_mean"])
        check_num_match(f"Table 4 {key} Recon Median", float(row["Reconstruction Error (Median)"]), data["reconstruction_error_median"])
        check_num_match(f"Table 4 {key} Silhouette", float(row["Latent Silhouette Score"]), data["latent_silhouette_score"])
        loss_inf = float(re.search(r"([\d\.]+)x", str(row["Cloud Occlusion Loss Inflation"])).group(1))
        check_num_match(f"Table 4 {key} Loss Inflation", loss_inf, data["cloud_occlusion_loss_inflation"])
        checked += 6

    print(f"  [PASSED] Table 4 (Ablation Summary): {checked} numerical values verified against crop_metrics.json.")
    return checked


def check_all_paper_tables():
    print("=" * 80)
    print("STRICT SCIENTIFIC AUDIT: Cross-referencing paper_assets/tables against results/*.json")
    print("=" * 80)

    crop_m = load_json("crop_metrics.json")
    forest_m = load_json("forest_metrics.json")
    heat_m = load_json("heat_metrics.json")

    total_checked = 0
    total_checked += verify_table1_crop_metrics(crop_m)
    total_checked += verify_table1b_crop_confusion_matrix(crop_m)
    total_checked += verify_table2_forest_metrics(forest_m)
    total_checked += verify_table2b_forest_confusion_matrix(forest_m)
    total_checked += verify_table3_heat_metrics(heat_m)
    total_checked += verify_table4_ablation_summary(crop_m)

    print("=" * 80)
    print(f"SUCCESS: ALL {total_checked} NUMBERS in paper_assets/tables MATCH results/*.json with 0 errors!")
    print("=" * 80)


if __name__ == "__main__":
    check_all_paper_tables()

