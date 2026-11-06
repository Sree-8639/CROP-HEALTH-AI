"""
experiments/run_all.py:
Master reproducibility runner for the Crop Health AI Multimodal Earth Observation Suite.
Executes all experimental modules with fixed seeds and YAML configurations:
1. Limitations Section Generator (paper_assets/LIMITATIONS.md & limitations.tex)
2. Crop Vigour Anomaly & Multimodal Fusion (experiments/run_crop_experiments.py)
3. Forest Disturbance & Canopy Change Detection (experiments/run_forest_experiments.py)
4. Urban Heat Island & Thermal Downscaling (experiments/run_heat_experiments.py)

Enforces:
- 100% deterministic reproducibility with fixed seeds (seed=42)
- Zero untracked numbers: all metrics in UI and paper originate directly from these runs
- Complete artifact verification across paper_assets/ and results/
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from experiments.generate_limitations import generate_limitations_documents
from experiments.check_paper_tables import check_all_paper_tables
from experiments.run_crop_experiments import main as run_crop
from experiments.run_forest_experiments import main as run_forest
from experiments.run_heat_experiments import main as run_heat


REQUIRED_OUTPUTS = [
    # Limitations & Claims
    "paper_assets/LIMITATIONS.md",
    "paper_assets/limitations.tex",
    "paper_assets/CLAIMS.md",
    # JSON Metrics
    "results/crop_metrics.json",
    "results/forest_metrics.json",
    "results/heat_metrics.json",
    "results/forest_overlays.json",
    "results/heat_overlays.json",
    "backend/results/crop_metrics.json",
    "backend/results/forest_metrics.json",
    "backend/results/heat_metrics.json",
    "backend/results/forest_overlays.json",
    "backend/results/heat_overlays.json",
    "frontend/public/crop_metrics.json",
    "frontend/public/forest_metrics.json",
    "frontend/public/heat_metrics.json",
    "frontend/public/forest_overlays.json",
    "frontend/public/heat_overlays.json",
    # Tables
    "paper_assets/tables/table1_crop_metrics.csv",
    "paper_assets/tables/table1_crop_metrics.md",
    "paper_assets/tables/table1_crop_metrics.tex",
    "paper_assets/tables/table1b_crop_confusion_matrix.csv",
    "paper_assets/tables/table1b_crop_confusion_matrix.md",
    "paper_assets/tables/table1b_crop_confusion_matrix.tex",
    "paper_assets/tables/table2_forest_metrics.csv",
    "paper_assets/tables/table2_forest_metrics.md",
    "paper_assets/tables/table2_forest_metrics.tex",
    "paper_assets/tables/table2b_forest_confusion_matrix.csv",
    "paper_assets/tables/table2b_forest_confusion_matrix.md",
    "paper_assets/tables/table2b_forest_confusion_matrix.tex",
    "paper_assets/tables/table3_heat_metrics.csv",
    "paper_assets/tables/table3_heat_metrics.md",
    "paper_assets/tables/table3_heat_metrics.tex",
    "paper_assets/tables/table4_ablation_summary.csv",
    "paper_assets/tables/table4_ablation_summary.md",
    "paper_assets/tables/table4_ablation_summary.tex",
    # Figures
    "paper_assets/figures/fig1_crop_vigour_anomaly.png",
    "paper_assets/figures/fig1b_crop_confusion_matrix.png",
    "paper_assets/figures/fig2_forest_canopy_disturbance.png",
    "paper_assets/figures/fig2b_forest_confusion_matrix.png",
    "paper_assets/figures/fig3_urban_heat_downscaling.png",
    "paper_assets/figures/fig4_model_comparisons.png",
]


def run_master_experiment_suite(modules=None, audit_only=False):
    start_total = time.time()
    modules = modules or ["limitations", "crop", "forest", "heat"]

    print("=" * 80)
    print("MASTER EXPERIMENTAL PIPELINE — REPRODUCIBLE SCIENTIFIC BENCHMARK")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    if audit_only:
        print("Mode: AUDIT ONLY (verifying existing generated artifacts)")
    else:
        print(f"Modules Selected: {', '.join(modules)}")
    print("=" * 80)

    if not audit_only:
        # 1. Limitations Generator
        if "limitations" in modules:
            t0 = time.time()
            print("\n>>> [1/4] Generating Scientific Limitations & Disclosures...")
            generate_limitations_documents()
            print(f"    Completed in {time.time() - t0:.2f}s")

        # 2. Crop Vigour Anomaly
        if "crop" in modules:
            t0 = time.time()
            print("\n>>> [2/4] Executing Crop Vigour Anomaly Experiments...")
            run_crop()
            print(f"    Completed in {time.time() - t0:.2f}s")

        # 3. Forest Disturbance
        if "forest" in modules:
            t0 = time.time()
            print("\n>>> [3/4] Executing Forest Disturbance & Canopy Change Experiments...")
            run_forest()
            print(f"    Completed in {time.time() - t0:.2f}s")

        # 4. Urban Heat Downscaling
        if "heat" in modules:
            t0 = time.time()
            print("\n>>> [4/4] Executing Urban Heat Island & Thermal Downscaling Experiments...")
            run_heat()
            print(f"    Completed in {time.time() - t0:.2f}s")

    # Verification & Deep Content Audit
    print("\n" + "=" * 80)
    print("DEEP CONTENT AUDIT OF EXPERIMENTAL ARTIFACTS IN paper_assets/ & results/")
    print("=" * 80)

    missing = []
    verified = []
    for rel_path in REQUIRED_OUTPUTS:
        full_path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            size_kb = os.path.getsize(full_path) / 1024.0
            verified.append((rel_path, f"{size_kb:.1f} KB"))
        else:
            missing.append(rel_path)

    for path, size in verified:
        print(f"  [OK] {path:<55} ({size})")

    if missing:
        print("\n[WARNING] The following expected artifacts were not found:")
        for m in missing:
            print(f"  [MISSING] {m}")
        raise RuntimeError(f"Audit failed: {len(missing)} missing artifacts.")

    # -------------------------------------------------------------
    # Content & Cross-Directory Synchronization Validation
    # -------------------------------------------------------------
    print("\n--- Cross-Directory Synchronization Validation ---")
    sync_files = [
        "crop_metrics.json",
        "forest_metrics.json",
        "heat_metrics.json",
        "forest_overlays.json",
        "heat_overlays.json",
    ]
    for m_file in sync_files:
        p_res = os.path.join(BASE_DIR, "results", m_file)
        p_back = os.path.join(BASE_DIR, "backend", "results", m_file)
        p_front = os.path.join(BASE_DIR, "frontend", "public", m_file)

        with open(p_res, "r", encoding="utf-8") as f:
            d_res = json.load(f)
        with open(p_back, "r", encoding="utf-8") as f:
            d_back = json.load(f)
        with open(p_front, "r", encoding="utf-8") as f:
            d_front = json.load(f)

        if d_res != d_back:
            raise ValueError(f"Content mismatch between results/{m_file} and backend/results/{m_file}!")
        if d_res != d_front:
            raise ValueError(f"Content mismatch between results/{m_file} and frontend/public/{m_file}!")

        print(f"  [SYNC VERIFIED] {m_file} is identical across results/, backend/results/, and frontend/public/.")

    # -------------------------------------------------------------
    # Content Schema & Numerical Range Auditing
    # -------------------------------------------------------------
    print("\n--- Content Schema & Numerical Metric Range Validation ---")

    # 1. Crop Metrics Schema & Ranges
    with open(os.path.join(BASE_DIR, "results", "crop_metrics.json"), "r", encoding="utf-8") as f:
        crop_data = json.load(f)
    crop_required_keys = ["project", "data_source", "timestamp_utc", "aoi", "dataset_summary", "baseline_method", "improved_method"]
    for k in crop_required_keys:
        if k not in crop_data:
            raise KeyError(f"crop_metrics.json missing required key: '{k}'")
    b_acc = crop_data["baseline_method"]["metrics"]["accuracy"]["mean"]
    i_acc = crop_data["improved_method"]["metrics"]["accuracy"]["mean"]
    i_f1 = crop_data["improved_method"]["metrics"]["f1_macro"]["mean"]
    if not (0.0 <= b_acc <= 1.0 and 0.0 <= i_acc <= 1.0 and 0.0 <= i_f1 <= 1.0):
        raise ValueError(f"Crop accuracy/F1 out of range [0, 1]: b_acc={b_acc}, i_acc={i_acc}, i_f1={i_f1}")
    print(f"  [RANGE OK] Crop: Descals Acc={b_acc:.4f}, Improved Acc={i_acc:.4f}, Macro F1={i_f1:.4f}")

    # 2. Forest Metrics Schema & Ranges
    with open(os.path.join(BASE_DIR, "results", "forest_metrics.json"), "r", encoding="utf-8") as f:
        forest_data = json.load(f)
    forest_required_keys = ["project", "data_source", "timestamp_utc", "reference_dataset", "model_performance_test_region", "out_of_region_generalisation_test"]
    for k in forest_required_keys:
        if k not in forest_data:
            raise KeyError(f"forest_metrics.json missing required key: '{k}'")
    primary_models = forest_data["model_performance_test_region"]["primary_test_region"]
    for m_name, m_eval in primary_models.items():
        f1_val = m_eval.get("f1", 0.0)
        iou_val = m_eval.get("iou", 0.0)
        if not (0.0 <= f1_val <= 1.0 and 0.0 <= iou_val <= 1.0):
            raise ValueError(f"Forest model {m_name} metrics out of range [0, 1]: F1={f1_val}, IoU={iou_val}")
    print(f"  [RANGE OK] Forest: All {len(primary_models)} models have valid F1/IoU in [0, 1].")

    # 3. Heat Metrics Schema & Ranges
    with open(os.path.join(BASE_DIR, "results", "heat_metrics.json"), "r", encoding="utf-8") as f:
        heat_data = json.load(f)
    heat_required_keys = ["project", "data_source", "timestamp_utc", "aoi_metadata", "sensor_specifications", "downscaling_evaluation_spatial_split", "uhi_hotspots_visakhapatnam", "tree_planting_priority_program"]
    for k in heat_required_keys:
        if k not in heat_data:
            raise KeyError(f"heat_metrics.json missing required key: '{k}'")
    held_out = heat_data["downscaling_evaluation_spatial_split"]["held_out_metrics"]
    r2_val = held_out["r2"]
    rmse_val = held_out["rmse_celsius"]
    mae_val = held_out["mae_celsius"]
    if not (-1.0 <= r2_val <= 1.0 and 0.0 < rmse_val < 30.0 and 0.0 < mae_val < 30.0):
        raise ValueError(f"Heat downscaling metrics out of range: R2={r2_val}, RMSE={rmse_val}, MAE={mae_val}")
    print(f"  [RANGE OK] Heat: Downscaling R2={r2_val:.4f}, RMSE={rmse_val:.3f} °C, MAE={mae_val:.3f} °C.")

    # -------------------------------------------------------------
    # Strict Cross-Reference: Check Paper Tables vs Results JSONs
    # -------------------------------------------------------------
    print("\n--- Exhaustive Paper Table vs. JSON Number Validation ---")
    check_all_paper_tables()

    # -------------------------------------------------------------
    # Mandatory Data Source (REAL vs SIMULATED) Enforcement
    # -------------------------------------------------------------
    print("\n--- Data Source Provenance Check (REAL vs. SIMULATED) ---")
    simulated_found = []
    for m_file, data_dict in [("crop_metrics.json", crop_data), ("forest_metrics.json", forest_data), ("heat_metrics.json", heat_data)]:
        ds = data_dict.get("data_source")
        if not ds:
            raise ValueError(f"Mandatory 'data_source' field missing in results/{m_file}!")
        if ds == "SIMULATED":
            simulated_found.append((m_file, ds))
            print(f"  [SIMULATED DETECTED] {m_file}: data_source = '{ds}'")
        elif ds == "REAL":
            print(f"  [REAL VERIFIED] {m_file}: data_source = 'REAL'")
        else:
            raise ValueError(f"Invalid data_source '{ds}' in {m_file}. Must be 'REAL' or 'SIMULATED'.")

    total_duration = time.time() - start_total
    print(f"\nTotal Suite Duration: {total_duration:.2f} seconds.")
    print("=" * 80)

    if simulated_found:
        msg = (
            f"\n[REPRODUCIBILITY AUDIT FAILED]: {len(simulated_found)} metric files contain 'SIMULATED' data:\n"
            + "\n".join(f"  - {f}: data_source = '{ds}'" for f, ds in simulated_found)
            + "\n\nIn accordance with DATA_PROVENANCE.md, production certification mandates 'REAL' satellite data.\n"
            + "To bypass this failure for developmental/offline benchmarking, pass --allow-simulated."
        )
        if not getattr(run_master_experiment_suite, "allow_simulated", False):
            raise RuntimeError(msg)
        else:
            print("[NOTICE] --allow-simulated flag active. Proceeding despite SIMULATED data source.")

    print(f"\n[AUDIT PASSED] All {len(verified)} required artifacts successfully generated and content-validated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run master reproducible experimental pipeline.")
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=["limitations", "crop", "forest", "heat"],
        default=["limitations", "crop", "forest", "heat"],
        help="Specify modules to run",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Skip training and execution, and perform only the completeness audit of all generated artifacts.",
    )
    parser.add_argument(
        "--allow-simulated",
        action="store_true",
        help="Permit SIMULATED data source without failing the audit suite (development only).",
    )
    args = parser.parse_args()
    run_master_experiment_suite.allow_simulated = args.allow_simulated
    run_master_experiment_suite(args.modules, audit_only=args.audit_only)

