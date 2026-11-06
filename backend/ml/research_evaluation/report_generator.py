"""
Crop Health AI — Research Evaluation & Baseline Comparison Report Generator (Phase 21.9)

Synthesizes scientific evaluation results across mandatory baselines and trained ML models.
Generates structured JSON artifacts and publication-grade Markdown reports.
"""

from typing import Dict, Any, List, Optional
import datetime
import json
import os


class ResearchEvaluationReportGenerator:
    """Generates comprehensive scientific evaluation and benchmark comparison reports."""

    @staticmethod
    def generate_comparison_summary(
        baseline_results: Dict[str, Dict[str, Any]],
        ml_model_results: Dict[str, Any],
        dataset_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Synthesizes baseline vs ML model comparisons into a structured result object.
        """
        persistence_mae = baseline_results.get("LastObservation_Persistence", {}).get("metrics", {}).get("mae", 999.0)
        ml_mae = ml_model_results.get("metrics", {}).get("mae", 999.0)

        delta_mae = ml_mae - persistence_mae
        pct_improvement = ((persistence_mae - ml_mae) / persistence_mae) * 100.0 if persistence_mae > 0 else 0.0

        ml_beats_persistence = ml_mae < persistence_mae

        summary = {
            "evaluation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "target_variable": "Future Satellite-Observed NDVI (t+h)",
            "task_type": "Satellite Vegetation-Index Time-Series Regression",
            "dataset_info": {
                "total_samples": dataset_meta.get("total_records"),
                "unique_fields": dataset_meta.get("unique_fields_count") or dataset_meta.get("unique_fields"),
                "temporal_span": dataset_meta.get("temporal_span") or dataset_meta.get("temporal_range") or {},
            },
            "models_evaluated": {
                "baselines": list(baseline_results.keys()),
                "ml_model": ml_model_results.get("model_name"),
            },
            "baseline_metrics": baseline_results,
            "ml_model_metrics": ml_model_results,
            "comparison_against_persistence": {
                "persistence_mae": persistence_mae,
                "ml_model_mae": ml_mae,
                "delta_mae": round(delta_mae, 4),
                "percentage_improvement": round(pct_improvement, 2),
                "ml_outperformed_baseline": ml_beats_persistence,
            },
            "scientific_recommendation": (
                "Deploy ML model for experimental index forecasting."
                if ml_beats_persistence
                else "Retain Persistence baseline. The machine learning model did not consistently outperform last observation on this validation partition."
            ),
            "scientific_caveat": (
                "Predicting future NDVI validates satellite spectral-index forecasting. "
                "It does not independently establish crop biomass, official yield, disease etiology, or biological plant health."
            ),
        }
        return summary

    @staticmethod
    def generate_markdown_report(summary: Dict[str, Any]) -> str:
        """Renders evaluation summary into an audit-ready Markdown document."""
        ds = summary.get("dataset_info", {})
        cmp = summary.get("comparison_against_persistence", {})
        ml_res = summary.get("ml_model_metrics", {})
        baselines = summary.get("baseline_metrics", {})

        temporal = ds.get("temporal_span") or {}
        start_t = temporal.get("start") or temporal.get("start_date") or "N/A"
        end_t = temporal.get("end") or temporal.get("end_date") or "N/A"

        md = f"""# Scientific Model Evaluation & Baseline Comparison Report

**Task**: {summary.get('task_type')}  
**Target Variable**: `{summary.get('target_variable')}`  
**Evaluation Timestamp**: {summary.get('evaluation_timestamp')}  

---

## 1. Dataset & Validation Scope
- **Total Evaluated Records**: {ds.get('total_samples')}
- **Independent Field Parcels**: {ds.get('unique_fields')}
- **Temporal Span**: {start_t} to {end_t}
- **Data Provenance**: Copernicus Data Space Ecosystem (CDSE) Sentinel-2 L2A & Sentinel-1 GRD

---

## 2. Performance Comparison Matrix

| Model Architecture | Model Role | Test MAE | Test RMSE | Test R² | Mean Bias |
|---|---|---|---|---|---|
"""
        for b_name, b_data in baselines.items():
            m = b_data.get("metrics", {})
            md += f"| **{b_name}** | Mandatory Baseline | {m.get('mae')} | {m.get('rmse')} | {m.get('r2')} | {m.get('mean_bias')} |\n"

        ml_m = ml_res.get("metrics", {})
        md += f"| **{ml_res.get('model_name', 'HistGradientBoosting')}** | Primary ML Model | **{ml_m.get('mae')}** | **{ml_m.get('rmse')}** | **{ml_m.get('r2')}** | **{ml_m.get('mean_bias')}** |\n\n"

        md += f"""---

## 3. Baseline Comparison Delta
- **Persistence Baseline MAE**: `{cmp.get('persistence_mae')}`
- **Machine Learning Model MAE**: `{cmp.get('ml_model_mae')}`
- **Absolute Delta (Δ MAE)**: `{cmp.get('delta_mae')}`
- **Relative Difference**: `{cmp.get('percentage_improvement')}%`
- **Outperformed Persistence**: `{'YES' if cmp.get('ml_outperformed_baseline') else 'NO'}`

---

## 4. Scientific Recommendation & Limitations
> **Operational Status**: {summary.get('scientific_recommendation')}

> [!IMPORTANT]
> **Scientific Integrity Caveat**: {summary.get('scientific_caveat')}
"""
        return md
