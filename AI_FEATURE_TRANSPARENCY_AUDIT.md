# AI Feature Transparency & Scientific Integrity Audit

## Scope
This document details the scientific claims, mathematical definitions, and validation statuses of all analytical pipelines in **Crop Health AI**.

---

## 1. Feature Classification Matrix

| Feature | Execution Type | Current Operational Status | Scientific Transparency Claim |
|---|---|---|---|
| **Vegetation Index Analysis** | Deterministic Spectral Arithmetic | **PRODUCTION ACTIVE** | Calculates vegetation and moisture-related spectral indices (NDVI, NDMI, NDRE, EVI) from calibrated Sentinel-2 MSI surface reflectance observations (L2A). |
| **Rule-Based Crop Condition Assessment** | Engineering Heuristics | **PRODUCTION ACTIVE** | Uses spectral-index thresholds and transparent engineering rules to summarize observed crop conditions into empirical vigor tiers. |
| **Multimodal Data Fusion** | Geospatial Reprojection & Colormap Raster Alignment | **PRODUCTION ACTIVE** | Aligns available Sentinel-1 SAR and Sentinel-2 optical observations into a common spatial feature grid (canonical UTM 10m). Default temporal matching window: ±3 days (configurable 1–14 days). |
| **Early Crop Stress Detection** | Calibrated Heuristic & Prototypical Classifier | **RESEARCH STATUS / DEMO ONLY** | Supervised production stress detection is currently unavailable pending independently verified agronomic ground-truth data. Demonstrates rule-based sensitivity and synthetic evaluation. |
| **Growth Prediction** | Multi-Temporal Time-Series Modeling | **BASELINE DEMONSTRATOR** | Provides baseline projections from historical satellite observations, subject to observation availability and validation limitations. |
| **Explainable AI & Uncertainty** | SHAP / Feature Attribution & Sensitivity | **ACTIVE ANALYTIC** | Displays feature attribution and sensitivity information where supported by available models and input data. |
| **Field Risk Mapping** | Multi-Criteria Spatial Segmentation | **HEURISTIC VISUALIZATION** | Visualizes spatial indicators and heuristic risk categories. These categories are not validated agronomic diagnoses. |

---

## 2. Model Benchmarks vs. Measured Runtime Performance

### Planned Target Disclaimers
In `/model-performance`:
- All metric cards display:
  > **"PLANNED TARGET — NOT YET ACHIEVED"**
  > *"These values represent internal development goals and have not been established through independent real-world agronomic validation."*
- Unvalidated metrics (91.4% accuracy, 87.3% stress precision, R² 0.88 growth forecasting, 0.86 IoU segmentation) are presented in muted grey styling without green achievement badges or fake progress bars.

### Measured Software Performance (Real Benchmarks)
The measured computational performance of the software stack is maintained separately:
- **Optical Spectral Index Calculation**: `<85ms` per 2km AOI (NumPy vectorized).
- **Canonical UTM Spatial Reprojection**: `<25ms` per raster tile.
- **Multimodal Temporal Search**: `<120ms` query resolution.
- **Unified Quality Mask Generation**: `<40ms` bitwise classification.
- **FastAPI End-to-End Latency**: `<180ms` (warm cache).

---

## 3. Agronomic Safety & Non-Causation Principles
- **No Direct Chemical Prescriptions**: The platform does not prescribe fertilizer quantities, chemical formulations, or pesticide dosages without physical ground-truth field inspection.
- **Non-Causation**: Variations in spectral indices reflect combined phenological maturity, solar illumination, and atmospheric effects. They are never asserted as proof of a single disease or pest organism.
