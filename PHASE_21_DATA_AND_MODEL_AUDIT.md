# Phase 21.1 — Complete Data and Model Audit Report

## Executive Summary
This audit inspects the complete codebase of **Crop Health AI** to evaluate all existing data assets, satellite ingestion services, model artifacts, and ground-truth validation pipelines. The objective is to establish an uncompromised foundation for genuine scientific model training and evaluation, strictly isolating synthetic demonstrator logic from real research datasets.

Audit Date: 2026-11-03  
Branch: `feat/phase-21-real-data-model-validation`  
Repository: `crop-health-monitor`

---

## 1. Audit Findings by Classification Category

### 1.1 REAL DATA AVAILABLE
- **Copernicus Data Space Ecosystem (CDSE) Integration**: Active OAuth2 client integration in `backend/services/copernicus_auth.py` and `sentinel2_service.py`. When live network access and CDSE tokens are active, genuine Sentinel-2 Level-2A (B02, B03, B04, B05, B08, B11, SCL) and Sentinel-1 GRD (VV, VH) observations can be searched and retrieved on-demand.
- **Georeferenced Field Coordinates**: Verified geographic definitions for agricultural parcels in Andhra Pradesh:
  - `field_1`: Tadepalligudem Paddy, West Godavari District (`16.8102° N, 81.5266° E`, Radius: 2.0 km)
  - `field_2`: Krishna Delta Maize, Krishna Basin (`16.5062° N, 80.6480° E`, Radius: 1.8 km)
  - `field_3`: Guntur Chilli & Cotton, Guntur District (`16.3067° N, 80.4365° E`, Radius: 2.5 km)

### 1.2 SYNTHETIC DATA AVAILABLE
- **Demo Reflectance Generator**: `generate_all_demo_spectral_bands()` in `backend/services/ndvi_service.py` simulates 10m spatial rasters with synthetic chlorophyll and water absorption signatures.
- **Demo Stress Model**: `backend/ml/models/demo_stress_model.joblib` and `demo_stress_model_metadata.json`. Formally classified as `"Synthetic agricultural demonstrator (agronomic physics curves)"` with 210 training and 45 test synthetic samples.
- **Demo Growth Simulator**: `GrowthPredictionService._generate_demo_trajectory()` produces synthetic time-series trajectories for UI demonstrations.

### 1.3 GROUND TRUTH AVAILABLE
- **In-Situ Biological Ground Truth**: **NONE STORED LOCALLY**. No physical crop cutting, SPAD meter, destructive biomass, or disease lab assays are stored in the repository.
- **Satellite Index Target (For NDVI Forecasting)**: **AVAILABLE VIA TEMPORAL SEQUENCING**. In satellite time-series forecasting, the reference target $y_{t+h}$ is the genuinely calculated NDVI from an independent subsequent satellite acquisition ($t+h$). This evaluates satellite vegetation-index forecasting, not biological disease or yield.

### 1.4 DATA MISSING
- No versioned Parquet, GeoParquet, or CSV analytical training tables currently exist in `backend/ml/` or `backend/data/`.
- Pre-downloaded raw SAFE or GeoTIFF files from CDSE are not committed to Git (consistent with best practices for geospatial data repositories).
- Multi-year continuous cloud-free historical observations for training require automated compilation and feature caching.

### 1.5 BLOCKED
- **Supervised Crop Stress Production Inference**: Intentionally blocked with status `MODEL_TRAINING_BLOCKED_PENDING_GROUND_TRUTH_DATA` in `backend/services/stress_detection_service.py`. This block must remain strictly preserved until certified in-situ agronomist ground truth is collected and reviewed.
- **Crop Yield / Biomass Prediction**: Blocked. Satellite spectral indices must never be presented as validated yield predictions without independent harvest ground truth.

### 1.6 NOT VERIFIED
- Offline synthetic fallback arrays must never be assumed to be real satellite observations simply because they appear in service outputs.
- In-situ scouting submissions in `FieldValidationService` currently exist only in memory for quality control review testing and are not certified for production ML training.

---

## 2. Answers to the 8 Strategic Audit Questions

| # | Question | Finding | Evidence / Reference |
|---|---|---|---|
| **1** | Which real satellite observations are already stored? | No raw satellite TIFF/SAFE rasters are permanently committed in the Git repository. Real observations are fetched dynamically via CDSE APIs or run with cached responses. | `backend/data/` is not tracked in Git. |
| **2** | Which observations can be retrieved again? | Any Sentinel-2 L2A tile and Sentinel-1 GRD IW acquisition covering West Godavari (`16.81° N, 81.52° E`) from 2017 to present can be retrieved using Copernicus Data Space OpenID Connect authentication. | `services/copernicus_auth.py`, `services/sentinel2_service.py`. |
| **3** | Whether field identifiers are available? | Yes. Persistent field identifiers (`field_1`, `field_2`, `field_3`, and dynamic UUIDs) are structured in `FarmerDashboardService` and `FieldValidationService`. | `backend/services/farmer_dashboard_service.py:118-150`. |
| **4** | Whether field boundaries are available? | Yes. Circular bounding radii (1.8 km to 2.5 km) and GeoJSON polygon boundaries with canonical UTM bounding boxes are calculated via `geospatial.py`. | `backend/services/geospatial.py`. |
| **5** | Whether acquisition dates are reliable? | Yes. When retrieved from CDSE, scene metadata contains ISO 8601 UTC acquisition timestamps directly from the satellite payload catalog. In demo mode, timestamps are pinned to simulated dates. | `Sentinel2Service.search_scenes()`. |
| **6** | Whether observations are genuinely from Sentinel-1/Sentinel-2? | Real requests authenticate directly with `sh.dataspace.copernicus.eu`. Simulated runs are flagged with `is_demo=True`. The pipeline explicitly checks and propagates this boolean. | `backend/services/sentinel2_service.py:47`. |
| **7** | Whether the existing forecasting baseline can be reused? | Yes. `backend/ml/growth_prediction/forecaster.py` contains 4 modular, uncompromised baselines: Persistence (Last Observation), Moving Average, Linear Trend with slope damping, and Seasonal Naive (DOY). | `backend/ml/growth_prediction/forecaster.py`. |
| **8** | Whether any independent ground-truth observations already exist? | No biological or agronomic ground-truth datasets exist in the repository. For NDVI forecasting, independent future satellite observations serve as reference targets. | `backend/services/field_validation_service.py`. |

---

## 3. Strategic Recommendations for Phase 21 Implementation
1. **Focus Exclusively on Satellite Time-Series Forecasting**: Train a gradient boosting regressor (`HistGradientBoostingRegressor`) to forecast future NDVI $y_{t+h}$ from historical Sentinel-1 and Sentinel-2 features.
2. **Explicit Scientific Disclaimers**: Clarify that NDVI forecasting validates satellite reflectance trajectories, not biological crop condition, pest damage, or yield.
3. **Build Research Data Subsystem**: Implement `backend/ml/research_data/` with dedicated acquisition, validation, dataset building, and manifest generation utilities.
4. **Mandatory Baseline Comparison**: Compare the trained ML model directly against the 4 existing baselines on identical walk-forward validation folds.
5. **No Data Leakage**: Enforce strict grouped field validation and expanding-window chronological splits.

