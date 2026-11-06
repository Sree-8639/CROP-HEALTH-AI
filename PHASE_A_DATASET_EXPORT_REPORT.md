# Phase A: Multimodal Crop Condition Dataset Export Report

## 1. Stack and File Locations Audit
- **Frontend Stack**: Next.js (App Router, Tailwind CSS, Leaflet) deployed on Vercel (`https://crop-health-ai-q8iv.vercel.app/`).
  - Source directory: `frontend/src/`
- **Backend Stack**: Python 3.13 / FastAPI ASGI container hosted on Render (`https://crophealth-backend.onrender.com/`).
  - Source directory: `backend/`
- **Multimodal Fusion Service**:
  - Core implementation: `backend/services/fusion_service.py`
  - Raster alignment & CRS grid: `backend/services/raster_alignment.py`
- **Phase A Dataset Exporter & Pipeline**:
  - Exporter module: `backend/ml/research_data/condition_dataset.py`
  - Unit tests: `backend/tests/test_phase_a_dataset_export.py`
- **Export Data Artifacts**:
  - Output directory: `backend/ml/research_data/data/phase_a/`
  - Parquet dataset: `crop_condition_dataset.parquet` (24 KB, 96 rows, 23 columns)
  - CSV dataset: `crop_condition_dataset.csv` (46 KB)
  - Manifest: `manifest.json`
  - Image Chips:
    - 224x224 Sentinel-2 True Color RGB Chips: `images/rgb/*.png` (96 files)
    - 224x224 Sentinel-2 Continuous NDVI Colormap Chips: `images/ndvi/*.png` (96 files)

---

## 2. Dataset Schema & Features

Each row represents an independent **field-date observation** with multimodal features:

| Field Group | Columns | Description |
| :--- | :--- | :--- |
| **Identifiers & Geometry** | `sample_id`, `field_id`, `field_name`, `date`, `latitude`, `longitude`, `crop_type` | Unique observation key, field name, observation date, geospatial center coordinates, and agricultural crop type. |
| **Optical Spectral Indices** | `ndvi`, `ndmi`, `ndre`, `evi` | Normalized Difference indices from Sentinel-2 (B02, B04, B05, B08, B11). |
| **Synthetic Aperture Radar (SAR)** | `vv`, `vh`, `vv_vh_ratio` | Sentinel-1 GRD co-pol and cross-pol radar backscatter in decibels (dB), plus structural polarization difference. |
| **Quality & Provenance** | `cloud_quality`, `is_demo`, `data_provenance` | Cloud/valid pixel percentage (%), synthetic/live catalog indicator, and source pipeline citation. |
| **Classification Target** | `label`, `label_source`, `label_reason`, `is_weak_label` | Physiological condition class, origin source (`weak_weather_based`), agronomic explanation, and boolean weak label flag. |
| **Visual Paired Chips** | `rgb_chip_path`, `ndvi_chip_path` | Relative paths to paired 224×224 RGB and NDVI PNG image patches. |

---

## 3. Strict Non-Circular Weak Labeling Methodology

In accordance with scientific honesty rules, **NDVI was strictly excluded from the label assignment logic**. Labels were derived solely from:
1. **Canopy Moisture & Dielectric Drop**: Attenuated shortwave infrared water content (`NDMI < 0.18`) combined with radar dielectric permittivity loss (`VV < -13.5 dB`) and 14-day rainfall deficit (`> 38%`) $\rightarrow$ **`water-stressed`**.
2. **Chlorophyll & Nitrogen Attenuation**: Depressed red-edge slope (`NDRE < 0.22`) while shortwave infrared moisture remains adequate (`NDMI >= 0.22`) $\rightarrow$ **`nutrient-stressed`**.
3. **Canopy Architecture Degradation**: Radar structural depolarization surge (`VV - VH >= 9.5 dB`) indicating cross-polarization canopy volume scattering disruption or foliar collapse $\rightarrow$ **`suspected disease-pest`**.
4. **Vigorous Balanced Canopy**: Adequate moisture (`NDMI >= 0.22`), healthy red-edge chlorophyll (`NDRE >= 0.25`), and normal radar volumetric scattering $\rightarrow$ **`healthy`**.

---

## 4. Dataset Summary & Class Distribution

- **Total Samples**: 96 field-date observations
- **Independent Agricultural Fields**: 8 fields across West Godavari, Krishna, and Guntur districts (Andhra Pradesh, India)
- **Temporal Range**: 2023-08-10 to 2024-06-25 (Kharif, Rabi, and Summer seasons)
- **Class Breakdown**:
  - `healthy`: 45 samples (46.9%)
  - `nutrient-stressed`: 19 samples (19.8%)
  - `water-stressed`: 18 samples (18.8%)
  - `suspected disease-pest`: 14 samples (14.6%)
- **Weak Label Percentage**: 100.0% (`is_weak_label = True`, `label_source = 'weak_weather_based'`)
- **Demo / Synthetic Data Flag**: `is_demo = True` (clearly marked in metadata and table)
- **Image Chips**: 192 total chips (96 RGB + 96 NDVI, exactly 224×224 pixels)

---

## 5. Verification & Automated Test Status

Ran `pytest backend/tests/test_phase_a_dataset_export.py -v`:
- `test_phase_a_files_exist`: **PASSED**
- `test_phase_a_dataframe_schema`: **PASSED**
- `test_phase_a_classes_represented`: **PASSED**
- `test_phase_a_image_chips_dimension_and_format`: **PASSED**
- `test_phase_a_non_circular_labeling_guarantee`: **PASSED**
- `test_phase_a_honesty_rules`: **PASSED**

Result: **6 passed in 12.40s**.

