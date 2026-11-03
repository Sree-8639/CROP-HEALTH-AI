# DATASET & VALIDATION DOCUMENTATION REPORT
**Crop Health AI: Multimodal Satellite Data Pipeline & Split Architecture**
*Date: 2026-11-02 | Version: 1.8.0 | Document ID: CH-AI-DATA-2026-P18*

---

## 1. Dataset Provenance and Licensing

All satellite data consumed by the Crop Health AI platform originates from the **Copernicus Programme**, operated by the **European Commission (EC)** in partnership with the **European Space Agency (ESA)**.

| Satellite Constellation | Instrument & Sensor | Data Product Level | Spatial Resolution | Revisit Interval | Distribution Mechanism | Licensing Terms |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentinel-2 (A/B)** | MultiSpectral Instrument (MSI) | Level-2A (Bottom of Atmosphere Surface Reflectance) | 10m (B02, B03, B04, B08), 20m (B05, B11, SCL) | 5 days (with 2 satellites) | Copernicus Data Space Ecosystem (CDSE) / Sentinel Hub Process API | Open Access (Creative Commons / EU Copernicus Legal Notice) |
| **Sentinel-1 (A/C)** | C-band Synthetic Aperture Radar (SAR) | Level-1 Ground Range Detected (GRD) | 10m pixel spacing (High Resolution, IW mode) | 6–12 days | CDSE OData & Process APIs | Open Access (Creative Commons / EU Copernicus Legal Notice) |

### Legal & Attribution Compliance
- Data products adhere to the *Copernicus Open Access Policy* allowing free, full, and open access for research, public, and commercial utilization.
- Attribution is embedded in all API responses and export metadata: `"Data Source: Contains modified Copernicus Sentinel data [year], processed by Crop Health AI"`.

---

## 2. Spatial Coverage & Observation Windows

### 2.1 Geographic Target Regions
The platform’s demonstration and evaluation catalogs focus on representative agricultural basins in the Krishna-Godavari Delta and surrounding agro-climatic zones in Andhra Pradesh, India:

| AOI Identifier | Field Name | Central Coordinates | Radius / Area | Primary Crop Types | Soil & Agro-climatic Zone |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `field_1` | Godavari Delta Paddy Parcel | $16.8100^\circ\text{N}, 81.5200^\circ\text{E}$ | $2.0\text{ km} \approx 1,257\text{ ha}$ | Wetland Paddy (Rice) | Alluvial clay loam, irrigated deltaic basin |
| `field_2` | Krishna Basin Mixed Parcel | $16.5062^\circ\text{N}, 80.6480^\circ\text{E}$ | $2.5\text{ km} \approx 1,963\text{ ha}$ | Sugarcane / Pulses | Deep black cotton soil (Vertisols) |
| `field_3` | Guntur Chilli & Cotton Parcel | $16.3067^\circ\text{N}, 80.4365^\circ\text{E}$ | $1.5\text{ km} \approx 707\text{ ha}$ | Chilli / Cotton / Maize | Semi-arid red sandy loam |

### 2.2 Observation Time Horizon
- **Historical Analysis Baseline**: Multi-seasonal observation window spanning 2024-01-01 through 2026-11-01.
- **Acquisition Cadence**: 5-day optical revisit and 6-to-12-day SAR revisit.
- **Temporal Query Constraints**: User interfaces enforce minimum query windows of 14 days and maximum of 365 days per analysis session.

---

## 3. Data Preprocessing & Alignment Pipeline

### 3.1 Preprocessing Workflow
1. **Atmospheric Correction (Sentinel-2)**: Level-2A products use Sen2Cor surface reflectance corrected for atmospheric aerosol scattering, water vapor, and Rayleigh effects.
2. **Speckle Mitigation & Radiometric Calibration (Sentinel-1)**:
   - Ground range detection with multi-looking.
   - Radiometric conversion to sigma-nought ($\sigma^0$) backscatter:
     $$\sigma^0_{\text{dB}} = 10 \log_{10}(\sigma^0_{\text{linear}})$$
   - Lee speckle filtering to suppress multiplicative radar noise while preserving field boundaries.
3. **Resampling & Canonical Reprojection**:
   - Both datasets are reprojected to the field's canonical WGS84 / UTM zone (e.g., UTM Zone 44N, EPSG:32644).
   - Bilinear interpolation for continuous reflectance and backscatter grids.
   - Nearest-neighbor interpolation for categorical Scene Classification Layer (SCL) and binary quality masks.
4. **Point Scale-Factor Correction**:
   - To eliminate conformal UTM projection distortion, differential ground pixel areas are computed using the point scale factor:
     $$k = k_0 \left[1 + \frac{(x - 500000)^2}{2 R^2}\right]$$
     $$\text{Area}_{\text{pixel}} = \frac{100.0\text{ m}^2}{k^2}$$

### 3.2 13-Channel Multimodal Feature Stack

```
Channel 01: B02 (Blue Reflectance)
Channel 02: B03 (Green Reflectance)
Channel 03: B04 (Red Reflectance)
Channel 04: B05 (RedEdge 1 Reflectance)
Channel 05: B08 (NIR Reflectance)
Channel 06: B11 (SWIR 1 Reflectance)
Channel 07: NDVI (Normalized Difference Vegetation Index)
Channel 08: NDMI (Normalized Difference Moisture Index)
Channel 09: NDRE (Normalized Difference Red Edge Index)
Channel 10: EVI  (Enhanced Vegetation Index)
Channel 11: VV   (C-band Vertical Co-polarization in dB)
Channel 12: VH   (C-band Cross-polarization in dB)
Channel 13: VV_VH_RATIO (Polarimetric Structural Ratio in dB)
```

---

## 4. Leakage Prevention & Dataset Partitioning

To ensure scientific validity and prevent inflated validation metrics, the platform enforces strict data partitioning rules implemented in `backend/ml/stress_detection/data_pipeline.py`:

```
+--------------------------------------------------------------------------+
| DATASET PARTITIONING ARCHITECTURE                                        |
|                                                                          |
| 1. Spatial Isolation: Unique Field IDs Partitioned (No Overlapping AOIs) |
|    - Training Set:   70% of Unique Fields                                |
|    - Validation Set: 15% of Unique Fields                                |
|    - Test Set:       15% of Unique Fields                                |
|                                                                          |
| 2. Zero Spatial Leakage: Intersect(Train_IDs, Test_IDs) = Empty Set      |
| 3. Zero Temporal Leakage: Features use strictly past acquisitions (t < T)|
| 4. Final Test Set Isolation: Test set never used for model tuning or     |
|    probability calibration                                               |
+--------------------------------------------------------------------------+
```

### 4.1 Spatial Autocorrelation Leakage Prevention
Random pixel-level or random record-level train/test splits severely contaminate agricultural ML benchmarks because adjacent pixels from the same field share soil, weather, management practices, and planting dates.
- **Rule**: Splitting is executed **strictly by unique field/AOI identifier**.
- **Enforcement**: In `spatial_temporal_split()`, unique `field_id` values are shuffled and assigned into disjoint subsets:
  $$\text{Fields}_{\text{train}} \cap \text{Fields}_{\text{val}} = \emptyset, \quad \text{Fields}_{\text{train}} \cap \text{Fields}_{\text{test}} = \emptyset, \quad \text{Fields}_{\text{val}} \cap \text{Fields}_{\text{test}} = \emptyset$$
- Automated regression test `test_spatial_temporal_split_no_leakage()` in `test_stress_detection_service.py` programmatically verifies 0% ID overlap across splits.

### 4.2 Temporal Lookahead Leakage Prevention
- In time-series feature extraction (`extract_observation_features()`), all temporal derivatives ($\Delta\text{NDVI}$, $\Delta\text{NDMI}$, rate-of-change, lagged features) are computed strictly from historical observations acquired **prior** to the current observation timestamp $t_{\text{current}}$.
- No future observations ($t > t_{\text{current}}$) are accessible during feature calculation or inference.

### 4.3 Final Test Set Isolation
The test dataset partition is held out exclusively for final model benchmarking. Hyperparameter optimization, early stopping, and Platt probability scaling calibration are restricted entirely to the training and validation partitions.

---

## 5. Quality Masking & Missing Sensor Handling

### 5.1 Cloud, Shadow, and Water Masking
The Scene Classification Layer (SCL) from Sentinel-2 L2A is decoded at every pixel:
- Masked Classes:
  - Class 3: Cloud Shadows
  - Class 6: Water Bodies
  - Class 8: Cloud Medium Probability
  - Class 9: Cloud High Probability
  - Class 10: Thin Cirrus
  - Class 11: Snow/Ice
- If cloud coverage across the AOI exceeds 60% or valid pixel percentage drops below 25%, the inference engine automatically returns:
  $$\text{assessment\_status} = \text{"POOR\_DATA\_QUALITY"}$$
  $$\text{predicted\_class} = \text{"No reliable assessment"}$$
  $$\text{stress\_probability} = \text{null}$$

### 5.2 Graceful Sensor Degradation
When a Sentinel-1 SAR acquisition is unavailable within the $\pm 3\text{-day}$ window:
1. `missing_sar_flag` is set to `1.0`.
2. Missing radar channels ($\text{VV}, \text{VH}, \text{VV/VH}$) are substituted with neutral median regional imputation.
3. The response issues an explicit advisory: `"Sentinel-1 SAR observation unavailable within ±3 days. Inference evaluated using optical spectral indices only."`
4. The system never crashes or returns silent zero values.

---

## 6. Ground-Truth Deficit & Production Model Gaps

### 6.1 Current Status of Agronomic Ground-Truth
As of November 2026:
- **Verified In-Situ Agronomic Labels Ingested**: **0 (None)**
- **Production Stress Model File (`production_stress_model.joblib`)**: **Absent by Design**
- **Synthetic Demo Model (`demo_stress_model.joblib`)**: Restricted exclusively to `DEMO_MODE=true` development and demonstration workflows.

### 6.2 Mandatory Ground-Truth Requirements for Production Training
To transition the supervised stress classification model from development to certified production status, the platform requires an ingested dataset meeting the following scientific specifications:
1. **Minimum Sample Size**: $\ge 1,000$ georeferenced agricultural plots across $\ge 3$ distinct agro-climatic zones.
2. **Temporal Multi-Season Coverage**: $\ge 2$ full growing seasons (Kharif and Rabi).
3. **Independent Reference Labels**:
   - Phenological stage (BBCH scale).
   - Confirmed stress type: Soil moisture deficit (drought), waterlogging (drainage failure), nitrogen deficiency, fungal/bacterial blight, insect pest infestation, or salinity stress.
   - Lab-verified or field instrument measurements: Chlorophyll SPAD readings, Time-Domain Reflectometry (TDR) soil moisture, or destructive dry matter biomass.
4. **Inter-Rater Reliability**: Multi-expert consensus scoring (Cohen's $\kappa \ge 0.80$) on all subjective crop condition annotations.

---

## 7. Dataset Documentation Summary

The dataset architecture of Crop Health AI guarantees reproducibility, rigorous spatial-temporal isolation, and adherence to scientific remote sensing best practices. The explicit separation between synthetic benchmarks and unvalidated production models ensures that no misleading diagnostic claims are delivered to farmers or agricultural stakeholders.
