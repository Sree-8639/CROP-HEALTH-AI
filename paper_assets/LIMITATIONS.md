# Scientific Limitations & Methodological Disclosures

**Project**: Crop Health AI / Multimodal Earth Observation Suite  
**Generated On**: 2026-11-06 02:22:00 UTC  
**Audit Standard**: Transparent Remote Sensing & AI Accountability  

> [!IMPORTANT]
> In accordance with strict honesty guidelines, no metric or result presented in this repository
> or user interface may be interpreted as field-verified ground truth. All analytical results
> are computational models evaluated against satellite and gridded reference products.

---

## 1. Proxy-Based Validation (Crop Vigour & Anomaly Detection)

### Context & Implementation
In `ml/crop/`, model evaluation operates under an **unsupervised paradigm** on **100% REAL** Copernicus CDSE Sentinel-2 L2A and Sentinel-1 GRD observations over Tadepalligudem, Andhra Pradesh:
- **Baseline**: Multi-year same-season climatology z-score anomaly computed on Sentinel-2 NDVI.
- **Improved**: Deep autoencoder trained on fused Sentinel-2 (NDVI, NDMI, NDRE) and Sentinel-1 SAR (VV, VH, ratio) time series, where reconstruction mean squared error serves as the anomaly score and latent space k-means generates vigour zones.
- **Evaluation Metrics & Verified Findings**:
  - *Monsoon Cloud Penetration*: Multimodal SAR fusion does not increase observational coverage (0.0% vs. 0.0%).
  - *Precipitation Deficit Sensitivity*: Optical NDVI correlates more strongly with regional CHIRPS rainfall deficits (r = 0.576) than the fused autoencoder anomaly (r = 0.041, Delta r = -0.536). The fused autoencoder does NOT outperform optical NDVI on linear macro-drought tracking, as multimodal reconstruction error responds to localized sub-field dielectric and structural phenology rather than monotonic regional rainfall deficits.
  - *Latent Coherence*: Unsupervised latent representation yields a silhouette score of 0.396 and a Markovian temporal transition stability of 0.914, confirming smooth phenological progression without spurious temporal oscillation.

### Key Limitations
1. **Absence of Agronomic Ground Truth**:
   Gridded precipitation deficits from CHIRPS (0.05° ≈ 5.5 km resolution, 274 daily records, 813.0 mm total) serve strictly as an environmental drought proxy. Precipitation anomalies do not capture localized farm management practices such as tube-well irrigation, canal discharge, sub-surface drainage, or micro-dosing of fertilizer.
2. **Non-Equivalence to Pathological Diagnosis**:
   While our pipeline strictly segregates weak labels and reports proxy correlations, anomaly scores reflect statistical divergence from historical trajectories rather than confirmed pathological diagnoses (e.g. bacterial leaf blight vs. nitrogen deficiency).
3. **Mixed Pixel Dilemma in Smallholder Agriculture**:
   Sentinel-2 10 m and Sentinel-1 10 m pixels often span multiple crop boundaries in smallholder farming systems (average field size in Andhra Pradesh < 0.8 hectares), resulting in mixed spectral and radar backscatter signatures.

---

## 2. Hansen Global Forest Change Reference & Methodological Realities (Forest Disturbance)

### Context & Implementation
In `ml/forest/`, the pipeline ingests **100% REAL** satellite observations:
- **Ground Truth**: Official University of Maryland / Google Earth Engine Partners **Hansen Global Forest Change (GFC v1.11)** (`treecover2000` baseline canopy closure ≥ 30% and `lossyear` ∈ {20, 21}).
- **Satellite Data**: Dual-date Copernicus Sentinel-2 L2A multispectral reflectance and Sentinel-1 C-band SAR GRD calibrated backscatter over the Eastern Ghats Maredumilli Reserve (Primary AOI) and Hasdeo Arand PEKB Coal Block (Out-of-Region Indian AOI).
- **Spatial Splitting**: Non-overlapping geographic bounding extents (Train: Northern/Western, Val: North-Eastern, Test: South-Eastern) to eliminate spatial autocorrelation leakage.

### Verified Findings & Methodological Disclosures
1. **Primary Benchmark Evaluation**:
   - On the primary test block, the Multimodal Siamese U-Net outperforms all benchmarked baselines, achieving F1 = 0.4858 and IoU = 0.3208, compared to Spectral Differencing (Delta-NDVI F1 = 0.1739, Delta-NBR F1 = 0.1708) and Torres et al. (2021) Early-Concat U-Net (F1 = 0.0204).
2. **Cloud Robustness Subset**:
   - Under optical cloud occlusion, Sentinel-1 SAR integration outperforms the optical-only Torres et al. U-Net under heavy cloud occlusion (F1 = 0.0238 vs. 0.0204). However, the Siamese U-Net does NOT outperform simple Delta-NBR differencing under cloud cover (F1 = 0.0238 vs. 0.1288), reflecting that convolutional texture representations suffer significant corruption from high-contrast cloud and cloud shadow edges.
3. **Out-of-Region Domain Shift**:
   - When evaluated on the out-of-region Hasdeo Arand PEKB coal mining block without regional adaptation, the Multimodal Siamese U-Net exhibits severe domain shift degradation, with F1 dropping from 0.4858 in the Eastern Ghats to 0.0000 against Hansen GFC 2020-2021 reference labels. By contrast, the Torres et al. architecture achieves F1 = 0.0019. While the Siamese model detected 118.6 ha of active open-cast mining across 344 polygons, temporal attribution lag in Hansen GFC (which flagged earlier clearings) resulted in zero intersection with the 2020-2021 mask.
4. **Explanation of Previous Synthetic F1 = 1.0 Baseline**:
   In preliminary prototyping, synthetic post-disturbance optical arrays were generated by directly conditioning on the binary disturbance mask: NIR = where(Y=1, 0.18, 0.46) and SWIR = where(Y=1, 0.32, 0.10). This injected an artificial drop of Delta-NBR ≈ 1.10 with minimal intra-class variance (sigma = 0.02), allowing simple thresholding to reach an artificial F1 = 1.0. On real satellite data and independent reference labels, spectral differencing achieves realistic performance (Delta-NBR F1 = 0.1708) due to mixed 10m/30m boundary pixels, topography shadows, and seasonal deciduous phenology.
5. **Spatial Footprint Mismatch (30 m Landsat vs. 10 m Sentinel)**:
   Hansen GFC is produced from Landsat 30 m multispectral imagery. Upsampling or matching Hansen binary loss masks to Sentinel-2/Sentinel-1 10 m grids introduces sub-pixel boundary uncertainty (1 Hansen pixel = 9 Sentinel pixels).
6. **Temporal Attribution Lags**:
   Hansen annual loss products assign disturbance to an entire calendar year without day- or month-level precision. Forest clearing that occurs late in the dry season may not be recorded until the subsequent cloud-free composite.

---

## 3. Urban Heat Downscaling & Absence of In-Situ Thermal Weather Stations

### Context & Implementation
In `ml/heat/`:
- Ingests **100% REAL** satellite observations over Visakhapatnam, Andhra Pradesh:
  - **Landsat 9 C2 L2 ST_B10** winter pass (LC09_L2SP_141048_20231224_02_T1, Dec 24, 2023, cloud 1.6%)
  - **Landsat 8 C2 L2 ST_B10** summer peak heat pass (LC08_L2SP_141048_20230522_02_T1, May 22, 2023, cloud 10.0%)
  - **Sentinel-2 L2A** 10m multispectral predictors (S2B_MSIL2A_20231221T050219, Dec 21, 2023, cloud 0.26%, matched within 3 days)
  - **MODIS Terra** daily 1km diurnal Day vs Night LST (MOD11A1.A2023364, Dec 30, 2023)
- Evaluates Yao et al. (2020) Random Forest thermal downscaling (30m → 10m) using strict spatial block cross-validation (1.92 km × 1.92 km geographic blocks, 30% spatial holdout, water excluded).

### Verified Findings & Plain Physical Assessment
1. **Spatial Downscaling Generalization**:
   - Yao et al. (2020) Random Forest downscaling achieves R² = 0.0608 (RMSE = 2.194 °C, MAE = 1.693 °C) on strictly held-out spatial blocks. The model does NOT achieve high spatial variance explanation (R² < 0.10) on unseen geographic sections, trailing in-sample fitting (R² = 0.1102). This reflects the fundamental physical limitation that instantaneous 2D optical indices (NDVI, NDBI, MNDWI, Albedo) cannot resolve 3D urban canopy geometry, variable construction material thermal inertia, coastal marine sea-breeze micro-advection, or point-source industrial heat emissions. The low absolute RMSE (2.194 °C) confirms that downscaled temperatures remain physically bounded.
2. **Predictor Relative Importance**:
   - NDBI dominates Random Forest thermal prediction with 70.9% relative Gini importance, indicating that built-up imperviousness is the primary driver of Land Surface Temperature in Visakhapatnam.
3. **Data-Derived Cooling Deficits vs. Species Prescriptions**:
   - Spatial multi-criteria analysis delineates 12 high-priority planting parcels covering 425.7 hectares. Computed average -5.3 °C empirical canopy deficit across 12 priority parcels (data-derived; species are literature-based suggestions, not computed).
4. **Satellite Skin Temperature vs. Ambient Air Temperature**:
   Landsat LST measures radiometric surface "skin" temperature (the top micrometers of roofs, asphalt, or tree leaves). Ambient air temperature (T_air measured at 1.5–2.0 m shelter height by meteorological stations) differs systematically by 3°C to 12°C during daytime peak insolation.
5. **Atmospheric Profile Uncertainties**:
   USGS Level-2 LST relies on atmospheric radiative transfer modelling (MODTRAN) driven by reanalysis profiles (MERRA-2). Humid tropical coastal atmospheres (e.g. Visakhapatnam) introduce water vapor attenuation errors of up to ±1.5°C.
