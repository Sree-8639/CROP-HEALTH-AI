# Ground-Truth Data Collection & Target Definition Protocol

## 1. Scientific Integrity Principle: Avoiding Circular Validation

A major error in satellite agricultural machine learning is **circular threshold validation**, where:
1. Vegetation indices (e.g. NDVI, NDMI) are computed from optical bands.
2. A machine learning model is trained on these spectral bands to predict an artificial "Crop Health" label that was created by applying a hardcoded rule to the very same NDVI values.
3. The model achieves high accuracy by simply learning the inverted heuristic formula, creating the illusion of predictive success without independent agronomic validity.

To prevent circular validation:
- **For Satellite Time-Series Forecasting**: The reference target $y_{t+h}$ is the **future observed NDVI** from an independent, subsequent satellite acquisition ($t+h$). The input features contain only information available at timestamp $t$.
- **For Agronomic Crop Condition & Stress Detection**: Models must **never** be trained on synthetic heuristics. Supervised models require independently collected, certified in-situ agronomist labels.

---

## 2. Satellite-Index Reference Target Formulation

For the primary time-series forecasting model:
- **Task**: 1-step ahead vegetation index regression.
- **Input Features ($t$)**:
  - Optical: $\text{NDVI}_t, \text{NDMI}_t, \text{NDRE}_t, \text{EVI}_t, \text{B02}_t, \text{B03}_t, \text{B04}_t, \text{B05}_t, \text{B08}_t, \text{B11}_t$
  - SAR: $\text{VV}_t, \text{VH}_t, (\text{VV}/\text{VH})_t$
  - Temporal Lags: $\text{NDVI}_{t-1}, \Delta \text{NDVI}, \text{rate\_per\_day}$
  - Quality: $\text{cloud\_pct}, \text{valid\_pixel\_pct}, \text{missing\_sar\_indicator}$
- **Target ($t+h$)**:
  - Future observed $\text{NDVI}_{t+h}$, where $5 \le h \le 35$ days.
  - Calculated independently from a later Copernicus Sentinel-2 Level-2A scene.

---

## 3. Agronomic In-Situ Ground-Truth Collection Workflow

For downstream biological classification and stress diagnosis, Crop Health AI implements a strict field protocol:

### 3.1 Certified Observations
- **Observer Qualifications**: Certified Crop Advisor (CCA), Agricultural University Faculty, or M.Sc/Ph.D. Agronomist.
- **Informed Landowner Consent**: Mandatory written or digital consent registered prior to data capture.
- **Double-Blind Verification**: Every field scouting record requires independent approval from a second credentialed agronomist.

### 3.2 Physical Measurement Standards
| Measurement | Instrument / Standard | Sampling Protocol |
|---|---|---|
| **Canopy Chlorophyll** | Minolta SPAD-502 or optical meter | Minimum 3 replicate leaves per plant, 10 plants per AOI quadrant. |
| **Leaf Area Index (LAI)** | Licor LAI-2200C or Ceptometer | Above/below canopy paired readings during diffuse solar illumination. |
| **Above-Ground Biomass** | Destructive quadrat sampling ($1\text{m}^2$) | Oven-dried at 65°C for 48 hours to constant dry mass. |
| **Harvest Yield** | Calibrated weigh-wagon or grain moisture meter | Corrected to 14% standardized paddy grain moisture content. |

---

## 4. Evaluation of External Datasets

| Dataset Source | Modality | Scientific Applicability | Protocol & Licensing Restrictions |
|---|---|---|---|
| **Krishi Vigyan Kendra (KVK)** | Localized field trials in Andhra Pradesh | Crop phenology dates and disease survey reports. | Non-commercial research collaboration with institutional attribution. |
| **Radiant MLHub** | Crop type & boundary benchmark masks | Ground-truth crop type classification. | CC-BY-4.0 open geospatial data license. |
| **CropHarvest** | Global agricultural classification corpus | Benchmarking baseline multi-crop models. | Permissive research open license. |
| **ERA5 Atmospheric Reanalysis** | Temperature, precipitation, radiation | Explanatory environmental covariates. | Copernicus Open Access terms; environmental context, not direct stress labels. |
| **SMAP Soil Moisture** | L-band microwave passive radiometry | Root-zone dielectric permittivity. | NASA open data; explanatory covariate, not crop stress ground truth. |

---

## 5. Non-Fabrication Rule
If independent field measurements are unavailable for an observation date, the label remains **strictly missing (`None` / `NaN`)**. The system rejects synthetic label substitution.

