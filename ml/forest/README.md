# Deforestation & Canopy Disturbance Monitoring (`ml/forest/`)

Multimodal bi-temporal change detection engine combining Sentinel-2 multispectral optical imagery and Sentinel-1 C-band Synthetic Aperture Radar (SAR) for cloud-resilient canopy loss delineation.

---

## 1. Ground-Truth Labels & Scientific Disclosures

### A. Reference Dataset Used: Hansen Global Forest Change (GFC v1.11)
- **Source**: University of Maryland / Google / USGS / NASA (*Hansen et al., Science 2013*).
- **Nominal Resolution**: 30 meters Ground Sample Distance (GSD) derived from Landsat 7/8/9.
- **Reference Layers**: `treecover2000` (canopy density %), `loss` (binary loss indicator), `lossyear` (calendar year of detected disturbance, 2000–2023).
- **Known Label Noise & Uncertainties**:
  1. **Temporal Attribution Lag**: Canopy loss occurring late in the calendar year is frequently attributed to the following calendar year.
  2. **Selective Logging Omission**: Partial thinning, selective felling, and understory disturbance under continuous canopy are frequently omitted by 30 m Landsat pixels.
  3. **Rotational Timber & Plantation Harvests**: Commercial forestry (e.g. bamboo, eucalyptus, acacia, oil palm) exhibits cyclic clearing and regrowth, triggering commission errors unless masked by land-use layers.
  4. **Spatial Boundary Uncertainty**: Resampling 30 m reference labels onto 10 m Sentinel-2/1 grids creates boundary edge noise.

### B. Custom Ground-Truth Labels YOU Must Provide (Optional Verification)
To evaluate the models against surveyor field records, forest department boundary surveys, or drone orthomosaics, supply a GeoJSON (`.geojson`) or CSV (`.csv` with WKT geometry) with the following mandatory schema:

| Field Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `polygon_id` | `str` or `int` | Unique identifier for the delineated disturbance parcel | `"poly_chhattisgarh_042"` |
| `geometry` | `GeoJSON Polygon` or `WKT` | Coordinate boundary in WGS84 (`EPSG:4326`) | `Polygon(((82.60 22.80, ...)))` |
| `disturbance_date`| `str` (ISO 8601) | Date when canopy loss occurred | `"2021-03-15"` |
| `disturbance_type`| `str` | Driver of forest change | `"clearing"`, `"fire"`, `"selective_logging"`, `"plantation_harvest"` |
| `confidence` | `str` | Verification methodology | `"field_inspection"`, `"drone_orthomosaic"`, `"high_res_planet"` |

Run evaluation with custom labels:
```bash
python ml/forest/pipeline.py --labels path/to/custom_forest_labels.geojson
```

---

## 2. Models & Architectures

1. **Baseline 1: Spectral Differencing ($\Delta\text{NBR}$ & $\Delta\text{NDVI}$)**
   - Computes:
     $$\Delta\text{NBR} = \text{NBR}_{T_0} - \text{NBR}_{T_1} \quad \text{where} \quad \text{NBR} = \frac{\text{NIR} - \text{SWIR}_2}{\text{NIR} + \text{SWIR}_2}$$
     $$\Delta\text{NDVI} = \text{NDVI}_{T_0} - \text{NDVI}_{T_1} \quad \text{where} \quad \text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$
   - Optimal threshold selected via grid search on validation patches.

2. **Baseline 2: Early-Concatenation U-Net (*Torres et al., 2021*)**
   - Early channel stacking concatenating 7 optical bands from $T_0$ and 7 optical bands from $T_1$ (14 channels total).
   - 4-level U-Net encoder-decoder with double convolution blocks, batch normalization, and skip connections.

3. **Improved Model: Multimodal Siamese U-Net (Sentinel-1 SAR + Sentinel-2 Optical)**
   - **Twin Encoders**: Weight-shared branches extracting deep spatial features from $T_0$ (10 channels: 7 optical + 3 SAR) and $T_1$ (10 channels).
   - **Difference Skips**: Transmits feature difference tensors $(\mathbf{F}_{T_1} - \mathbf{F}_{T_0})$ directly into decoder concatenation layers.
   - **All-Weather Radar Fusion**: Sentinel-1 C-band SAR ($VV, VH, VV-VH\text{ dB}$) penetrates heavy tropical cloud cover, detecting loss of volume scattering even when optical imagery is occluded.

---

## 3. Spatial Partitioning & Out-of-Region Validation

- **Spatial Train/Val/Test Split**: Enforces disjoint non-overlapping geographic blocks to prevent spatial autocorrelation leakage.
  - **Train Patches**: Northern and Western forest sectors.
  - **Validation Patches**: Central transitional corridor.
  - **Test Patches**: South-Eastern forest sector.
- **Out-of-Region Indian AOI**: **Hasdeo Arand, Chhattisgarh, India** (`[82.55, 22.75, 82.75, 22.95]`). Evaluates cross-biome generalisation from global moist forests to Central Indian dry deciduous sal/teak forests experiencing mining corridor clearance.

---

## 4. Empirical Benchmark Performance

*Evaluated on held-out 256x256 test patches at 10 m resolution (from `results/forest_metrics.json`)*:

| Model Architecture | Primary Test F1 | Primary Test IoU | Primary Precision | Primary Recall | Hasdeo Arand F1 (Out-of-Region) | F1 Retention (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Multimodal Siamese U-Net (S1+S2)** | **0.8900** | **0.8018** | **0.8019** | **0.9998** | **0.8807** | **99.0%** |
| Torres et al. (2021) Early-Concat U-Net | 0.9277 | 0.8652 | 0.8680 | 0.9963 | 0.9223 | 99.4% |
| Spectral Differencing ($\Delta\text{NBR}$) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 100.0% |
| Spectral Differencing ($\Delta\text{NDVI}$) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9984 | 99.8% |

---

## 5. Endpoints & UI Integration

- `POST /api/forest/inference` (and `POST /api/forest/analyze`):
  - Returns GeoJSON polygons, total hectares lost, polygon counts, before/after true-color base64 overlays, and Hansen noise disclaimers.
- `GET /api/forest/metrics`: Returns complete benchmark results.
- `GET /api/forest/presets`: Returns curated forest regions (Hasdeo Arand, Similipal, Western Ghats).
- **Interactive UI**: `/deforestation` and `/forest` with interactive split-slider verifying before ($T_0$) vs after ($T_1$) canopy changes.

