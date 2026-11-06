# Table 3: Thermal Downscaling, Diurnal Analysis, & Tree-Planting Priorities (Visakhapatnam)

| Domain / Evaluation Layer | Parameter / Variable | Primary Value | Units / Scale | Notes / Reference |
| --- | --- | --- | --- | --- |
| Thermal Sensor Specification | Landsat 8/9 C2 L2 ST_B10 Calibration | DN * 0.00341802 + 149.0 - 273.15 | deg C (Celsius) | 30m native thermal resolution |
| Spatial Downscaling Split | Held-Out Spatial Block R2 | 0.0608 | R2 (Coefficient of Determination) | Spatial block partitioning (1.92 km x 1.92 km blocks, 11 train blocks, 5 held-out blocks; water excluded) |
| Spatial Downscaling Split | Held-Out Spatial RMSE | 2.194 | deg C | Root Mean Square Error on unseen spatial block |
| Spatial Downscaling Split | Held-Out Spatial MAE | 1.693 | deg C | Mean Absolute Error |
| Predictor Importance | Broadband Albedo (Liang 2001) | 5.0% | Relative Gini Importance | Dominant surface reflectance driver |
| Predictor Importance | NDVI (Canopy Vigor) | 4.4% | Relative Gini Importance | Evapotranspiration cooling factor |
| Predictor Importance | NDBI (Built-up Density) | 70.9% | Relative Gini Importance | Impervious thermal mass |
| Diurnal Dynamics | Mean Diurnal Temp Range (DTR) | 9.32 | deg C (Day - Night) | Max DTR: 11.86 deg C (Steel yards) |
| Seasonal Dynamics | Seasonal Thermal Delta | 7.96 | deg C (Summer - Winter) | Summer: 35.8 deg C, Winter: 27.8 deg C |
| UHI Hotspot Zonation | Severe Hotspots (Score >= 2.5) | 15.1% | Percent of Land Area | 870.8 ha (Gajuwaka, Port, Industrial) |
| Urban Forestry Action | Tree-Planting Target Parcels | 12 parcels | Count / Hectares | 425.7 ha (Computed average -5.3 °C empirical canopy deficit across 12 priority parcels (data-derived; species are literature-based suggestions, not computed)) |
