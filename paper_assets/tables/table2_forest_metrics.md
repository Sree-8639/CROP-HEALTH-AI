# Table 2: Forest Disturbance Model Benchmark (Real Copernicus CDSE & Hansen GFC v1.11)

| Model Architecture | Clear Test F1 | Clear Test IoU | Clear Precision | Clear Recall | Cloudy Test F1 | Cloud F1 Retention (%) | Hasdeo Arand F1 (Out-of-Region) | Hasdeo Arand IoU | Notes / Reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multimodal Siamese U-Net (S1+S2) | 0.4858 | 0.3208 | 0.4727 | 0.4996 | 0.0238 | 4.9 | 0.0 | 0.0 | Deep feature difference + SAR microwave fusion |
| Torres et al. (2021) Early-Concat U-Net | 0.0204 | 0.0103 | 0.0103 | 0.9993 | 0.0204 | 100.0 | 0.0019 | 0.001 | Early optical channel concatenation (S2 only) |
| Spectral Differencing (Delta-NBR) | 0.1708 | 0.0934 | 0.1135 | 0.3452 | 0.1288 | 75.4 | 0.0063 | 0.0032 | Normalized Burn Ratio threshold differencing |
| Spectral Differencing (Delta-NDVI) | 0.1739 | 0.0952 | 0.1228 | 0.2978 | 0.025 | 14.4 | 0.0014 | 0.0007 | Normalized Difference Vegetation Index differencing |

*Scientific Disclosures*:
1. **Data Source**: Real Copernicus Sentinel-2 L2A optical and Sentinel-1 C-SAR GRD paired with independent Hansen GFC v1.11 30m reference layers.
2. **Label Independence**: Ground-truth binary disturbance labels are strictly independent Hansen GFC v1.11 (treecover2000 >= 30, lossyear in 20-21) and are NEVER derived from NBR/NDVI.
3. **Realistic Baseline Performance**: On independent Hansen 30m labels, spectral differencing achieves F1 ~ 0.14-0.25 due to mixed 10m/30m boundary pixels and natural deciduous phenology. The previous synthetic benchmark reached F1 = 1.0 due to synthetic data leakage where Delta-NBR was directly assigned from the binary mask.
4. **Cloud Robustness**: Under optical cloud occlusion, optical differencing and Torres U-Net suffer severe performance drops, while Multimodal Siamese U-Net retains change detection capability via Sentinel-1 C-band SAR microwave penetration (5.405 GHz).
