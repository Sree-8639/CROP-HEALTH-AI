# Table 4: Modality Ablation Summary (S2-only vs S1-only vs Fused S1+S2)

| Configuration | Channels | Train Loss | Reconstruction Error (Mean) | Reconstruction Error (Median) | Latent Silhouette Score | Cloud Occlusion Loss Inflation | Vigour Anomaly Pixels (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sentinel-2 Optical (NDVI, NDMI, NDRE) | 10 | 0.0126 | 0.0126 | 0.0062 | 0.3650 | 3.42x | 55.10% |
| Sentinel-1 SAR (VV, VH, VV-VH dB) | 9 | 0.1279 | 0.1279 | 0.0902 | 0.3090 | 1.02x | 55.10% |
| Fused S1 (SAR) + S2 (Optical) | 19 | 0.1398 | 0.1398 | 0.0957 | 0.3960 | 1.14x | 55.10% |
