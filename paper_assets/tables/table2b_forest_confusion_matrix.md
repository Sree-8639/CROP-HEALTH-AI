# Table 2b: Forest Disturbance Confusion Matrices (Binary Disturbance vs Stable Canopy)

| Model Architecture | True Positive (TP) | False Positive (FP) | True Negative (TN) | False Negative (FN) | Sensitivity (Recall) | Specificity | Precision (PPV) | Accuracy | Total Evaluated Pixels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multimodal Siamese U-Net (S1+S2) | 1349 | 1505 | 257939 | 1351 | 0.4996 | 0.9942 | 0.4727 | 0.9891 | 262144 |
| Torres et al. (2021) Early-Concat U-Net | 2698 | 258654 | 790 | 2 | 0.9993 | 0.0030 | 0.0103 | 0.0133 | 262144 |
| Spectral Differencing (Delta-NBR) | 932 | 7280 | 252164 | 1768 | 0.3452 | 0.9719 | 0.1135 | 0.9655 | 262144 |
| Spectral Differencing (Delta-NDVI) | 804 | 5743 | 253701 | 1896 | 0.2978 | 0.9779 | 0.1228 | 0.9709 | 262144 |

*Hansen Label Disclosure*: Evaluated on independent test grid cells against 30m Hansen GFC reference masks.
