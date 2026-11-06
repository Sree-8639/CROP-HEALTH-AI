"""
ml/forest/baselines.py:
Implements forest loss baseline models:
1. Spectral Differencing Baseline (Delta-NBR & Delta-NDVI with threshold optimization)
2. SAR Differencing Baseline (Delta-VH backscatter drop in dB)
3. Multimodal Differencing Baseline (Joint Delta-NBR and Delta-VH fusion)
4. Torres et al. (2021) Early Concatenation U-Net with class-balanced loss

SCIENTIFIC PRINCIPLE:
Evaluated against strictly independent Hansen GFC v1.11 reference labels.
Threshold optimization is performed STRICTLY on the validation split (zero leakage into held-out test splits).
Class-balanced loss (weighted BCE + Dice loss) is used to address the ~1-5% deforestation class imbalance.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, jaccard_score


class SpectralDifferencingBaseline:
    """
    Standard remote sensing spectral index differencing baseline.
    Computes Delta-NBR = NBR(T0) - NBR(T1) or Delta-NDVI = NDVI(T0) - NDVI(T1).
    A drop in canopy vigor/water results in positive delta.
    """

    def __init__(self, index_name: str = "delta_nbr"):
        self.index_name = index_name
        self.best_threshold: float = 0.20

    def fit(self, val_dataset) -> float:
        """Finds optimal threshold strictly on validation patches to maximize F1."""
        all_deltas = []
        all_targets = []
        idx = 5 if self.index_name == "delta_nbr" else 6

        for x0, x1, y in val_dataset:
            delta = (x0[idx] - x1[idx]).numpy().flatten()
            target = y[0].numpy().flatten()
            all_deltas.append(delta)
            all_targets.append(target)

        deltas = np.concatenate(all_deltas)
        targets = np.concatenate(all_targets)

        best_f1 = -1.0
        best_thresh = 0.20
        candidate_thresholds = np.linspace(0.05, 0.45, 41)
        for th in candidate_thresholds:
            pred = (deltas >= th).astype(np.uint8)
            _, _, f1, _ = precision_recall_fscore_support(targets, pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(th)

        self.best_threshold = best_thresh
        return self.best_threshold

    def evaluate(self, dataset) -> Dict[str, Any]:
        """Evaluates differencing baseline on a dataset split."""
        all_preds = []
        all_targets = []
        idx = 5 if self.index_name == "delta_nbr" else 6

        for x0, x1, y in dataset:
            delta = (x0[idx] - x1[idx]).numpy()
            pred = (delta >= self.best_threshold).astype(np.uint8).flatten()
            target = y[0].numpy().flatten()
            all_preds.append(pred)
            all_targets.append(target)

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        iou = jaccard_score(y_true, y_pred, average="binary", zero_division=0)

        return {
            "model_name": f"Spectral Differencing ({self.index_name.upper()})",
            "threshold": round(self.best_threshold, 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "iou": round(float(iou), 4),
            "total_pixels_evaluated": int(len(y_true)),
            "true_loss_pixels": int(np.sum(y_true)),
            "pred_loss_pixels": int(np.sum(y_pred)),
        }


class SARDifferencingBaseline:
    """
    Synthetic Aperture Radar (SAR) differencing baseline for forest loss detection.
    Computes Delta-VH = VH(T0) - VH(T1) in dB.
    Forest canopy volume scattering produces high cross-polarized backscatter (VH approx -13 to -16 dB).
    Deforestation removes canopy volume scattering, reducing VH backscatter (VH approx -18 to -23 dB),
    producing a positive drop: Delta-VH = VH_t0 - VH_t1 > 0.
    Operates all-weather through cloud occlusion.
    """

    def __init__(self, vh_channel_idx: int = 8):
        self.vh_channel_idx = vh_channel_idx
        self.best_threshold: float = 0.15

    def fit(self, val_dataset) -> float:
        """Finds optimal threshold strictly on validation patches to maximize F1."""
        all_deltas = []
        all_targets = []

        for x0, x1, y in val_dataset:
            delta = (x0[self.vh_channel_idx] - x1[self.vh_channel_idx]).numpy().flatten()
            target = y[0].numpy().flatten()
            all_deltas.append(delta)
            all_targets.append(target)

        deltas = np.concatenate(all_deltas)
        targets = np.concatenate(all_targets)

        best_f1 = -1.0
        best_thresh = 0.15
        candidate_thresholds = np.linspace(0.05, 0.60, 45)
        for th in candidate_thresholds:
            pred = (deltas >= th).astype(np.uint8)
            _, _, f1, _ = precision_recall_fscore_support(targets, pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(th)

        self.best_threshold = best_thresh
        return self.best_threshold

    def evaluate(self, dataset) -> Dict[str, Any]:
        all_preds = []
        all_targets = []

        for x0, x1, y in dataset:
            delta = (x0[self.vh_channel_idx] - x1[self.vh_channel_idx]).numpy()
            pred = (delta >= self.best_threshold).astype(np.uint8).flatten()
            target = y[0].numpy().flatten()
            all_preds.append(pred)
            all_targets.append(target)

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        iou = jaccard_score(y_true, y_pred, average="binary", zero_division=0)

        return {
            "model_name": "SAR Differencing (Delta-VH dB)",
            "threshold": round(self.best_threshold, 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "iou": round(float(iou), 4),
            "total_pixels_evaluated": int(len(y_true)),
            "true_loss_pixels": int(np.sum(y_true)),
            "pred_loss_pixels": int(np.sum(y_pred)),
        }


class MultimodalDifferencingBaseline:
    """
    Joint Optical-SAR Differencing Baseline combining Delta-NBR and Delta-VH.
    Detects change if either optical Delta-NBR exceeds its threshold OR SAR Delta-VH exceeds its threshold.
    """

    def __init__(self, nbr_idx: int = 5, vh_idx: int = 8):
        self.nbr_idx = nbr_idx
        self.vh_idx = vh_idx
        self.best_th_nbr: float = 0.20
        self.best_th_vh: float = 0.20

    def fit(self, val_dataset) -> Tuple[float, float]:
        all_nbr, all_vh, all_targets = [], [], []
        for x0, x1, y in val_dataset:
            d_nbr = (x0[self.nbr_idx] - x1[self.nbr_idx]).numpy().flatten()
            d_vh = (x0[self.vh_idx] - x1[self.vh_idx]).numpy().flatten()
            all_nbr.append(d_nbr)
            all_vh.append(d_vh)
            all_targets.append(y[0].numpy().flatten())

        nbrs = np.concatenate(all_nbr)
        vhs = np.concatenate(all_vh)
        targets = np.concatenate(all_targets)

        best_f1 = -1.0
        best_tn, best_tv = 0.20, 0.20
        for tn in np.linspace(0.10, 0.40, 7):
            for tv in np.linspace(0.10, 0.40, 7):
                pred = ((nbrs >= tn) | (vhs >= tv)).astype(np.uint8)
                _, _, f1, _ = precision_recall_fscore_support(targets, pred, average="binary", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_tn, best_tv = float(tn), float(tv)

        self.best_th_nbr, self.best_th_vh = best_tn, best_tv
        return best_tn, best_tv

    def evaluate(self, dataset) -> Dict[str, Any]:
        all_preds, all_targets = [], []
        for x0, x1, y in dataset:
            d_nbr = (x0[self.nbr_idx] - x1[self.nbr_idx]).numpy().flatten()
            d_vh = (x0[self.vh_idx] - x1[self.vh_idx]).numpy().flatten()
            pred = ((d_nbr >= self.best_th_nbr) | (d_vh >= self.best_th_vh)).astype(np.uint8)
            all_preds.append(pred)
            all_targets.append(y[0].numpy().flatten())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        iou = jaccard_score(y_true, y_pred, average="binary", zero_division=0)

        return {
            "model_name": "Multimodal Differencing (Delta-NBR + Delta-VH)",
            "threshold_nbr": round(self.best_th_nbr, 4),
            "threshold_vh": round(self.best_th_vh, 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "iou": round(float(iou), 4),
            "total_pixels_evaluated": int(len(y_true)),
            "true_loss_pixels": int(np.sum(y_true)),
            "pred_loss_pixels": int(np.sum(y_pred)),
        }


# -----------------------------------------------------------------------------
# Torres et al. (2021) Early Concatenation U-Net Architecture
# -----------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """[Conv2d -> BatchNorm -> ReLU] x 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorresEarlyConcatUNet(nn.Module):
    def __init__(self, in_channels: int = 14, out_channels: int = 1, base_features: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features, base_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 2, base_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 4, base_features * 8))

        self.up1 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_features * 4, base_features * 2)

        self.up3 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_features * 2, base_features)

        self.outc = nn.Conv2d(base_features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x0, x1], dim=1)
        x_inc = self.inc(x)
        x_d1 = self.down1(x_inc)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)

        u1 = self.up1(x_d3)
        u1 = self.conv_up1(torch.cat([u1, x_d2], dim=1))

        u2 = self.up2(u1)
        u2 = self.conv_up2(torch.cat([u2, x_d1], dim=1))

        u3 = self.up3(u2)
        u3 = self.conv_up3(torch.cat([u3, x_inc], dim=1))

        logits = self.outc(u3)
        return self.sigmoid(logits)


def train_unet_model(
    model: nn.Module,
    train_dataset,
    val_dataset,
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: str = "cpu",
    pos_weight: float = 15.0,
) -> Dict[str, Any]:
    """Trains U-Net model with Adam optimizer and class-balanced weighted BCE + Dice loss."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def class_balanced_loss(pred, target):
        eps = 1e-6
        pred = torch.clamp(pred, eps, 1.0 - eps)
        # Positive class weighted BCE
        w_bce = -(pos_weight * target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred)).mean()
        # Soft Dice Loss
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        dice_l = 1.0 - dice.mean()
        return 0.4 * w_bce + 0.6 * dice_l

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.to(device)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x0, x1, y in train_loader:
            x0, x1, y = x0.to(device), x1.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x0, x1)
            loss = class_balanced_loss(preds, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x0)

        epoch_loss /= len(train_dataset)
        history.append({"epoch": epoch, "train_loss": round(epoch_loss, 5)})

    return {"final_train_loss": history[-1]["train_loss"], "history": history}


def optimize_threshold_on_val(
    model: nn.Module,
    val_dataset,
    device: str = "cpu",
) -> float:
    """Finds optimal decision threshold strictly on the validation set."""
    model.eval()
    model.to(device)
    loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    all_probs, all_targets = [], []
    with torch.no_grad():
        for x0, x1, y in loader:
            x0, x1 = x0.to(device), x1.to(device)
            probs = model(x0, x1).cpu().numpy().flatten()
            targets = y.numpy().astype(np.uint8).flatten()
            all_probs.append(probs)
            all_targets.append(targets)

    y_probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)

    best_thresh = 0.35
    best_f1 = -1.0
    for th in np.linspace(0.15, 0.65, 26):
        preds = (y_probs >= th).astype(np.uint8)
        _, _, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(th)

    return best_thresh


def evaluate_torch_model(
    model: nn.Module,
    dataset,
    model_name: str,
    threshold: float = 0.35,
    device: str = "cpu",
) -> Dict[str, Any]:
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x0, x1, y in loader:
            x0, x1 = x0.to(device), x1.to(device)
            probs = model(x0, x1).cpu().numpy()
            preds = (probs >= threshold).astype(np.uint8)
            all_preds.append(preds.flatten())
            all_targets.append(y.numpy().astype(np.uint8).flatten())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    iou = jaccard_score(y_true, y_pred, average="binary", zero_division=0)

    return {
        "model_name": model_name,
        "threshold": round(threshold, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "iou": round(float(iou), 4),
        "total_pixels_evaluated": int(len(y_true)),
        "true_loss_pixels": int(np.sum(y_true)),
        "pred_loss_pixels": int(np.sum(y_pred)),
    }
