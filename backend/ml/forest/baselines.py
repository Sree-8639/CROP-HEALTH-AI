"""
ml/forest/baselines.py:
Implements forest loss baseline models:
1. Spectral Differencing Baseline (Delta-NBR & Delta-NDVI with threshold optimization)
2. Torres et al. (2021) Early Concatenation U-Net for Sentinel-2 change detection
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, jaccard_score


class SpectralDifferencingBaseline:
    """
    Standard remote sensing spectral index differencing baseline.
    Computes Delta-NBR = NBR(T0) - NBR(T1) and Delta-NDVI = NDVI(T0) - NDVI(T1).
    A drop in canopy vigor/water results in large positive delta.
    """

    def __init__(self, index_name: str = "delta_nbr"):
        self.index_name = index_name
        self.best_threshold: float = 0.25

    def fit(self, val_dataset) -> float:
        """Finds optimal threshold on validation patches to maximize F1."""
        all_deltas = []
        all_targets = []
        for x0, x1, y in val_dataset:
            # x shape: (10, 256, 256). Index 5 is NBR, index 6 is NDVI
            idx = 5 if self.index_name == "delta_nbr" else 6
            delta = (x0[idx] - x1[idx]).numpy().flatten()
            target = y[0].numpy().flatten()
            all_deltas.append(delta)
            all_targets.append(target)

        deltas = np.concatenate(all_deltas)
        targets = np.concatenate(all_targets)

        best_f1 = -1.0
        best_thresh = 0.25
        # Grid search over candidate thresholds
        candidate_thresholds = np.linspace(0.10, 0.50, 41)
        for th in candidate_thresholds:
            pred = (deltas >= th).astype(np.uint8)
            p, r, f1, _ = precision_recall_fscore_support(targets, pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(th)

        self.best_threshold = best_thresh
        return self.best_threshold

    def evaluate(self, dataset) -> Dict[str, float]:
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
            "threshold": self.best_threshold,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "total_pixels_evaluated": int(len(y_true)),
            "true_loss_pixels": int(np.sum(y_true)),
            "pred_loss_pixels": int(np.sum(y_pred)),
        }


# -----------------------------------------------------------------------------
# 2. Torres et al. (2021) Early Concatenation U-Net Architecture
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
    """
    Standard U-Net change detection architecture in the style of Torres et al. (2021).
    Early concatenation: stacks Before (T0) and After (T1) input bands into
    a single 20-channel (or 14-channel S2 only) input tensor.
    """

    def __init__(self, in_channels: int = 14, out_channels: int = 1, base_features: int = 32):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features, base_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 2, base_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 4, base_features * 8))

        # Decoder with Skip Connections
        self.up1 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_features * 4, base_features * 2)

        self.up3 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_features * 2, base_features)

        # Output head
        self.outc = nn.Conv2d(base_features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # Early concatenation along channel dimension
        x = torch.cat([x0, x1], dim=1)
        
        # Encoder
        x_inc = self.inc(x)
        x_d1 = self.down1(x_inc)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)

        # Decoder
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
) -> Dict[str, Any]:
    """Trains U-Net model with Adam optimizer and combined BCE + Dice loss."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    bce_loss = nn.BCELoss()

    def dice_loss(pred, target, smooth=1e-5):
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

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
            loss = 0.5 * bce_loss(preds, y) + 0.5 * dice_loss(preds, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x0)

        epoch_loss /= len(train_dataset)
        history.append({"epoch": epoch, "train_loss": round(epoch_loss, 5)})

    return {"final_train_loss": history[-1]["train_loss"], "history": history}


def evaluate_torch_model(
    model: nn.Module,
    dataset,
    model_name: str,
    threshold: float = 0.5,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluates PyTorch model for Precision, Recall, F1, and IoU."""
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    all_preds = []
    all_targets = []
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
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "total_pixels_evaluated": int(len(y_true)),
        "true_loss_pixels": int(np.sum(y_true)),
        "pred_loss_pixels": int(np.sum(y_pred)),
    }
