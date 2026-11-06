"""
ml/forest/siamese_unet.py:
Improved change detection model: Multimodal Siamese U-Net for Sentinel-2 optical + Sentinel-1 SAR.
Implements twin encoder weight sharing, multi-scale feature differencing skip connections,
and spatial decoder for fine-grained 10m forest loss segmentation.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, jaccard_score


class ConvBlock(nn.Module):
    """[Conv2d -> BatchNorm -> LeakyReLU] x 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MultimodalSiameseUNet(nn.Module):
    """
    Multimodal Siamese U-Net for Sentinel-2 and Sentinel-1 change detection.
    
    Inputs:
      - x0 (Before at T0): shape (B, 10, H, W) -> [7 S2 optical + 3 S1 SAR]
      - x1 (After at T1):  shape (B, 10, H, W) -> [7 S2 optical + 3 S1 SAR]
      
    Architecture:
      - Shared Twin Encoder: Extracts multi-scale hierarchical feature maps for T0 and T1.
      - Multi-Scale Difference Skip Connections: Computes absolute feature differences |F0 - F1|
        at 4 spatial resolutions (256x256, 128x128, 64x64, 32x32).
      - Decoder: Fuses multi-scale difference features with transposed convolutions to output
        calibrated binary forest disturbance probability.
    """

    def __init__(self, in_channels: int = 10, out_channels: int = 1, base_filters: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters

        # Shared Twin Encoder
        self.enc1 = ConvBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_filters * 4, base_filters * 8)

        # Cross-temporal difference fusion at bottleneck
        # |B0 - B1| concatenated with B0 and B1 -> (8 * 3 = 24 filters -> 8 filters)
        self.bottleneck_diff = nn.Sequential(
            nn.Conv2d(base_filters * 8 * 3, base_filters * 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_filters * 8),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Decoder with Difference Skip Connections
        # Level 3: 32x32 -> 64x64
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        # Skip connection receives |F0_3 - F1_3| + upsampled -> (4 + 4 = 8 filters)
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4)

        # Level 2: 64x64 -> 128x128
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2)

        # Level 1: 128x128 -> 256x256
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_filters * 2, base_filters)

        # Segmentation Head
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # 1. Twin Encoder passes (Shared weights)
        # Stage 1 (256x256)
        f0_1 = self.enc1(x0)
        f1_1 = self.enc1(x1)
        diff_1 = torch.abs(f0_1 - f1_1)

        p0_1 = self.pool1(f0_1)
        p1_1 = self.pool1(f1_1)

        # Stage 2 (128x128)
        f0_2 = self.enc2(p0_1)
        f1_2 = self.enc2(p1_1)
        diff_2 = torch.abs(f0_2 - f1_2)

        p0_2 = self.pool2(f0_2)
        p1_2 = self.pool2(f1_2)

        # Stage 3 (64x64)
        f0_3 = self.enc3(p0_2)
        f1_3 = self.enc3(p1_2)
        diff_3 = torch.abs(f0_3 - f1_3)

        p0_3 = self.pool3(f0_3)
        p1_3 = self.pool3(f1_3)

        # Bottleneck (32x32)
        b0 = self.bottleneck(p0_3)
        b1 = self.bottleneck(p1_3)
        diff_b = torch.abs(b0 - b1)
        
        # Fused bottleneck representation
        b_fused = self.bottleneck_diff(torch.cat([b0, b1, diff_b], dim=1))

        # 2. Decoder with Multi-scale Difference Skips
        u3 = self.up3(b_fused)
        u3 = self.dec3(torch.cat([u3, diff_3], dim=1))

        u2 = self.up2(u3)
        u2 = self.dec2(torch.cat([u2, diff_2], dim=1))

        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, diff_1], dim=1))

        logits = self.final_conv(u1)
        return self.sigmoid(logits)


def train_siamese_model(
    model: MultimodalSiameseUNet,
    train_dataset,
    val_dataset,
    epochs: int = 18,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: str = "cpu",
    pos_weight: float = 15.0,
) -> Dict[str, Any]:
    """
    Trains Siamese U-Net using class-balanced weighted BCE + Dice loss
    to address severe class imbalance (~1-5% disturbance pixels).
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def class_balanced_loss(pred, target):
        eps = 1e-6
        pred = torch.clamp(pred, eps, 1.0 - eps)
        w_bce = -(pos_weight * target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred)).mean()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        dice_l = 1.0 - dice.mean()
        return 0.4 * w_bce + 0.6 * dice_l

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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

        scheduler.step()
        epoch_loss /= len(train_dataset)
        history.append({"epoch": epoch, "train_loss": round(epoch_loss, 5)})

    return {"final_train_loss": history[-1]["train_loss"], "history": history}


def optimize_siamese_threshold_on_val(
    model: MultimodalSiameseUNet,
    val_dataset,
    device: str = "cpu",
) -> float:
    """Finds optimal decision threshold strictly on validation patches."""
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

    best_thresh = 0.50
    best_f1 = -1.0
    for th in np.linspace(0.15, 0.70, 23):
        preds = (y_probs >= th).astype(np.uint8)
        _, _, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(th)

    return float(best_thresh)


def evaluate_siamese_model(
    model: MultimodalSiameseUNet,
    dataset,
    model_name: str = "Siamese U-Net (S1+S2 Fused)",
    threshold: float = 0.5,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluates Siamese U-Net for Precision, Recall, F1, and IoU.
    """
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x0, x1, y in loader:
            x0, x1 = x0.to(device), x1.to(device)
            probs = model(x0, x1).cpu().numpy()
            preds = (probs >= threshold).astype(np.uint8)
            all_preds.append(preds.flatten())
            all_targets.append(y.numpy().astype(np.uint8).flatten())
            all_probs.append(probs)

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
        "hansen_caveat": "Evaluated against 30m Landsat-derived Hansen GFC labels. Discretization artifacts and sub-pixel noise inherently bound maximum empirical IoU on 10m satellite grids.",
    }

