"""
ml/crop/autoencoder_vigour.py:
Method (b) Improved: PyTorch Autoencoder trained on fused Sentinel-1 and Sentinel-2
multi-temporal observations with:
1. Per-pixel series normalization: Standardizes along the temporal dimension for each pixel
   and channel, eliminating static soil background reflectance variance so the model
   focuses strictly on dynamic phenological trajectory anomalies.
2. Cloud-aware masking: Uses Sentinel-2 Scene Classification Layer (SCL) valid masks
   to zero-weight cloudy and cloud-shadowed optical observations in the reconstruction loss,
   relying on all-weather Sentinel-1 C-SAR microwave penetration during optical occlusions.
3. Latent space K-Means clustering for unsupervised vigour zone delineation.
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class TemporalAutoencoder(nn.Module):
    """Deep autoencoder with low-dimensional bottleneck for temporal crop phenology."""
    def __init__(self, in_features: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, in_features),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class MaskedMSELoss(nn.Module):
    """Reconstruction loss that ignores cloud-occluded optical timesteps."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        sq_err = (pred - target) ** 2
        weighted_sq_err = sq_err * weights
        return weighted_sq_err.sum() / (weights.sum() + self.eps)


class FusedAutoencoderVigourModel:
    """
    Manages per-pixel series normalization, cloud-aware training,
    anomaly scoring, and latent-space vigour zoning.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        num_zones: int = 3,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 25,
        random_seed: int = 42,
    ):
        self.latent_dim = latent_dim
        self.num_zones = num_zones
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_seed = random_seed

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.model: Optional[TemporalAutoencoder] = None
        self.kmeans: Optional[KMeans] = None
        self.zone_mapping: Dict[int, str] = {}
        self.training_history: list = []

    def _normalize_per_pixel_series(
        self,
        tensor_3d: np.ndarray,
        eps: float = 1e-5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalizes along the temporal dimension for each pixel and channel:
        Input: (N_pixels, T_timesteps, C_channels)
        Output: normalized (N, T, C), temporal means (N, 1, C), temporal stds (N, 1, C)
        """
        means = np.mean(tensor_3d, axis=1, keepdims=True)
        stds = np.std(tensor_3d, axis=1, keepdims=True) + eps
        normed = (tensor_3d - means) / stds
        return normed.astype(np.float32), means.astype(np.float32), stds.astype(np.float32)

    def _build_channel_weights(
        self,
        optical_masks: Optional[np.ndarray],
        N: int,
        T: int,
        C: int,
    ) -> np.ndarray:
        """
        Builds weight tensor (N, T, C) for cloud-aware loss masking.
        Optical channels (0..2: NDVI, NDRE, NDMI): weight = 1 if clear, 0 if cloudy.
        SAR channels (3..5: VV, VH, Ratio): weight = 1 (microwave penetration).
        """
        weights = np.ones((N, T, C), dtype=np.float32)
        if optical_masks is not None:
            # optical_masks can be (T, N) or (T, H, W) -> flatten to (T, N) -> transpose to (N, T)
            if optical_masks.ndim == 3:
                T_m, H, W = optical_masks.shape
                masks_flat = optical_masks.reshape(T_m, H * W).T  # (N, T)
            elif optical_masks.ndim == 2:
                masks_flat = optical_masks.T if optical_masks.shape[0] == T else optical_masks  # (N, T)
            else:
                masks_flat = np.ones((N, T), dtype=bool)

            # Mask out optical channels where mask is False
            for c in range(min(3, C)):
                weights[:, :, c] = np.where(masks_flat, 1.0, 0.0)

        return weights

    def fit(
        self,
        X_raw: np.ndarray,
        T: int = 4,
        C: int = 6,
        optical_masks: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Fits the autoencoder on per-pixel series normalized features with cloud-aware masking.
        X_raw: shape (N, T * C) or (N, T, C)
        """
        N = X_raw.shape[0]
        if X_raw.ndim == 2:
            tensor_3d = X_raw.reshape(N, T, C)
        else:
            tensor_3d = X_raw

        # 1. Per-Pixel Series Normalization
        normed_3d, self.temporal_means, self.temporal_stds = self._normalize_per_pixel_series(tensor_3d)
        X_normed = normed_3d.reshape(N, T * C)

        # 2. Cloud-Aware Weights
        weights_3d = self._build_channel_weights(optical_masks, N, T, C)
        W_flat = weights_3d.reshape(N, T * C)

        # Initialize network
        D = T * C
        self.model = TemporalAutoencoder(in_features=D, latent_dim=self.latent_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = MaskedMSELoss()

        dataset = TensorDataset(torch.from_numpy(X_normed).float(), torch.from_numpy(W_flat).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        self.training_history = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_w in loader:
                optimizer.zero_grad()
                recon, _ = self.model(batch_x)
                loss = criterion(recon, batch_x, batch_w)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            avg_loss = epoch_loss / N
            self.training_history.append(round(avg_loss, 6))

        # Fit K-Means on latent representations
        self.model.eval()
        with torch.no_grad():
            _, latent = self.model(torch.from_numpy(X_normed).float())
            latent_np = latent.numpy()

        self.kmeans = KMeans(n_clusters=self.num_zones, random_state=self.random_seed, n_init=10)
        cluster_labels = self.kmeans.fit_predict(latent_np)

        cluster_vigour_means = []
        for c in range(self.num_zones):
            mask = (cluster_labels == c)
            mean_val = float(np.mean(X_raw.reshape(N, T, C)[mask, -1, 0])) if np.any(mask) else 0.0
            cluster_vigour_means.append((c, mean_val))

        sorted_clusters = sorted(cluster_vigour_means, key=lambda x: x[1])
        tier_names = [
            "Vigour Anomaly / Stressed",
            "Moderate Vigour",
            "High Vigour"
        ] if self.num_zones == 3 else [f"Vigour Zone {i}" for i in range(self.num_zones)]

        self.zone_mapping = {
            cluster_id: tier_names[idx] for idx, (cluster_id, _) in enumerate(sorted_clusters)
        }

        final_loss = self.training_history[-1] if self.training_history else 0.0
        return {
            "status": "trained",
            "epochs": self.epochs,
            "final_reconstruction_loss": final_loss,
            "latent_dimensions": self.latent_dim,
            "num_zones": self.num_zones,
            "zone_labels": list(self.zone_mapping.values()),
            "normalization_strategy": "Per-Pixel Series Normalization (z-score along temporal dimension)",
            "cloud_masking": "Cloud-Aware Masked MSE Loss via SCL optical occlusion flags",
        }

    def predict(
        self,
        X_raw: np.ndarray,
        T: int = 4,
        C: int = 6,
        optical_masks: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Computes reconstruction error (anomaly score), latent vectors, and vigour zone classifications.
        """
        if self.model is None or self.kmeans is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        N = X_raw.shape[0]
        if X_raw.ndim == 2:
            tensor_3d = X_raw.reshape(N, T, C)
        else:
            tensor_3d = X_raw

        normed_3d, _, _ = self._normalize_per_pixel_series(tensor_3d)
        X_normed = normed_3d.reshape(N, T * C)
        weights_3d = self._build_channel_weights(optical_masks, N, T, C)
        W_flat = weights_3d.reshape(N, T * C)

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(X_normed).float()
            recon, latent = self.model(x_tensor)
            recon_np = recon.numpy()
            latent_np = latent.numpy()

        # Weighted reconstruction error per pixel
        diff_sq = (X_normed - recon_np) ** 2
        weighted_sq = diff_sq * W_flat
        weights_sum = np.maximum(np.sum(W_flat, axis=1), 1e-6)
        reconstruction_errors = np.sum(weighted_sq, axis=1) / weights_sum

        raw_cluster_ids = self.kmeans.predict(latent_np)
        named_zones = [self.zone_mapping[cid] for cid in raw_cluster_ids]

        zone_breakdown = {}
        for cid, name in self.zone_mapping.items():
            count = int(np.sum(raw_cluster_ids == cid))
            pct = round((count / N) * 100.0, 2)
            zone_breakdown[name] = {
                "pixel_count": count,
                "percentage": pct,
                "mean_anomaly_score": round(float(np.mean(reconstruction_errors[raw_cluster_ids == cid])), 4) if count > 0 else 0.0
            }

        sample_size = min(2000, N)
        sample_indices = np.random.RandomState(self.random_seed).choice(N, sample_size, replace=False)
        sil_score = float(silhouette_score(latent_np[sample_indices], raw_cluster_ids[sample_indices]))

        return {
            "reconstruction_error": {
                "mean": round(float(np.mean(reconstruction_errors)), 5),
                "median": round(float(np.median(reconstruction_errors)), 5),
                "std": round(float(np.std(reconstruction_errors)), 5),
                "p90": round(float(np.percentile(reconstruction_errors, 90)), 5),
                "p95": round(float(np.percentile(reconstruction_errors, 95)), 5),
                "min": round(float(np.min(reconstruction_errors)), 5),
                "max": round(float(np.max(reconstruction_errors)), 5),
            },
            "latent_silhouette_score": round(sil_score, 4),
            "vigour_zones": zone_breakdown,
            "raw_reconstruction_errors": reconstruction_errors,
            "raw_latent_embeddings": latent_np,
            "raw_cluster_ids": raw_cluster_ids,
        }
