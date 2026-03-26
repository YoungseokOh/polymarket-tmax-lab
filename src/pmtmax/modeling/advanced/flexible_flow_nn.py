"""Flexible probabilistic neural network approximation via MDN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from pmtmax.modeling.advanced.torch_device import get_torch_device

# MDN with 2 components has more parameters than det2prob; needs more data to generalise.
_MIN_TRAIN_ROWS = 50


class MixtureDensityNet(nn.Module):
    """Small MDN with two Gaussian components."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, components: int = 2) -> None:
        super().__init__()
        self.components = components
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.logits = nn.Linear(hidden_dim, components)
        self.means = nn.Linear(hidden_dim, components)
        self.scales = nn.Linear(hidden_dim, components)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.body(x)
        weights = torch.softmax(self.logits(hidden), dim=-1)
        means = self.means(hidden)
        scales = torch.nn.functional.softplus(self.scales(hidden)) + 1e-3
        return weights, means, scales


@dataclass
class FlexibleProbNNModel:
    """Public-data-friendly approximation to flexible probabilistic NN postprocessing."""

    feature_names: list[str]
    epochs: int = 40
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        self.constant_mean_: float | None = None
        self.constant_std_: float = 1.0
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self.device = get_torch_device()
        if self.feature_names:
            self.network = MixtureDensityNet(len(self.feature_names)).to(self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise_fit(self, x_raw: np.ndarray, y_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._feat_mean = x_raw.mean(axis=0)
        feat_std = x_raw.std(axis=0)
        self._feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)

        self._y_mean = float(y_raw.mean())
        self._y_std = float(max(float(y_raw.std()), 0.5))

        x_norm = (x_raw - self._feat_mean) / self._feat_std
        y_norm = (y_raw - self._y_mean) / self._y_std
        return x_norm.astype(np.float32), y_norm.astype(np.float32)

    def _normalise_x(self, x_raw: np.ndarray) -> np.ndarray:
        assert self._feat_mean is not None, "call fit() before predict()"
        return ((x_raw - self._feat_mean) / self._feat_std).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, frame: pd.DataFrame) -> None:
        y_raw = frame["realized_daily_max"].to_numpy(dtype=np.float32)

        if not self.feature_names or len(frame) < _MIN_TRAIN_ROWS:
            self.constant_mean_ = float(y_raw.mean())
            self.constant_std_ = float(max(float(y_raw.std()) if len(y_raw) > 1 else 1.0, 0.5))
            return

        x_raw = frame[self.feature_names].to_numpy(dtype=np.float32)
        x_norm, y_norm = self._normalise_fit(x_raw, y_raw)

        x = torch.tensor(x_norm).to(self.device)
        yt = torch.tensor(y_norm).to(self.device)

        self.network.train()
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

        for _ in range(self.epochs):
            weights, means, scales = self.network(x)
            y_exp = yt.unsqueeze(1)
            # Log-likelihood of each Gaussian component (normalised space)
            log_component = -0.5 * (((y_exp - means) / scales) ** 2) - torch.log(scales)
            log_prob = torch.logsumexp(torch.log(weights + 1e-8) + log_component, dim=1)
            loss = -log_prob.mean()
            if not torch.isfinite(loss):
                break
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            optimizer.step()

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.constant_mean_ is not None:
            size = len(frame)
            weights = np.full((size, 2), 0.5, dtype=np.float32)
            means = np.full((size, 2), self.constant_mean_, dtype=np.float32)
            scales = np.full((size, 2), self.constant_std_, dtype=np.float32)
            return weights, means, scales

        x_norm = self._normalise_x(frame[self.feature_names].to_numpy(dtype=np.float32))
        self.network.eval()
        with torch.no_grad():
            x = torch.tensor(x_norm).to(self.device)
            weights, means_norm, scales_norm = self.network(x)

        # Denormalise component means and scales back to Celsius
        means_out = means_norm.cpu().numpy() * self._y_std + self._y_mean
        scales_out = scales_norm.cpu().numpy() * self._y_std
        return weights.cpu().numpy(), means_out, scales_out
