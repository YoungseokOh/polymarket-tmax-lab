"""Flexible probabilistic neural network approximation via MDN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from pmtmax.modeling.advanced.torch_device import get_torch_device


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
        self.device = get_torch_device()
        if self.feature_names:
            self.network = MixtureDensityNet(len(self.feature_names)).to(self.device)

    def fit(self, frame: pd.DataFrame) -> None:
        y = frame["realized_daily_max"]
        if not self.feature_names:
            self.constant_mean_ = float(y.mean())
            self.constant_std_ = float(max((y - self.constant_mean_).abs().mean(), 0.5))
            return
        x = torch.tensor(frame[self.feature_names].to_numpy(dtype=np.float32)).to(self.device)
        yt = torch.tensor(y.to_numpy(dtype=np.float32)).to(self.device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        for _ in range(self.epochs):
            weights, means, scales = self.network(x)
            y_exp = yt.unsqueeze(1)
            log_component = -0.5 * (((y_exp - means) / scales) ** 2) - torch.log(scales)
            log_prob = torch.logsumexp(torch.log(weights) + log_component, dim=1)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.constant_mean_ is not None:
            size = len(frame)
            ones = np.ones((size, 2), dtype=np.float32) * 0.5
            means = np.full((size, 2), self.constant_mean_, dtype=np.float32)
            scales = np.full((size, 2), self.constant_std_, dtype=np.float32)
            return ones, means, scales
        with torch.no_grad():
            x = torch.tensor(frame[self.feature_names].to_numpy(dtype=np.float32)).to(self.device)
            weights, means, scales = self.network(x)
        return weights.cpu().numpy(), means.cpu().numpy(), scales.cpu().numpy()

