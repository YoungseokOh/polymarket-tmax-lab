"""Flexible probabilistic neural network approximation via MDN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn


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
        self.network = MixtureDensityNet(len(self.feature_names))

    def fit(self, frame: pd.DataFrame) -> None:
        x = torch.tensor(frame[self.feature_names].to_numpy(dtype=np.float32))
        y = torch.tensor(frame["realized_daily_max"].to_numpy(dtype=np.float32))
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        for _ in range(self.epochs):
            weights, means, scales = self.network(x)
            y_exp = y.unsqueeze(1)
            log_component = -0.5 * (((y_exp - means) / scales) ** 2) - torch.log(scales)
            log_prob = torch.logsumexp(torch.log(weights) + log_component, dim=1)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            x = torch.tensor(frame[self.feature_names].to_numpy(dtype=np.float32))
            weights, means, scales = self.network(x)
        return weights.numpy(), means.numpy(), scales.numpy()

