"""Deterministic-to-probabilistic neural postprocessor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Det2ProbNet(nn.Module):
    """Small MLP that predicts Gaussian mean and scale."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.body(x)
        mean = out[:, 0]
        std = torch.nn.functional.softplus(out[:, 1]) + 1e-3
        return mean, std


@dataclass
class Det2ProbNNModel:
    """Landry-style public-data-compatible det2prob baseline."""

    feature_names: list[str]
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        self.network = Det2ProbNet(len(self.feature_names))

    def fit(self, frame: pd.DataFrame) -> None:
        x = torch.tensor(frame[self.feature_names].to_numpy(dtype=np.float32))
        y = torch.tensor(frame["realized_daily_max"].to_numpy(dtype=np.float32))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                mean, std = self.network(batch_x)
                loss = torch.mean(0.5 * torch.log(std**2) + 0.5 * ((batch_y - mean) / std) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self.network.eval()
        with torch.no_grad():
            x = torch.tensor(frame[self.feature_names].to_numpy(dtype=np.float32))
            mean, std = self.network(x)
        return mean.numpy(), std.numpy()

