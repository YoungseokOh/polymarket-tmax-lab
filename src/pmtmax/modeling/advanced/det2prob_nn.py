"""Deterministic-to-probabilistic neural postprocessor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pmtmax.modeling.advanced.torch_device import get_torch_device


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
        self.constant_mean_: float | None = None
        self.constant_std_: float = 1.0
        self.device = get_torch_device()
        if self.feature_names:
            self.network = Det2ProbNet(len(self.feature_names)).to(self.device)

    def fit(self, frame: pd.DataFrame) -> None:
        y = frame["realized_daily_max"]
        if not self.feature_names:
            self.constant_mean_ = float(y.mean())
            self.constant_std_ = float(max((y - self.constant_mean_).abs().mean(), 0.5))
            return
        x = torch.tensor(frame[self.feature_names].to_numpy(dtype=np.float32)).to(self.device)
        yt = torch.tensor(y.to_numpy(dtype=np.float32)).to(self.device)
        dataset = TensorDataset(x, yt)
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
        if self.constant_mean_ is not None:
            size = len(frame)
            return np.full(size, self.constant_mean_, dtype=float), np.full(size, self.constant_std_, dtype=float)
        self.network.eval()
        with torch.no_grad():
            x = torch.tensor(frame[self.feature_names].to_numpy(dtype=np.float32)).to(self.device)
            mean, std = self.network(x)
        return mean.cpu().numpy(), std.cpu().numpy()

