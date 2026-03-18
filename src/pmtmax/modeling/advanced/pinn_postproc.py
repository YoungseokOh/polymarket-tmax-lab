"""Permutation-invariant neural postprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class DeepSetPostproc(nn.Module):
    """Simple Deep Sets encoder over pseudo-ensemble members."""

    def __init__(self, member_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(member_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.phi(x)
        pooled = encoded.mean(dim=1)
        out = self.rho(pooled)
        mean = out[:, 0]
        std = torch.nn.functional.softplus(out[:, 1]) + 1e-3
        return mean, std


@dataclass
class PermutationInvariantNNModel:
    """Approximation of permutation-invariant ensemble postprocessing."""

    member_dim: int
    epochs: int = 30
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        self.network = DeepSetPostproc(self.member_dim)

    def fit(self, ensemble: np.ndarray, targets: np.ndarray) -> None:
        x = torch.tensor(ensemble.astype(np.float32))
        y = torch.tensor(targets.astype(np.float32))
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        for _ in range(self.epochs):
            mean, std = self.network(x)
            loss = torch.mean(0.5 * torch.log(std**2) + 0.5 * ((y - mean) / std) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, ensemble: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            x = torch.tensor(ensemble.astype(np.float32))
            mean, std = self.network(x)
        return mean.numpy(), std.numpy()

