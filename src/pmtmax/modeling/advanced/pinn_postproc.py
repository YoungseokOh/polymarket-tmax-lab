"""Permutation-invariant neural postprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from pmtmax.modeling.advanced.torch_device import get_torch_device

_MIN_TRAIN_ROWS = 20


class DeepSetPostproc(nn.Module):
    """Simple Deep Sets encoder over pseudo-ensemble members."""

    def __init__(self, member_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(member_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, members, member_dim)
        encoded = self.phi(x)               # (batch, members, hidden)
        pooled = encoded.mean(dim=1)        # (batch, hidden)
        out = self.rho(pooled)              # (batch, 2)
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
        self.constant_mean_: float | None = None
        self.constant_std_: float = 1.0
        # Ensemble and target scalers — set in fit
        self._ens_mean: float = 0.0
        self._ens_std: float = 1.0
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self.device = get_torch_device()
        self.network = DeepSetPostproc(self.member_dim).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, ensemble: np.ndarray, targets: np.ndarray) -> None:
        """
        Parameters
        ----------
        ensemble : (N, M, member_dim) float32 — NWP ensemble daily_max values
        targets  : (N,) float32 — realised daily max temperatures
        """
        if len(targets) < _MIN_TRAIN_ROWS:
            self.constant_mean_ = float(targets.mean())
            self.constant_std_ = float(max(float(targets.std()) if len(targets) > 1 else 1.0, 0.5))
            return

        # Normalise ensemble values (all members share the same physical scale as the target)
        ens_flat = ensemble.reshape(-1)
        self._ens_mean = float(ens_flat.mean())
        self._ens_std = float(max(float(ens_flat.std()), 0.5))

        self._y_mean = float(targets.mean())
        self._y_std = float(max(float(targets.std()), 0.5))

        x_norm = ((ensemble - self._ens_mean) / self._ens_std).astype(np.float32)
        y_norm = ((targets - self._y_mean) / self._y_std).astype(np.float32)

        x = torch.tensor(x_norm).to(self.device)
        y = torch.tensor(y_norm).to(self.device)

        self.network.train()
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

        for _ in range(self.epochs):
            mean, std = self.network(x)
            loss = torch.mean(0.5 * torch.log(std**2) + 0.5 * ((y - mean) / std) ** 2)
            if not torch.isfinite(loss):
                break
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            optimizer.step()

    def predict(self, ensemble: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.constant_mean_ is not None:
            n = ensemble.shape[0]
            return (
                np.full(n, self.constant_mean_, dtype=float),
                np.full(n, self.constant_std_, dtype=float),
            )

        x_norm = ((ensemble - self._ens_mean) / self._ens_std).astype(np.float32)
        self.network.eval()
        with torch.no_grad():
            x = torch.tensor(x_norm).to(self.device)
            mean_norm, std_norm = self.network(x)

        mean_out = mean_norm.cpu().numpy() * self._y_std + self._y_mean
        std_out = std_norm.cpu().numpy() * self._y_std
        return mean_out, std_out
