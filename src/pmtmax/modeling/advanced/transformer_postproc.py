"""Transformer-style postprocessing for hourly trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from pmtmax.modeling.advanced.torch_device import get_torch_device

_MIN_TRAIN_ROWS = 30


class TrajectoryTransformer(nn.Module):
    """Transformer encoder over NWP model daily-max pseudo-sequence."""

    def __init__(self, sequence_length: int, d_model: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(d_model, 2)
        self.sequence_length = sequence_length

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len) — already normalised
        projected = self.input_proj(x.unsqueeze(-1))  # (batch, seq_len, d_model)
        encoded = self.encoder(projected)              # (batch, seq_len, d_model)
        pooled = encoded.mean(dim=1)                   # (batch, d_model)
        out = self.head(pooled)                        # (batch, 2)
        mean = out[:, 0]
        std = torch.nn.functional.softplus(out[:, 1]) + 1e-3
        return mean, std


@dataclass
class TransformerPostprocModel:
    """Approximation of transformer-based temperature postprocessing."""

    sequence_length: int
    epochs: int = 25
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        self.constant_mean_: float | None = None
        self.constant_std_: float = 1.0
        self._seq_mean: float = 0.0
        self._seq_std: float = 1.0
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self.device = get_torch_device()
        self.network = TrajectoryTransformer(self.sequence_length).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        """
        Parameters
        ----------
        sequences : (N, seq_len) float32 — NWP model daily_max values as pseudo-sequence
        targets   : (N,) float32 — realised daily max temperatures
        """
        if len(targets) < _MIN_TRAIN_ROWS:
            self.constant_mean_ = float(targets.mean())
            self.constant_std_ = float(max(float(targets.std()) if len(targets) > 1 else 1.0, 0.5))
            return

        # Normalise sequence values and targets separately
        self._seq_mean = float(sequences.mean())
        self._seq_std = float(max(float(sequences.std()), 0.5))

        self._y_mean = float(targets.mean())
        self._y_std = float(max(float(targets.std()), 0.5))

        x_norm = ((sequences - self._seq_mean) / self._seq_std).astype(np.float32)
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

    def predict(self, sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.constant_mean_ is not None:
            n = sequences.shape[0]
            return (
                np.full(n, self.constant_mean_, dtype=float),
                np.full(n, self.constant_std_, dtype=float),
            )

        x_norm = ((sequences - self._seq_mean) / self._seq_std).astype(np.float32)
        self.network.eval()
        with torch.no_grad():
            x = torch.tensor(x_norm).to(self.device)
            mean_norm, std_norm = self.network(x)

        mean_out = mean_norm.cpu().numpy() * self._y_std + self._y_mean
        std_out = std_norm.cpu().numpy() * self._y_std
        return mean_out, std_out
