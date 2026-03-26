"""Transformer-style postprocessing for hourly trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from pmtmax.modeling.advanced.torch_device import get_torch_device


class TrajectoryTransformer(nn.Module):
    """Transformer encoder over hourly trajectories."""

    def __init__(self, sequence_length: int, d_model: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(d_model, 2)
        self.sequence_length = sequence_length

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(self.input_proj(x.unsqueeze(-1)))
        pooled = encoded.mean(dim=1)
        out = self.head(pooled)
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
        self.device = get_torch_device()
        self.network = TrajectoryTransformer(self.sequence_length).to(self.device)

    def fit(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        x = torch.tensor(sequences.astype(np.float32)).to(self.device)
        y = torch.tensor(targets.astype(np.float32)).to(self.device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        for _ in range(self.epochs):
            mean, std = self.network(x)
            loss = torch.mean(0.5 * torch.log(std**2) + 0.5 * ((y - mean) / std) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            x = torch.tensor(sequences.astype(np.float32)).to(self.device)
            mean, std = self.network(x)
        return mean.cpu().numpy(), std.cpu().numpy()

