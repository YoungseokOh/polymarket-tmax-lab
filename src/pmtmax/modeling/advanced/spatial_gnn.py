"""Spatial extension using simple graph aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class SpatialGNNModel:
    """GNN-inspired linear model over aggregated neighbor features."""

    feature_names: list[str]
    alpha: float = 1.0

    def __post_init__(self) -> None:
        self.model = Ridge(alpha=self.alpha)

    def fit(self, frame: pd.DataFrame) -> None:
        x = frame[self.feature_names + ["neighbor_mean_temp", "neighbor_spread"]]
        y = frame["realized_daily_max"]
        self.model.fit(x, y)

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame[self.feature_names + ["neighbor_mean_temp", "neighbor_spread"]]
        mean = self.model.predict(x)
        std = np.clip(frame["neighbor_spread"].to_numpy(dtype=float), 0.5, None)
        return mean, std

