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
        self._fit_columns: list[str] | None = None

    def _design_matrix(self, frame: pd.DataFrame, *, fit: bool = False) -> pd.DataFrame:
        columns = [*self.feature_names, "neighbor_mean_temp"]
        if "neighbor_spread" in frame.columns:
            columns.append("neighbor_spread")
        if fit:
            self._fit_columns = columns
        elif self._fit_columns is not None:
            columns = self._fit_columns
        return frame.reindex(columns=columns, fill_value=0.0)

    def fit(self, frame: pd.DataFrame) -> None:
        x = self._design_matrix(frame, fit=True)
        y = frame["realized_daily_max"]
        self.model.fit(x, y)

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = self._design_matrix(frame)
        mean = self.model.predict(x)
        if "neighbor_spread" in frame.columns:
            std = np.clip(frame["neighbor_spread"].to_numpy(dtype=float), 0.5, None)
        else:
            std = np.full(len(frame), 1.0, dtype=float)
        return mean, std
