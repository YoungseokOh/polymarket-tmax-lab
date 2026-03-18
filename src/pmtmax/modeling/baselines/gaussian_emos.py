"""Gaussian EMOS-style baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class GaussianEMOSModel:
    """Simple heteroscedastic Gaussian regression."""

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self.mean_model = LinearRegression()
        self.std_model = LinearRegression()

    def fit(self, frame: pd.DataFrame) -> None:
        x = frame[self.feature_names]
        y = frame["realized_daily_max"]
        self.mean_model.fit(x, y)
        residuals = np.abs(y - self.mean_model.predict(x))
        self.std_model.fit(x, residuals)

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame[self.feature_names]
        mean = self.mean_model.predict(x)
        std = np.clip(self.std_model.predict(x), 0.5, None)
        return mean.astype(float), std.astype(float)

