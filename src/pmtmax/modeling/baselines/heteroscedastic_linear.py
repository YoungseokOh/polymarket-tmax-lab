"""Heteroscedastic linear regression with per-feature variance modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class HeteroscedasticLinearModel:
    """Ridge regression with separate mean and log-variance heads."""

    def __init__(self, feature_names: list[str], alpha: float = 1.0) -> None:
        self.feature_names = feature_names
        self.mean_model = Ridge(alpha=alpha)
        self.var_model = Ridge(alpha=alpha)

    def fit(self, frame: pd.DataFrame) -> None:
        x = frame[self.feature_names]
        y = frame["realized_daily_max"]
        self.mean_model.fit(x, y)
        residuals_sq = (y - self.mean_model.predict(x)) ** 2
        self.var_model.fit(x, np.log(np.clip(residuals_sq, 1e-4, None)))

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame[self.feature_names]
        mean = self.mean_model.predict(x)
        log_var = self.var_model.predict(x)
        std = np.sqrt(np.clip(np.exp(log_var), 0.25, None))
        return mean.astype(float), std.astype(float)
