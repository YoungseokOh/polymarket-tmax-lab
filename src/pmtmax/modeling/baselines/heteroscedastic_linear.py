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
        self.constant_mean_: float | None = None
        self.constant_std_: float | None = None

    def fit(self, frame: pd.DataFrame) -> None:
        y = frame["realized_daily_max"]
        if not self.feature_names:
            self.constant_mean_ = float(y.mean())
            residuals_sq = (y - self.constant_mean_) ** 2
            self.constant_std_ = float(np.sqrt(np.clip(residuals_sq.mean(), 0.25, None)))
            return
        x = frame[self.feature_names]
        self.mean_model.fit(x, y)
        residuals_sq = (y - self.mean_model.predict(x)) ** 2
        self.var_model.fit(x, np.log(np.clip(residuals_sq, 1e-4, None)))

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.constant_mean_ is not None:
            size = len(frame)
            mean = np.full(size, self.constant_mean_, dtype=float)
            std = np.full(size, self.constant_std_ if self.constant_std_ is not None else 1.0, dtype=float)
            return mean, std
        x = frame[self.feature_names]
        mean = self.mean_model.predict(x)
        log_var = self.var_model.predict(x)
        std = np.sqrt(np.clip(np.exp(log_var), 0.25, None))
        return mean.astype(float), std.astype(float)
