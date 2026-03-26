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
        self.constant_mean_: float | None = None
        self.constant_std_: float | None = None
        self._medians: dict[str, float] = {}

    def _prepare_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names:
            return pd.DataFrame(index=frame.index)

        data: dict[str, pd.Series] = {}
        for feature in self.feature_names:
            if feature in frame.columns:
                numeric = pd.to_numeric(frame[feature], errors="coerce")
            else:
                numeric = pd.Series(np.nan, index=frame.index, dtype=float)
            data[feature] = numeric.fillna(self._medians.get(feature, 0.0)).astype(float)
        return pd.DataFrame(data, index=frame.index)

    def fit(self, frame: pd.DataFrame) -> None:
        y = frame["realized_daily_max"]
        if not self.feature_names:
            self.constant_mean_ = float(y.mean())
            residuals = np.abs(y - self.constant_mean_)
            self.constant_std_ = float(np.clip(residuals.mean(), 0.5, None))
            return
        for feature in self.feature_names:
            numeric = pd.to_numeric(frame.get(feature), errors="coerce") if feature in frame.columns else pd.Series(np.nan, index=frame.index, dtype=float)
            median = numeric.median()
            self._medians[feature] = float(median) if pd.notna(median) else 0.0
        x = self._prepare_features(frame)
        self.mean_model.fit(x, y)
        residuals = np.abs(y - self.mean_model.predict(x))
        self.std_model.fit(x, residuals)

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self.feature_names:
            if self.constant_mean_ is None or self.constant_std_ is None:
                msg = "GaussianEMOSModel must be fit before predict."
                raise ValueError(msg)
            size = len(frame)
            mean = np.full(size, self.constant_mean_, dtype=float)
            std = np.full(size, self.constant_std_, dtype=float)
            return mean, std
        x = self._prepare_features(frame)
        mean = self.mean_model.predict(x)
        std = np.clip(self.std_model.predict(x), 0.5, None)
        return mean.astype(float), std.astype(float)
