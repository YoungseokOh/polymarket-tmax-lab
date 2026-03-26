"""Time-series EMOS inspired baseline."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class TSEmosModel:
    """Approximate tsEMOS using lag/lead-aware polynomial regression."""

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self.mean_model = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("lr", LinearRegression())])
        self.std_model = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("lr", LinearRegression())])
        self.constant_mean_: float | None = None
        self.constant_std_: float = 1.0

    def fit(self, frame: pd.DataFrame) -> None:
        y = frame["realized_daily_max"]
        if not self.feature_names:
            self.constant_mean_ = float(y.mean())
            self.constant_std_ = float(max((y - self.constant_mean_).abs().mean(), 0.5))
            return
        x = frame[self.feature_names]
        self.mean_model.fit(x, y)
        residuals = (y - self.mean_model.predict(x)).abs()
        self.std_model.fit(x, residuals)

    def predict(self, frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        if self.constant_mean_ is not None:
            size = len(frame)
            return pd.Series([self.constant_mean_] * size), pd.Series([self.constant_std_] * size)
        x = frame[self.feature_names]
        mean = self.mean_model.predict(x)
        std = self.std_model.predict(x).clip(min=0.5)
        return mean, std

