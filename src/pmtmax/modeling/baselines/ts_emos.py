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

    def fit(self, frame: pd.DataFrame) -> None:
        x = frame[self.feature_names]
        y = frame["realized_daily_max"]
        self.mean_model.fit(x, y)
        residuals = (y - self.mean_model.predict(x)).abs()
        self.std_model.fit(x, residuals)

    def predict(self, frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        x = frame[self.feature_names]
        mean = self.mean_model.predict(x)
        std = self.std_model.predict(x).clip(min=0.5)
        return mean, std

