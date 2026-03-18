"""Lead-time continuous statistical baseline."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class LeadTimeContinuousModel:
    """Continuous lead-time approximation using polynomial features."""

    def __init__(self, feature_names: list[str], lead_feature: str = "lead_hours") -> None:
        self.feature_names = feature_names + [lead_feature]
        self.model = Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [("num", Pipeline([("scale", StandardScaler()), ("poly", PolynomialFeatures(degree=2, include_bias=False))]), self.feature_names)],
                        remainder="drop",
                    ),
                ),
                ("lr", LinearRegression()),
            ]
        )
        self.std_model = LinearRegression()

    def fit(self, frame: pd.DataFrame) -> None:
        x = frame[self.feature_names]
        y = frame["realized_daily_max"]
        self.model.fit(x, y)
        residuals = (y - self.model.predict(x)).abs()
        self.std_model.fit(x, residuals)

    def predict(self, frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        x = frame[self.feature_names]
        mean = self.model.predict(x)
        std = self.std_model.predict(x).clip(min=0.5)
        return mean, std

