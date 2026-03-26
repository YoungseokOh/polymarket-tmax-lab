"""AI + NWP blending."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class AifsNwpBlendModel:
    """Weighted and learned blends across physics-based and AI forecasts."""

    nwp_features: list[str]
    ai_features: list[str]
    use_learned_blend: bool = True

    def __post_init__(self) -> None:
        self.model = LinearRegression()
        self.weights_: dict[str, float] = {}
        self.constant_mean_: float | None = None

    def fit(self, frame: pd.DataFrame) -> None:
        columns = self.nwp_features + self.ai_features
        if not columns:
            self.constant_mean_ = float(frame["realized_daily_max"].mean())
            return
        if self.use_learned_blend:
            self.model.fit(frame[columns], frame["realized_daily_max"])
            coefficients = self.model.coef_
            self.weights_ = {name: float(weight) for name, weight in zip(columns, coefficients, strict=True)}
        else:
            weight = 1.0 / len(columns)
            self.weights_ = dict.fromkeys(columns, weight)

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.constant_mean_ is not None:
            size = len(frame)
            return np.full(size, self.constant_mean_, dtype=float), np.full(size, 1.0, dtype=float)
        columns = self.nwp_features + self.ai_features
        if self.use_learned_blend and self.weights_:
            mean = self.model.predict(frame[columns])
        else:
            mean = frame[columns].mean(axis=1).to_numpy(dtype=float)
        nwp_only = frame[self.nwp_features].mean(axis=1).to_numpy(dtype=float) if self.nwp_features else mean
        ai_only = frame[self.ai_features].mean(axis=1).to_numpy(dtype=float) if self.ai_features else mean
        spread = np.abs(nwp_only - ai_only)
        return mean, np.clip(spread, 0.5, None)

