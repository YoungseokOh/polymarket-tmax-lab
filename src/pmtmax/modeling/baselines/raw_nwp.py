"""Raw deterministic baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd


class RawBestModelBaseline:
    """Use the single preferred model daily max directly."""

    def __init__(self, feature_name: str = "model_daily_max") -> None:
        self.feature_name = feature_name

    def fit(self, frame: pd.DataFrame) -> None:
        del frame

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mean = frame[self.feature_name].to_numpy(dtype=float)
        std = np.full_like(mean, 2.0, dtype=float)
        return mean, std


class RawMultiModelAverageBaseline:
    """Average multiple deterministic model maxima."""

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names

    def fit(self, frame: pd.DataFrame) -> None:
        del frame

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mean = frame[self.feature_names].mean(axis=1).to_numpy(dtype=float)
        std = frame[self.feature_names].std(axis=1).fillna(2.0).to_numpy(dtype=float)
        return mean, np.clip(std, 0.5, None)

