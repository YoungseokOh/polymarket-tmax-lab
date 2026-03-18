"""Climatology baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ClimatologyModel:
    """Station-by-calendar climatology with Gaussian uncertainty."""

    default_mean: float = 0.0
    default_std: float = 3.0
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def fit(self, frame: pd.DataFrame) -> None:
        grouped = (
            frame.assign(month=frame["target_date"].dt.month, day=frame["target_date"].dt.day)
            .groupby(["station_id", "month", "day"])["realized_daily_max"]
            .agg(["mean", "std"])
            .reset_index()
        )
        grouped["std"] = grouped["std"].fillna(self.default_std).clip(lower=0.5)
        self.table = grouped
        if not frame.empty:
            self.default_mean = float(frame["realized_daily_max"].mean())
            self.default_std = float(frame["realized_daily_max"].std() or self.default_std)

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        query = frame.assign(month=frame["target_date"].dt.month, day=frame["target_date"].dt.day)
        merged = query.merge(self.table, on=["station_id", "month", "day"], how="left")
        mean = merged["mean"].fillna(self.default_mean).to_numpy(dtype=float)
        std = merged["std"].fillna(self.default_std).to_numpy(dtype=float)
        return mean, std

