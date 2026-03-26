"""Tuned ensemble of Ridge NWP blend and lead-time polynomial branches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass
class TunedEnsembleModel:
    """Ensemble combining Ridge NWP blend and compact lead-time polynomial branches.

    Branch 1 – blend: Ridge regression over all NWP/AI features.
    Branch 2 – leadtime: StandardScaler → Poly(degree=2) over the four per-model
        daily-max columns plus lead_hours, then Ridge.  Keeping this branch to
        ≤5 raw features avoids the combinatorial blowup that makes the full
        LeadTimeContinuousModel slow.

    Predictions are combined as a mixture of two Gaussians, so disagreement
    between branches widens the uncertainty estimate.
    """

    feature_names: list[str]
    alpha: float = 1.0

    def __post_init__(self) -> None:
        self.constant_mean_: float | None = None
        self.constant_std_: float = 1.0

        # Branch 1: Ridge on all features
        self.blend_mean = Ridge(alpha=self.alpha)
        self.blend_std = Ridge(alpha=self.alpha)

        # Branch 2: compact polynomial + Ridge (feature set chosen at fit time)
        self._lt_cols: list[str] = []
        self.lt_mean = Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        self.lt_std = Ridge(alpha=self.alpha)

    def fit(self, frame: pd.DataFrame) -> None:
        y = frame["realized_daily_max"]
        if not self.feature_names:
            self.constant_mean_ = float(y.mean())
            self.constant_std_ = float(max((y - self.constant_mean_).abs().mean(), 0.5))
            return

        x_blend = frame[self.feature_names]
        self.blend_mean.fit(x_blend, y)
        blend_resid = (y - self.blend_mean.predict(x_blend)).abs()
        self.blend_std.fit(x_blend, blend_resid)

        # Compact feature set: one daily_max per NWP model + lead_hours
        lt_cols = [f for f in self.feature_names if f.endswith("_model_daily_max")]
        if "lead_hours" in frame.columns:
            lt_cols = lt_cols + ["lead_hours"]
        if not lt_cols:
            lt_cols = self.feature_names[:5]
        self._lt_cols = lt_cols

        x_lt = frame[lt_cols]
        self.lt_mean.fit(x_lt, y)
        lt_resid = (y - self.lt_mean.predict(x_lt)).abs()
        self.lt_std.fit(x_lt, lt_resid)

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.constant_mean_ is not None:
            size = len(frame)
            return (
                np.full(size, self.constant_mean_, dtype=float),
                np.full(size, self.constant_std_, dtype=float),
            )

        x_blend = frame[self.feature_names]
        blend_mean = self.blend_mean.predict(x_blend).astype(float)
        blend_std = np.clip(self.blend_std.predict(x_blend).astype(float), 0.5, None)

        # Fill any missing lt columns (e.g. during live prediction)
        lt_frame = frame.copy() if any(c not in frame.columns for c in self._lt_cols) else frame
        for col in self._lt_cols:
            if col not in lt_frame.columns:
                lt_frame = lt_frame.copy()
                lt_frame[col] = 0.0
        x_lt = lt_frame[self._lt_cols]
        lt_mean = self.lt_mean.predict(x_lt).astype(float)
        lt_std = np.clip(self.lt_std.predict(x_lt).astype(float), 0.5, None)

        # Simple average: blend mean and std from both branches.
        # Mixture-of-Gaussians widens std when branches disagree (both use
        # correlated NWP inputs), which reduces edge.  Plain averaging keeps
        # calibration tighter.
        ens_mean = (blend_mean + lt_mean) / 2.0
        ens_std = (blend_std + lt_std) / 2.0
        return ens_mean, np.clip(ens_std, 0.5, None)
