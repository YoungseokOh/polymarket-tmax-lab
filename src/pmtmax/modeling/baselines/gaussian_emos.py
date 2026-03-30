"""Gaussian EMOS-style baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge


@dataclass(frozen=True)
class GaussianEMOSVariantConfig:
    name: str
    mean_model: Literal["ols", "ridge"]
    ridge_alpha: float
    use_recency_weights: bool
    recency_half_life_days: float
    std_model: Literal["ols", "ridge"]
    std_ridge_alpha: float


GAUSSIAN_EMOS_VARIANTS: dict[str, GaussianEMOSVariantConfig] = {
    "default": GaussianEMOSVariantConfig(
        name="default",
        mean_model="ols",
        ridge_alpha=1.0,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        std_model="ols",
        std_ridge_alpha=1.0,
    ),
    "ridge_mean": GaussianEMOSVariantConfig(
        name="ridge_mean",
        mean_model="ridge",
        ridge_alpha=10.0,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        std_model="ols",
        std_ridge_alpha=1.0,
    ),
    "ridge_recency": GaussianEMOSVariantConfig(
        name="ridge_recency",
        mean_model="ridge",
        ridge_alpha=10.0,
        use_recency_weights=True,
        recency_half_life_days=90.0,
        std_model="ridge",
        std_ridge_alpha=10.0,
    ),
    "recency_only": GaussianEMOSVariantConfig(
        name="recency_only",
        mean_model="ols",
        ridge_alpha=1.0,
        use_recency_weights=True,
        recency_half_life_days=90.0,
        std_model="ols",
        std_ridge_alpha=1.0,
    ),
}


def supported_gaussian_emos_variants() -> tuple[str, ...]:
    """Return all supported gaussian_emos ablation variants."""

    return tuple(GAUSSIAN_EMOS_VARIANTS)


def resolve_gaussian_emos_variant(variant: str | None = None) -> GaussianEMOSVariantConfig:
    """Return the config for a named variant, defaulting to 'default'."""

    key = variant or "default"
    if key not in GAUSSIAN_EMOS_VARIANTS:
        supported = ", ".join(GAUSSIAN_EMOS_VARIANTS)
        msg = f"Unsupported gaussian_emos variant: {key}. Supported: {supported}"
        raise ValueError(msg)
    return GAUSSIAN_EMOS_VARIANTS[key]


def _recency_weights(frame: pd.DataFrame, *, half_life_days: float = 90.0) -> np.ndarray:
    """Exponential decay weights based on target_date ordering."""

    if "target_date" not in frame.columns:
        return np.ones(len(frame), dtype=float)
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    if dates.isna().all():
        return np.ones(len(frame), dtype=float)
    max_date = dates.max()
    delta_days = (max_date - dates).dt.total_seconds() / 86400.0
    decay = 0.5 ** (delta_days.fillna(delta_days.median()) / half_life_days)
    weights = decay.to_numpy(dtype=float)
    total = weights.sum()
    if total > 0:
        weights = weights / total * len(weights)
    return weights


class GaussianEMOSModel:
    """Simple heteroscedastic Gaussian regression."""

    def __init__(self, feature_names: list[str], variant: str | None = None) -> None:
        self.feature_names = feature_names
        self._variant_config = resolve_gaussian_emos_variant(variant)
        self.variant = self._variant_config.name

        if self._variant_config.mean_model == "ridge":
            self.mean_model: LinearRegression | Ridge = Ridge(alpha=self._variant_config.ridge_alpha)
        else:
            self.mean_model = LinearRegression()

        if self._variant_config.std_model == "ridge":
            self.std_model: LinearRegression | Ridge = Ridge(alpha=self._variant_config.std_ridge_alpha)
        else:
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

        cfg = self._variant_config
        if cfg.use_recency_weights:
            sample_weight = _recency_weights(frame, half_life_days=cfg.recency_half_life_days)
        else:
            sample_weight = None

        if sample_weight is not None:
            self.mean_model.fit(x, y, sample_weight=sample_weight)
        else:
            self.mean_model.fit(x, y)

        residuals = np.abs(y - self.mean_model.predict(x))
        if sample_weight is not None:
            self.std_model.fit(x, residuals, sample_weight=sample_weight)
        else:
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
