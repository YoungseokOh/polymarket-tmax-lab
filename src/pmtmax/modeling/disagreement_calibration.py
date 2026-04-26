"""Post-hoc disagreement-driven Gaussian calibration wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from pmtmax.modeling.source_gating import DEFAULT_SOURCE_NAMES
from pmtmax.modeling.tail_calibration import _from_celsius, _market_units, _to_celsius

MeanBlendMode = Literal["none", "positive", "negative", "absolute"]


@dataclass(frozen=True)
class DisagreementCalibrationConfig:
    """Configuration for source-disagreement variance and optional mean shrink."""

    name: str = "disagreement_calibration"
    disagreement_variance_weight: float = 0.0
    source_range_variance_weight: float = 0.0
    fallback_extra_variance_weight: float = 0.0
    base_scale: float = 1.0
    std_floor_c: float = 0.1
    std_cap_c: float = 30.0
    mean_blend_mode: MeanBlendMode = "none"
    mean_blend_weight_per_c: float = 0.0
    max_mean_blend_weight: float = 0.0
    include_blend_variance: bool = True
    source_names: tuple[str, ...] = field(default_factory=lambda: DEFAULT_SOURCE_NAMES)


def source_range_celsius(
    frame: pd.DataFrame,
    *,
    units: np.ndarray | None = None,
    source_names: tuple[str, ...] = DEFAULT_SOURCE_NAMES,
) -> np.ndarray:
    """Return per-row range across available source daily-max forecasts in Celsius."""

    if len(frame) == 0:
        return np.asarray([], dtype=float)
    units_arr = _market_units(frame) if units is None else units
    source_values: list[np.ndarray] = []
    for source in source_names:
        column = f"{source}_model_daily_max"
        if column not in frame.columns:
            source_values.append(np.full(len(frame), np.nan, dtype=float))
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        source_values.append(_to_celsius(values, units_arr))
    if not source_values:
        return np.zeros(len(frame), dtype=float)
    matrix = np.column_stack(source_values)
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=1)
    source_min = np.min(np.where(finite, matrix, np.inf), axis=1)
    source_max = np.max(np.where(finite, matrix, -np.inf), axis=1)
    return np.where(counts > 0, source_max - source_min, 0.0)


@dataclass
class DisagreementCalibratedGaussianModel:
    """Wrap a Gaussian model with disagreement-driven mean/std calibration."""

    primary_model: object
    fallback_model: object
    config: DisagreementCalibrationConfig
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.feature_names:
            self.feature_names = list(getattr(self.primary_model, "feature_names", []))

    @staticmethod
    def _predict_gaussian(model: object, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        prediction = model.predict(frame)  # type: ignore[attr-defined]
        if not isinstance(prediction, (tuple, list)) or len(prediction) != 2:
            msg = "DisagreementCalibratedGaussianModel only supports Gaussian mean/std models."
            raise ValueError(msg)
        mean, std = prediction
        return np.asarray(mean).reshape(-1).astype(float), np.asarray(std).reshape(-1).astype(float)

    def predict_adjustments(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        frame_reset = frame.reset_index(drop=True)
        units = _market_units(frame_reset)
        primary_mean, primary_std = self._predict_gaussian(self.primary_model, frame_reset)
        fallback_mean, fallback_std = self._predict_gaussian(self.fallback_model, frame_reset)
        primary_mean_c = _to_celsius(primary_mean, units)
        primary_std_c = np.maximum(_to_celsius(primary_std, units, scale=True), self.config.std_floor_c)
        fallback_mean_c = _to_celsius(fallback_mean, units)
        fallback_std_c = np.maximum(_to_celsius(fallback_std, units, scale=True), self.config.std_floor_c)
        diff_c = primary_mean_c - fallback_mean_c
        source_range_c = source_range_celsius(
            frame_reset,
            units=units,
            source_names=self.config.source_names,
        )
        if self.config.mean_blend_mode == "positive":
            blend_signal = np.maximum(diff_c, 0.0)
        elif self.config.mean_blend_mode == "negative":
            blend_signal = np.maximum(-diff_c, 0.0)
        elif self.config.mean_blend_mode == "absolute":
            blend_signal = np.abs(diff_c)
        else:
            blend_signal = np.zeros(len(frame_reset), dtype=float)

        blend_weight = np.clip(
            self.config.mean_blend_weight_per_c * blend_signal,
            0.0,
            self.config.max_mean_blend_weight,
        )
        mean_c = (1.0 - blend_weight) * primary_mean_c + blend_weight * fallback_mean_c

        variance_c = np.square(primary_std_c * self.config.base_scale)
        variance_c += self.config.disagreement_variance_weight * np.square(diff_c)
        variance_c += self.config.source_range_variance_weight * np.square(source_range_c)
        variance_c += self.config.fallback_extra_variance_weight * np.maximum(
            np.square(fallback_std_c) - np.square(primary_std_c),
            0.0,
        )
        if self.config.include_blend_variance:
            variance_c += blend_weight * (1.0 - blend_weight) * np.square(diff_c)
        std_c = np.clip(
            np.sqrt(np.maximum(variance_c, np.square(self.config.std_floor_c))),
            self.config.std_floor_c,
            self.config.std_cap_c,
        )
        return {
            "units": units,
            "primary_mean_c": primary_mean_c,
            "primary_std_c": primary_std_c,
            "fallback_mean_c": fallback_mean_c,
            "fallback_std_c": fallback_std_c,
            "mean_c": mean_c,
            "std_c": std_c,
            "diff_c": diff_c,
            "source_range_c": source_range_c,
            "blend_weight": blend_weight,
        }

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        adjustments = self.predict_adjustments(frame)
        units = adjustments["units"]
        return (
            _from_celsius(adjustments["mean_c"], units),
            _from_celsius(adjustments["std_c"], units, scale=True),
        )
