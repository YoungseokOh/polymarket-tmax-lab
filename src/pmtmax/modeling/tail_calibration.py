"""Post-hoc Gaussian tail calibration wrappers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from pmtmax.markets.market_spec import MarketSpec

FallbackStdMode = Literal["primary", "max", "blend", "fallback"]


_NWP_DAILY_MAX_COLUMNS = (
    "ecmwf_ifs025_model_daily_max",
    "ecmwf_aifs025_single_model_daily_max",
    "kma_gdps_model_daily_max",
    "gfs_seamless_model_daily_max",
)


@dataclass(frozen=True)
class TailCalibrationConfig:
    """Configuration for unit-aware Gaussian mean/scale tail calibration."""

    name: str
    mean_shift_c: float = 0.0
    scale_multiplier: float = 1.0
    scale_additive_c: float = 0.0
    scale_floor_c: float = 0.1
    fallback_std_threshold_c: float | None = None
    fallback_sentinel_min_count: int | None = None
    fallback_mean_weight: float = 0.0
    fallback_std_mode: FallbackStdMode = "primary"
    fallback_std_weight: float | None = None
    sentinel_atol: float = 0.2
    nwp_daily_max_columns: tuple[str, ...] = field(default_factory=lambda: _NWP_DAILY_MAX_COLUMNS)


def _market_units(frame: pd.DataFrame) -> np.ndarray:
    units: list[str] = []
    for payload in frame.get("market_spec_json", pd.Series(["{}"] * len(frame), index=frame.index)).tolist():
        unit = "C"
        try:
            spec = MarketSpec.model_validate_json(str(payload))
            unit = spec.unit
        except Exception:
            if isinstance(payload, str):
                try:
                    raw = json.loads(payload)
                    if isinstance(raw, dict) and raw.get("unit") in {"C", "F"}:
                        unit = str(raw["unit"])
                except json.JSONDecodeError:
                    pass
        units.append(unit)
    return np.asarray(units, dtype=object)


def _to_celsius(values: np.ndarray, units: np.ndarray, *, scale: bool = False) -> np.ndarray:
    converted = values.astype(float, copy=True)
    fahrenheit = units == "F"
    if scale:
        converted[fahrenheit] = converted[fahrenheit] * (5.0 / 9.0)
    else:
        converted[fahrenheit] = (converted[fahrenheit] - 32.0) * (5.0 / 9.0)
    return converted


def _from_celsius(values_c: np.ndarray, units: np.ndarray, *, scale: bool = False) -> np.ndarray:
    converted = values_c.astype(float, copy=True)
    fahrenheit = units == "F"
    if scale:
        converted[fahrenheit] = converted[fahrenheit] * (9.0 / 5.0)
    else:
        converted[fahrenheit] = converted[fahrenheit] * (9.0 / 5.0) + 32.0
    return converted


def sentinel_daily_max_count(
    frame: pd.DataFrame,
    *,
    units: np.ndarray | None = None,
    columns: tuple[str, ...] = _NWP_DAILY_MAX_COLUMNS,
    atol: float = 0.2,
) -> np.ndarray:
    """Count source daily-max columns that look like zero-Celsius sentinels."""

    if len(frame) == 0:
        return np.asarray([], dtype=int)
    units_arr = _market_units(frame) if units is None else units
    values = np.full((len(frame), len(columns)), np.nan, dtype=float)
    for idx, column in enumerate(columns):
        if column in frame.columns:
            values[:, idx] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    sentinels = np.zeros_like(values, dtype=bool)
    fahrenheit = units_arr == "F"
    sentinels[fahrenheit] = np.isclose(values[fahrenheit], 32.0, atol=atol)
    sentinels[~fahrenheit] = np.isclose(values[~fahrenheit], 0.0, atol=atol)
    return np.nansum(sentinels, axis=1).astype(int)


@dataclass
class TailCalibratedGaussianModel:
    """Wrap a Gaussian model with unit-aware tail and fallback calibration."""

    primary_model: object
    config: TailCalibrationConfig
    fallback_model: object | None = None
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.feature_names:
            self.feature_names = list(getattr(self.primary_model, "feature_names", []))

    @staticmethod
    def _predict_gaussian(model: object, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        prediction = model.predict(frame)  # type: ignore[attr-defined]
        if not isinstance(prediction, (tuple, list)) or len(prediction) != 2:
            msg = "TailCalibratedGaussianModel only supports Gaussian mean/std models."
            raise ValueError(msg)
        mean, std = prediction
        return np.asarray(mean).reshape(-1).astype(float), np.asarray(std).reshape(-1).astype(float)

    def _fallback_mask(
        self,
        frame: pd.DataFrame,
        *,
        primary_std_c: np.ndarray,
        units: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        mask = np.zeros(len(frame), dtype=bool)
        if cfg.fallback_std_threshold_c is not None:
            mask |= primary_std_c < cfg.fallback_std_threshold_c
        if cfg.fallback_sentinel_min_count is not None:
            counts = sentinel_daily_max_count(
                frame,
                units=units,
                columns=cfg.nwp_daily_max_columns,
                atol=cfg.sentinel_atol,
            )
            mask |= counts >= cfg.fallback_sentinel_min_count
        return mask

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        frame_reset = frame.reset_index(drop=True)
        units = _market_units(frame_reset)
        primary_mean, primary_std = self._predict_gaussian(self.primary_model, frame_reset)
        primary_mean_c = _to_celsius(primary_mean, units)
        primary_std_c = np.maximum(_to_celsius(primary_std, units, scale=True), 0.1)

        cfg = self.config
        mean_c = primary_mean_c + cfg.mean_shift_c
        std_c = np.maximum(primary_std_c * cfg.scale_multiplier + cfg.scale_additive_c, cfg.scale_floor_c)

        if self.fallback_model is not None and (cfg.fallback_mean_weight > 0.0 or cfg.fallback_std_mode != "primary"):
            mask = self._fallback_mask(frame_reset, primary_std_c=primary_std_c, units=units)
            if mask.any():
                fallback_mean, fallback_std = self._predict_gaussian(self.fallback_model, frame_reset)
                fallback_mean_c = _to_celsius(fallback_mean, units)
                fallback_std_c = np.maximum(_to_celsius(fallback_std, units, scale=True), 0.1)
                mean_weight = float(np.clip(cfg.fallback_mean_weight, 0.0, 1.0))
                mean_c[mask] = (1.0 - mean_weight) * mean_c[mask] + mean_weight * fallback_mean_c[mask]

                if cfg.fallback_std_mode == "max":
                    std_c[mask] = np.maximum(std_c[mask], fallback_std_c[mask])
                elif cfg.fallback_std_mode == "blend":
                    std_weight = mean_weight if cfg.fallback_std_weight is None else float(
                        np.clip(cfg.fallback_std_weight, 0.0, 1.0)
                    )
                    std_c[mask] = (1.0 - std_weight) * std_c[mask] + std_weight * fallback_std_c[mask]
                elif cfg.fallback_std_mode == "fallback":
                    std_c[mask] = fallback_std_c[mask]
                std_c = np.maximum(std_c, cfg.scale_floor_c)

        return _from_celsius(mean_c, units), _from_celsius(std_c, units, scale=True)
