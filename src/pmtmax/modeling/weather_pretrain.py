"""Weather-pretrain feature injection for market models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from pmtmax.markets.market_spec import MarketSpec

PRETRAIN_FEATURE_SOURCE_MAP: dict[str, tuple[str, ...]] = {
    "forecast_temperature_2m_min": ("gfs_seamless_model_daily_min",),
    "forecast_temperature_2m_mean": ("gfs_seamless_model_daily_mean",),
    "forecast_temperature_2m_max": ("gfs_seamless_model_daily_max", "model_daily_max"),
    "forecast_dew_point_2m_mean": ("gfs_seamless_dew_point_mean",),
    "forecast_relative_humidity_2m_mean": ("gfs_seamless_humidity_mean",),
    "forecast_wind_speed_10m_mean": ("gfs_seamless_wind_speed_mean",),
    "forecast_cloud_cover_mean": ("gfs_seamless_cloud_cover_mean",),
}

WEATHER_PRETRAIN_FEATURE_COLUMNS = (
    "weather_pretrain_mean",
    "weather_pretrain_std",
    "weather_pretrain_delta_vs_gfs",
)
WeatherPretrainFeatureMode = Literal["full", "delta_only"]


def weather_pretrain_feature_columns(feature_mode: WeatherPretrainFeatureMode) -> tuple[str, ...]:
    """Return the active feature columns for one weather-pretrain mode."""

    if feature_mode == "full":
        return WEATHER_PRETRAIN_FEATURE_COLUMNS
    if feature_mode == "delta_only":
        return ("weather_pretrain_delta_vs_gfs",)
    msg = f"Unsupported weather pretrain feature mode: {feature_mode}"
    raise ValueError(msg)


def _is_temperature_feature(feature: str) -> bool:
    return feature.startswith("forecast_temperature_") or feature.startswith("forecast_dew_point_")


def _market_units(frame: pd.DataFrame) -> np.ndarray:
    if "market_spec_json" not in frame.columns:
        return np.full(len(frame), "C", dtype=object)
    units: list[str] = []
    for payload in frame["market_spec_json"].tolist():
        try:
            units.append(MarketSpec.model_validate_json(str(payload)).unit)
        except Exception:  # noqa: BLE001
            units.append("C")
    return np.asarray(units, dtype=object)


def _to_celsius(values: pd.Series, units: np.ndarray) -> pd.Series:
    converted = values.astype(float).copy()
    fahrenheit = units == "F"
    converted.loc[fahrenheit] = (converted.loc[fahrenheit] - 32.0) * (5.0 / 9.0)
    return converted


def _mean_to_market_units(values_c: np.ndarray, units: np.ndarray) -> np.ndarray:
    values = values_c.astype(float, copy=True)
    fahrenheit = units == "F"
    values[fahrenheit] = values[fahrenheit] * (9.0 / 5.0) + 32.0
    return values


def _scale_to_market_units(values_c: np.ndarray, units: np.ndarray) -> np.ndarray:
    values = values_c.astype(float, copy=True)
    values[units == "F"] = values[units == "F"] * (9.0 / 5.0)
    return values


def _first_numeric_series(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _pretrain_input_frame(frame: pd.DataFrame, weather_pretrain_model: Any) -> pd.DataFrame:
    units = _market_units(frame)
    feature_names = list(getattr(weather_pretrain_model, "feature_names", []))
    data: dict[str, pd.Series] = {}
    for feature in feature_names:
        candidates = PRETRAIN_FEATURE_SOURCE_MAP.get(feature, (feature,))
        values = _first_numeric_series(frame, candidates)
        if _is_temperature_feature(feature):
            values = _to_celsius(values, units)
        data[feature] = values
    return pd.DataFrame(data, index=frame.index)


def append_weather_pretrain_features(
    frame: pd.DataFrame,
    weather_pretrain_model: Any,
    *,
    feature_mode: WeatherPretrainFeatureMode = "full",
) -> pd.DataFrame:
    """Append weather-pretrain predictions in each row's market temperature unit."""

    augmented = frame.copy()
    if augmented.empty:
        for column in weather_pretrain_feature_columns(feature_mode):
            augmented[column] = pd.Series(dtype=float)
        return augmented

    units = _market_units(augmented)
    prediction = weather_pretrain_model.predict(_pretrain_input_frame(augmented, weather_pretrain_model))
    if not isinstance(prediction, (tuple, list)) or len(prediction) < 2:
        msg = "weather pretrain model must return mean/std predictions"
        raise ValueError(msg)
    mean_c = np.asarray(prediction[0]).reshape(-1).astype(float)
    std_c = np.asarray(prediction[1]).reshape(-1).astype(float)
    if len(mean_c) != len(augmented) or len(std_c) != len(augmented):
        msg = "weather pretrain prediction length does not match frame"
        raise ValueError(msg)

    mean = _mean_to_market_units(mean_c, units)
    std = np.maximum(_scale_to_market_units(std_c, units), 0.1)
    gfs_max = _first_numeric_series(augmented, ("gfs_seamless_model_daily_max", "model_daily_max")).to_numpy(dtype=float)

    if feature_mode == "full":
        augmented["weather_pretrain_mean"] = mean
        augmented["weather_pretrain_std"] = std
    augmented["weather_pretrain_delta_vs_gfs"] = mean - gfs_max
    return augmented


@dataclass
class WeatherPretrainAugmentedModel:
    """Model wrapper that applies weather-pretrain features before prediction."""

    base_model: Any
    weather_pretrain_model: Any
    pretrained_weather_model_path: str
    feature_mode: WeatherPretrainFeatureMode = "full"

    def __getattr__(self, name: str) -> Any:
        base_model = self.__dict__.get("base_model")
        if base_model is None:
            raise AttributeError(name)
        return getattr(base_model, name)

    def predict(self, frame: pd.DataFrame) -> object:
        feature_mode = self.__dict__.get("feature_mode", "full")
        return self.base_model.predict(
            append_weather_pretrain_features(
                frame,
                self.weather_pretrain_model,
                feature_mode=feature_mode,
            )
        )
