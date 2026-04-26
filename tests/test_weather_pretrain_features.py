from __future__ import annotations

import numpy as np
import pandas as pd

from pmtmax.examples import example_market_specs
from pmtmax.modeling.weather_pretrain import (
    WeatherPretrainAugmentedModel,
    append_weather_pretrain_features,
)


class _FakeWeatherPretrain:
    feature_names = ["forecast_temperature_2m_max", "forecast_dew_point_2m_mean", "lead_hours"]

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (
            pd.to_numeric(frame["forecast_temperature_2m_max"], errors="coerce").to_numpy(dtype=float) + 1.0,
            np.full(len(frame), 2.0, dtype=float),
        )


class _FakeBaseModel:
    feature_names = ["weather_pretrain_mean", "weather_pretrain_std", "weather_pretrain_delta_vs_gfs"]

    def __init__(self) -> None:
        self.seen_columns: list[str] = []

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self.seen_columns = list(frame.columns)
        return (
            pd.to_numeric(frame["weather_pretrain_mean"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(frame["weather_pretrain_std"], errors="coerce").to_numpy(dtype=float),
        )


def test_append_weather_pretrain_features_maps_gfs_to_pretrain_schema_and_units() -> None:
    seoul = example_market_specs(["Seoul"])[0]
    nyc = example_market_specs(["NYC"])[0]
    frame = pd.DataFrame(
        {
            "market_spec_json": [seoul.model_dump_json(), nyc.model_dump_json()],
            "gfs_seamless_model_daily_max": [10.0, 68.0],
            "gfs_seamless_dew_point_mean": [3.0, 50.0],
            "lead_hours": [12.0, 12.0],
        }
    )

    augmented = append_weather_pretrain_features(frame, _FakeWeatherPretrain())

    assert augmented["weather_pretrain_mean"].round(6).tolist() == [11.0, 69.8]
    assert augmented["weather_pretrain_std"].round(6).tolist() == [2.0, 3.6]
    assert augmented["weather_pretrain_delta_vs_gfs"].round(6).tolist() == [1.0, 1.8]


def test_weather_pretrain_augmented_model_adds_features_before_base_predict() -> None:
    base = _FakeBaseModel()
    model = WeatherPretrainAugmentedModel(
        base_model=base,
        weather_pretrain_model=_FakeWeatherPretrain(),
        pretrained_weather_model_path="weather.pkl",
    )
    spec = example_market_specs(["Seoul"])[0]
    frame = pd.DataFrame(
        {
            "market_spec_json": [spec.model_dump_json()],
            "gfs_seamless_model_daily_max": [10.0],
            "gfs_seamless_dew_point_mean": [3.0],
            "lead_hours": [12.0],
        }
    )

    mean, std = model.predict(frame)

    assert mean.tolist() == [11.0]
    assert std.tolist() == [2.0]
    assert "weather_pretrain_mean" in base.seen_columns


def test_delta_only_mode_excludes_biased_pretrain_level_features() -> None:
    spec = example_market_specs(["Seoul"])[0]
    frame = pd.DataFrame(
        {
            "market_spec_json": [spec.model_dump_json()],
            "gfs_seamless_model_daily_max": [10.0],
            "gfs_seamless_dew_point_mean": [3.0],
            "lead_hours": [12.0],
        }
    )

    augmented = append_weather_pretrain_features(
        frame,
        _FakeWeatherPretrain(),
        feature_mode="delta_only",
    )

    assert "weather_pretrain_mean" not in augmented.columns
    assert "weather_pretrain_std" not in augmented.columns
    assert augmented["weather_pretrain_delta_vs_gfs"].tolist() == [1.0]
