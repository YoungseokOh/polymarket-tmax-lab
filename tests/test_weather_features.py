from __future__ import annotations

from datetime import date

from pmtmax.weather.features import (
    build_hourly_feature_frame,
    summarize_hourly_trajectory,
    target_day_features,
)


def test_target_day_features_handles_empty_hourly_payload() -> None:
    package = build_hourly_feature_frame({"hourly": {"time": [], "temperature_2m": []}})

    assert target_day_features(package, date(2025, 12, 11)) == {}
    assert summarize_hourly_trajectory(package, date(2025, 12, 11)).size == 0


def test_target_day_features_handles_malformed_hourly_payload_without_date_column() -> None:
    package = build_hourly_feature_frame({"hourly": {"temperature_2m": [7.0, 8.0]}})

    assert target_day_features(package, date(2025, 12, 11)) == {}
    assert summarize_hourly_trajectory(package, date(2025, 12, 11)).size == 0
