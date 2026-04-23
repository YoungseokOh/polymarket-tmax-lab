from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd

from pmtmax.cli.main import _trust_check_report
from pmtmax.weather.training_data import (
    WeatherTrainingProgressEvent,
    collect_weather_training_data,
    default_weather_training_paths,
)


def _payload(*, target: str, temps: list[float]) -> dict:
    return {
        "hourly": {
            "time": [f"{target}T00:00", f"{target}T12:00"],
            "temperature_2m": temps,
            "dew_point_2m": [value - 2.0 for value in temps],
            "relative_humidity_2m": [60.0, 70.0],
            "wind_speed_10m": [3.0, 4.0],
            "cloud_cover": [10.0, 20.0],
        }
    }


class _FakeOpenMeteo:
    def historical_forecast(self, **kwargs: object) -> dict:
        return _payload(target=str(kwargs["start_date"]), temps=[5.0, 7.0])

    def historical_weather(self, **kwargs: object) -> dict:
        return _payload(target=str(kwargs["start_date"]), temps=[6.0, 9.0])


class _RateLimitedOpenMeteo:
    def historical_forecast(self, **_: object) -> dict:
        request = httpx.Request("GET", "https://example.test")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    def historical_weather(self, **_: object) -> dict:
        raise AssertionError("historical_weather should not run after forecast 429")


def _trust_config(tmp_path: Path, *, workspace_name: str, dataset_profile: str) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            workspace_name=workspace_name,
            dataset_profile=dataset_profile,
            data_dir=tmp_path / "data",
            artifacts_dir=tmp_path / "artifacts",
            duckdb_path=tmp_path / "data" / "duckdb" / "warehouse.duckdb",
            parquet_dir=tmp_path / "data" / "parquet",
            public_model_dir=tmp_path / "artifacts" / "public_models",
        )
    )


def test_collect_weather_training_writes_real_weather_rows(tmp_path: Path) -> None:
    result = collect_weather_training_data(
        openmeteo=_FakeOpenMeteo(),  # type: ignore[arg-type]
        raw_root=tmp_path / "raw",
        parquet_root=tmp_path / "parquet",
        station_cities=["Seoul"],
        date_from=date(2026, 1, 1),
        date_to=date(2026, 1, 2),
        model="gfs_seamless",
        missing_only=True,
        rate_limit_profile="test",
    )

    paths = default_weather_training_paths(tmp_path / "parquet")
    gold = pd.read_parquet(paths.gold_path)
    bronze = pd.read_parquet(paths.bronze_path)

    assert result.collected_rows == 2
    assert set(gold["station_id"]) == {"RKSI"}
    assert set(gold["model_name"]) == {"gfs_seamless"}
    assert gold["realized_daily_max"].tolist() == [9.0, 9.0]
    assert "market_id" not in gold.columns
    assert set(bronze["status"]) == {"available"}


def test_collect_weather_training_records_429_without_fake_rows(tmp_path: Path) -> None:
    result = collect_weather_training_data(
        openmeteo=_RateLimitedOpenMeteo(),  # type: ignore[arg-type]
        raw_root=tmp_path / "raw",
        parquet_root=tmp_path / "parquet",
        station_cities=["Seoul"],
        date_from=date(2026, 1, 1),
        date_to=date(2026, 1, 1),
        model="gfs_seamless",
        missing_only=True,
        rate_limit_profile="test",
    )

    paths = default_weather_training_paths(tmp_path / "parquet")
    bronze = pd.read_parquet(paths.bronze_path)
    gold = pd.read_parquet(paths.gold_path)

    assert result.collected_rows == 0
    assert result.failed_rows == 1
    assert bronze.loc[0, "status"] == "retryable_error"
    assert gold.empty


def test_collect_weather_training_emits_progress_events(tmp_path: Path) -> None:
    events: list[WeatherTrainingProgressEvent] = []

    result = collect_weather_training_data(
        openmeteo=_FakeOpenMeteo(),  # type: ignore[arg-type]
        raw_root=tmp_path / "raw",
        parquet_root=tmp_path / "parquet",
        station_cities=["Seoul"],
        date_from=date(2026, 1, 1),
        date_to=date(2026, 1, 1),
        model="gfs_seamless",
        missing_only=True,
        rate_limit_profile="test",
        progress_callback=events.append,
    )

    assert result.collected_rows == 1
    assert [event.phase for event in events] == ["start", "available"]
    assert events[0].station_id == "RKSI"
    assert events[1].realized_daily_max == 9.0
    assert events[1].elapsed_seconds >= 0.0


def test_trust_check_allows_weather_training_profile_only_for_weather_workflow(tmp_path: Path) -> None:
    weather_report = _trust_check_report(
        config=_trust_config(tmp_path, workspace_name="weather_train", dataset_profile="weather_real"),
        markets_path=None,
        workflow="weather_training",
    )
    market_report = _trust_check_report(
        config=_trust_config(tmp_path, workspace_name="weather_train", dataset_profile="weather_real"),
        markets_path=None,
    )

    assert weather_report["ok"] is True
    assert market_report["ok"] is False
    assert {issue["check"] for issue in market_report["issues"]} == {"dataset_profile"}
