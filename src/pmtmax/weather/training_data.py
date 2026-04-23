"""Real weather-only training data collection helpers."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from pmtmax.markets.station_registry import StationDefinition, lookup_station, supported_cities
from pmtmax.utils import dump_json, stable_hash
from pmtmax.weather.openmeteo_client import OpenMeteoClient

WEATHER_TRAINING_HOURLY = [
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
]

BRONZE_COLUMNS = [
    "station_id",
    "city",
    "target_date",
    "timezone",
    "model_name",
    "forecast_source_kind",
    "status",
    "error",
    "forecast_raw_path",
    "observation_raw_path",
    "forecast_raw_hash",
    "observation_raw_hash",
    "collected_at_utc",
]

SILVER_COLUMNS = [
    "station_id",
    "city",
    "target_date",
    "timezone",
    "model_name",
    "source",
    "time_local",
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
]

GOLD_COLUMNS = [
    "station_id",
    "city",
    "target_date",
    "timezone",
    "model_name",
    "forecast_source_kind",
    "issue_time_utc",
    "lead_hours",
    "forecast_temperature_2m_min",
    "forecast_temperature_2m_mean",
    "forecast_temperature_2m_max",
    "forecast_dew_point_2m_mean",
    "forecast_relative_humidity_2m_mean",
    "forecast_wind_speed_10m_mean",
    "forecast_cloud_cover_mean",
    "forecast_hourly_json",
    "realized_daily_max",
    "observation_hourly_json",
    "source_metadata_json",
    "forecast_raw_hash",
    "observation_raw_hash",
]


@dataclass(frozen=True)
class WeatherTrainingPaths:
    bronze_path: Path
    silver_path: Path
    gold_path: Path


@dataclass(frozen=True)
class WeatherTrainingCollectionResult:
    paths: WeatherTrainingPaths
    requested_rows: int
    collected_rows: int
    skipped_existing_rows: int
    failed_rows: int
    status_counts: dict[str, int]
    failures: list[dict[str, str]]
    workers_requested: int
    workers_effective: int


@dataclass(frozen=True)
class WeatherTrainingProgressEvent:
    phase: Literal["start", "skip", "available", "retryable_error", "error"]
    station_id: str
    city: str
    target_date: str
    model_name: str
    item_index: int
    item_total: int
    elapsed_seconds: float
    status_code: int | None = None
    realized_daily_max: float | None = None
    message: str = ""


def default_weather_training_paths(parquet_root: Path) -> WeatherTrainingPaths:
    """Return the bronze/silver/gold weather training parquet locations."""

    return WeatherTrainingPaths(
        bronze_path=parquet_root / "bronze" / "weather_training_requests.parquet",
        silver_path=parquet_root / "silver" / "weather_training_hourly.parquet",
        gold_path=parquet_root / "gold" / "weather_training_set.parquet",
    )


def default_weather_training_date_range(today: date | None = None) -> tuple[date, date]:
    """Return the conservative free-API rollout window: recent two years."""

    today = today or datetime.now(tz=UTC).date()
    end = today - timedelta(days=1)
    start = max(date(2021, 1, 1), end - timedelta(days=730))
    return start, end


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        msg = f"date_to {end} is before date_from {start}"
        raise ValueError(msg)
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _station_universe(cities: list[str] | None = None) -> list[StationDefinition]:
    labels = cities or supported_cities()
    stations: list[StationDefinition] = []
    seen: set[tuple[str, str]] = set()
    for city in labels:
        station = lookup_station(city)
        if station is None:
            msg = f"Unsupported station/city in catalog: {city}"
            raise ValueError(msg)
        key = (station.city, station.station_id)
        if key in seen:
            continue
        seen.add(key)
        stations.append(station)
    return stations


def _raw_hash(payload: dict[str, Any]) -> str:
    return stable_hash(json.dumps(payload, sort_keys=True, default=str))


def _write_raw_payload(raw_root: Path, *, station: StationDefinition, target_date: date, model: str, kind: str, payload: dict[str, Any]) -> Path:
    path = (
        raw_root
        / "bronze"
        / "weather_training"
        / station.station_id
        / f"{target_date.isoformat()}__{model}__{kind}.json"
    )
    dump_json(path, payload)
    return path


def _hourly_frame(payload: dict[str, Any], *, target_date: date) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if not isinstance(hourly, dict) or "time" not in hourly:
        return pd.DataFrame(columns=["time", *WEATHER_TRAINING_HOURLY])
    frame = pd.DataFrame({"time": hourly.get("time", [])})
    for column in WEATHER_TRAINING_HOURLY:
        values = hourly.get(column)
        if isinstance(values, list) and len(values) == len(frame):
            frame[column] = pd.to_numeric(pd.Series(values), errors="coerce")
    parsed = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.loc[parsed.dt.date == target_date].copy()
    frame["time"] = frame["time"].astype(str)
    return frame.reset_index(drop=True)


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").mean()
    return float(value) if pd.notna(value) else None


def _min(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").min()
    return float(value) if pd.notna(value) else None


def _max(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").max()
    return float(value) if pd.notna(value) else None


def _hourly_json(frame: pd.DataFrame) -> str:
    columns = ["time", *[column for column in WEATHER_TRAINING_HOURLY if column in frame.columns]]
    return json.dumps(frame[columns].to_dict(orient="records"), sort_keys=True, default=str)


def _silver_rows(
    frame: pd.DataFrame,
    *,
    station: StationDefinition,
    target_date: date,
    model: str,
    source: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "station_id": station.station_id,
                "city": station.city,
                "target_date": target_date.isoformat(),
                "timezone": station.timezone,
                "model_name": model,
                "source": source,
                "time_local": str(row.get("time", "")),
                "temperature_2m": row.get("temperature_2m"),
                "dew_point_2m": row.get("dew_point_2m"),
                "relative_humidity_2m": row.get("relative_humidity_2m"),
                "wind_speed_10m": row.get("wind_speed_10m"),
                "cloud_cover": row.get("cloud_cover"),
            }
        )
    return rows


def _status_code_from_exception(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return int(status_code) if isinstance(status_code, int) else None


def _existing_keys(gold_path: Path) -> set[tuple[str, str, str]]:
    if not gold_path.exists():
        return set()
    try:
        frame = pd.read_parquet(gold_path, columns=["station_id", "target_date", "model_name"])
    except Exception:  # noqa: BLE001
        return set()
    return {
        (str(row.station_id), str(row.target_date)[:10], str(row.model_name))
        for row in frame.itertuples(index=False)
    }


def _merge_and_write(path: Path, new_frame: pd.DataFrame, *, key_columns: list[str] | None = None, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        frame = pd.concat([existing, new_frame], ignore_index=True)
    else:
        frame = new_frame
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[columns]
    if key_columns and not frame.empty:
        frame = frame.drop_duplicates(subset=key_columns, keep="last").reset_index(drop=True)
    frame.to_parquet(path, index=False)


def collect_weather_training_data(
    *,
    openmeteo: OpenMeteoClient,
    raw_root: Path,
    parquet_root: Path,
    station_cities: list[str] | None,
    date_from: date,
    date_to: date,
    model: str,
    missing_only: bool,
    workers: int = 1,
    rate_limit_profile: str = "free",
    progress_callback: Callable[[WeatherTrainingProgressEvent], None] | None = None,
) -> WeatherTrainingCollectionResult:
    """Collect weather-only rows without Polymarket market ids or prices."""

    stations = _station_universe(station_cities)
    dates = _date_range(date_from, date_to)
    paths = default_weather_training_paths(parquet_root)
    existing = _existing_keys(paths.gold_path) if missing_only else set()
    sleep_seconds = 0.0 if rate_limit_profile in {"none", "test"} else 0.2
    workers_effective = 1
    item_total = len(stations) * len(dates)

    bronze_rows: list[dict[str, object]] = []
    silver_rows: list[dict[str, object]] = []
    gold_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    skipped = 0
    item_index = 0

    for station in stations:
        for target_date in dates:
            item_index += 1
            key = (station.station_id, target_date.isoformat(), model)
            if key in existing:
                skipped += 1
                if progress_callback is not None:
                    progress_callback(
                        WeatherTrainingProgressEvent(
                            phase="skip",
                            station_id=station.station_id,
                            city=station.city,
                            target_date=target_date.isoformat(),
                            model_name=model,
                            item_index=item_index,
                            item_total=item_total,
                            elapsed_seconds=0.0,
                            message="existing row",
                        )
                    )
                continue
            collected_at = datetime.now(tz=UTC).isoformat()
            started_at = time.perf_counter()
            if progress_callback is not None:
                progress_callback(
                    WeatherTrainingProgressEvent(
                        phase="start",
                        station_id=station.station_id,
                        city=station.city,
                        target_date=target_date.isoformat(),
                        model_name=model,
                        item_index=item_index,
                        item_total=item_total,
                        elapsed_seconds=0.0,
                    )
                )
            try:
                forecast_payload = openmeteo.historical_forecast(
                    latitude=station.lat,
                    longitude=station.lon,
                    model=model,
                    hourly=WEATHER_TRAINING_HOURLY,
                    start_date=target_date.isoformat(),
                    end_date=target_date.isoformat(),
                    timezone=station.timezone,
                )
                observation_payload = openmeteo.historical_weather(
                    latitude=station.lat,
                    longitude=station.lon,
                    hourly=WEATHER_TRAINING_HOURLY,
                    start_date=target_date.isoformat(),
                    end_date=target_date.isoformat(),
                    timezone=station.timezone,
                )
                forecast_raw_path = _write_raw_payload(
                    raw_root,
                    station=station,
                    target_date=target_date,
                    model=model,
                    kind="historical_forecast",
                    payload=forecast_payload,
                )
                observation_raw_path = _write_raw_payload(
                    raw_root,
                    station=station,
                    target_date=target_date,
                    model=model,
                    kind="historical_weather",
                    payload=observation_payload,
                )
                forecast_hash = _raw_hash(forecast_payload)
                observation_hash = _raw_hash(observation_payload)
                forecast_frame = _hourly_frame(forecast_payload, target_date=target_date)
                observation_frame = _hourly_frame(observation_payload, target_date=target_date)
                realized_daily_max = _max(observation_frame, "temperature_2m")
                if forecast_frame.empty or realized_daily_max is None:
                    msg = "missing forecast hours" if forecast_frame.empty else "missing realized daily max"
                    raise ValueError(msg)
                issue_time = datetime.combine(target_date, datetime.min.time(), tzinfo=UTC)
                source_metadata = {
                    "provider": "open-meteo",
                    "forecast_endpoint": "historical_forecast",
                    "observation_endpoint": "historical_weather",
                    "rate_limit_profile": rate_limit_profile,
                    "workers_requested": workers,
                    "workers_effective": workers_effective,
                }
                bronze_rows.append(
                    {
                        "station_id": station.station_id,
                        "city": station.city,
                        "target_date": target_date.isoformat(),
                        "timezone": station.timezone,
                        "model_name": model,
                        "forecast_source_kind": "historical_forecast",
                        "status": "available",
                        "error": "",
                        "forecast_raw_path": str(forecast_raw_path),
                        "observation_raw_path": str(observation_raw_path),
                        "forecast_raw_hash": forecast_hash,
                        "observation_raw_hash": observation_hash,
                        "collected_at_utc": collected_at,
                    }
                )
                silver_rows.extend(
                    _silver_rows(
                        forecast_frame,
                        station=station,
                        target_date=target_date,
                        model=model,
                        source="historical_forecast",
                    )
                )
                silver_rows.extend(
                    _silver_rows(
                        observation_frame,
                        station=station,
                        target_date=target_date,
                        model=model,
                        source="historical_weather",
                    )
                )
                gold_rows.append(
                    {
                        "station_id": station.station_id,
                        "city": station.city,
                        "target_date": target_date.isoformat(),
                        "timezone": station.timezone,
                        "model_name": model,
                        "forecast_source_kind": "historical_forecast",
                        "issue_time_utc": issue_time.isoformat(),
                        "lead_hours": 0.0,
                        "forecast_temperature_2m_min": _min(forecast_frame, "temperature_2m"),
                        "forecast_temperature_2m_mean": _mean(forecast_frame, "temperature_2m"),
                        "forecast_temperature_2m_max": _max(forecast_frame, "temperature_2m"),
                        "forecast_dew_point_2m_mean": _mean(forecast_frame, "dew_point_2m"),
                        "forecast_relative_humidity_2m_mean": _mean(forecast_frame, "relative_humidity_2m"),
                        "forecast_wind_speed_10m_mean": _mean(forecast_frame, "wind_speed_10m"),
                        "forecast_cloud_cover_mean": _mean(forecast_frame, "cloud_cover"),
                        "forecast_hourly_json": _hourly_json(forecast_frame),
                        "realized_daily_max": realized_daily_max,
                        "observation_hourly_json": _hourly_json(observation_frame),
                        "source_metadata_json": json.dumps(source_metadata, sort_keys=True),
                        "forecast_raw_hash": forecast_hash,
                        "observation_raw_hash": observation_hash,
                    }
                )
                if progress_callback is not None:
                    progress_callback(
                        WeatherTrainingProgressEvent(
                            phase="available",
                            station_id=station.station_id,
                            city=station.city,
                            target_date=target_date.isoformat(),
                            model_name=model,
                            item_index=item_index,
                            item_total=item_total,
                            elapsed_seconds=time.perf_counter() - started_at,
                            realized_daily_max=realized_daily_max,
                            message="weather row collected",
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                status_code = _status_code_from_exception(exc)
                status = "retryable_error" if status_code == 429 else "error"
                error = str(exc)
                failures.append(
                    {
                        "station_id": station.station_id,
                        "city": station.city,
                        "target_date": target_date.isoformat(),
                        "status": status,
                        "error": error,
                    }
                )
                bronze_rows.append(
                    {
                        "station_id": station.station_id,
                        "city": station.city,
                        "target_date": target_date.isoformat(),
                        "timezone": station.timezone,
                        "model_name": model,
                        "forecast_source_kind": "historical_forecast",
                        "status": status,
                        "error": error,
                        "forecast_raw_path": "",
                        "observation_raw_path": "",
                        "forecast_raw_hash": "",
                        "observation_raw_hash": "",
                        "collected_at_utc": collected_at,
                    }
                )
                if progress_callback is not None:
                    progress_callback(
                        WeatherTrainingProgressEvent(
                            phase=status,
                            station_id=station.station_id,
                            city=station.city,
                            target_date=target_date.isoformat(),
                            model_name=model,
                            item_index=item_index,
                            item_total=item_total,
                            elapsed_seconds=time.perf_counter() - started_at,
                            status_code=status_code,
                            message=error,
                        )
                    )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    _merge_and_write(
        paths.bronze_path,
        pd.DataFrame(bronze_rows),
        key_columns=["station_id", "target_date", "model_name"],
        columns=BRONZE_COLUMNS,
    )
    _merge_and_write(
        paths.silver_path,
        pd.DataFrame(silver_rows),
        key_columns=["station_id", "target_date", "model_name", "source", "time_local"],
        columns=SILVER_COLUMNS,
    )
    _merge_and_write(
        paths.gold_path,
        pd.DataFrame(gold_rows),
        key_columns=["station_id", "target_date", "model_name"],
        columns=GOLD_COLUMNS,
    )
    status_counts: dict[str, int] = {}
    for row in bronze_rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    return WeatherTrainingCollectionResult(
        paths=paths,
        requested_rows=len(stations) * len(dates),
        collected_rows=len(gold_rows),
        skipped_existing_rows=skipped,
        failed_rows=len(failures),
        status_counts=status_counts,
        failures=failures,
        workers_requested=workers,
        workers_effective=workers_effective,
    )
