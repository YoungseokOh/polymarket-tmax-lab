"""Historical and live feature-row construction."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pandas as pd

from pmtmax.http import CachedHttpClient
from pmtmax.logging_utils import get_logger
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.markets.station_registry import lookup_station
from pmtmax.modeling.bin_mapper import infer_winning_label
from pmtmax.storage.duckdb_store import DuckDBStore
from pmtmax.storage.parquet_store import ParquetStore
from pmtmax.storage.schemas import DecisionPoint, MarketSnapshot
from pmtmax.utils import dump_json
from pmtmax.weather.features import build_hourly_feature_frame, target_day_features
from pmtmax.weather.openmeteo_client import OpenMeteoClient
from pmtmax.weather.truth_sources import make_truth_source

LOGGER = get_logger(__name__)

HORIZON_OFFSETS: dict[str, tuple[int, int]] = {
    "market_open": (-2, 12),
    "previous_evening": (-1, 18),
    "morning_of": (0, 6),
    "hourly_refresh": (0, 9),
}
DEFAULT_MODELS = ["ecmwf_ifs025", "ecmwf_aifs025_single", "kma_gdps", "gfs_seamless"]
HOURLY = ["temperature_2m", "dew_point_2m", "relative_humidity_2m", "wind_speed_10m", "cloud_cover"]


def _as_float(value: object, default: float = 0.0) -> float:
    """Convert possibly object-typed row values to float."""

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    if isinstance(value, int | float):
        return float(value)
    if hasattr(value, "__float__"):
        return float(value)
    return default


@dataclass
class DatasetBuilder:
    """Build no-lookahead training datasets and live feature rows."""

    http: CachedHttpClient
    openmeteo: OpenMeteoClient
    duckdb_store: DuckDBStore | None
    parquet_store: ParquetStore | None
    snapshot_dir: Path | None = None
    fixture_dir: Path | None = None
    models: list[str] | None = None

    def build(
        self,
        snapshots: list[MarketSnapshot],
        output_name: str = "historical_training_set",
        decision_horizons: list[str] | None = None,
    ) -> pd.DataFrame:
        """Build and persist a training DataFrame for parsed market snapshots."""

        horizons = decision_horizons or ["market_open", "previous_evening", "morning_of"]
        if self.duckdb_store is None or self.parquet_store is None:
            msg = "duckdb_store and parquet_store are required for dataset builds"
            raise ValueError(msg)
        rows: list[dict[str, object]] = []
        valid_snapshots = [snapshot for snapshot in snapshots if snapshot.spec is not None]
        for snapshot in valid_snapshots:
            for horizon in horizons:
                rows.append(self._build_historical_row(snapshot, horizon))
        frame = pd.DataFrame(rows)
        self.duckdb_store.write_frame(output_name, frame)
        self.parquet_store.write_frame(f"{output_name}.parquet", frame)
        dump_json(
            self.parquet_store.root.parent / f"{output_name}_snapshots.json",
            [snapshot.model_dump(mode="json") for snapshot in valid_snapshots],
        )
        return frame

    def build_live_row(self, spec: MarketSpec, horizon: str = "morning_of") -> pd.DataFrame:
        """Build a single-row feature DataFrame for live or paper trading."""

        lat, lon, timezone = self._station_coords(spec)
        decision_point = self._decision_point(spec, horizon)
        tz = ZoneInfo(timezone)
        now_local_date = dt.datetime.now(tz=tz).date()
        forecast_days = max(
            1,
            (spec.target_local_date - decision_point.decision_time_utc.astimezone(tz).date()).days + 1,
            (spec.target_local_date - now_local_date).days + 2,
        )

        row: dict[str, object] = {
            "market_id": spec.market_id,
            "station_id": spec.station_id,
            "city": spec.city,
            "truth_track": spec.truth_track,
            "settlement_eligible": spec.settlement_eligible,
            "target_date": pd.Timestamp(spec.target_local_date),
            "decision_horizon": horizon,
            "decision_time_utc": pd.Timestamp(decision_point.decision_time_utc),
            "issue_time_utc": pd.Timestamp(decision_point.issue_time_utc),
            "lead_hours": decision_point.lead_hours,
            "market_spec_json": spec.model_dump_json(),
        }
        self._populate_weather_features(row, lat, lon, timezone, spec.target_local_date, historical=False, city=spec.city, forecast_days=forecast_days, unit=spec.unit)
        return pd.DataFrame([row])

    def _build_historical_row(self, snapshot: MarketSnapshot, horizon: str) -> dict[str, object]:
        spec = snapshot.spec
        if spec is None:
            msg = "Snapshot is missing a parsed spec"
            raise ValueError(msg)
        lat, lon, timezone = self._station_coords(spec)
        decision_point = self._decision_point(spec, horizon)
        row: dict[str, object] = {
            "market_id": spec.market_id,
            "station_id": spec.station_id,
            "city": spec.city,
            "truth_track": spec.truth_track,
            "settlement_eligible": spec.settlement_eligible,
            "target_date": pd.Timestamp(spec.target_local_date),
            "decision_horizon": horizon,
            "decision_time_utc": pd.Timestamp(decision_point.decision_time_utc),
            "issue_time_utc": pd.Timestamp(decision_point.issue_time_utc),
            "lead_hours": decision_point.lead_hours,
            "market_spec_json": spec.model_dump_json(),
            "market_prices_json": json.dumps(snapshot.outcome_prices, sort_keys=True),
        }
        self._populate_weather_features(row, lat, lon, timezone, spec.target_local_date, historical=True, city=spec.city, unit=spec.unit)

        truth_source = make_truth_source(spec, self.http, snapshot_dir=self.snapshot_dir)
        observation = truth_source.fetch_daily_observation(spec, spec.target_local_date)
        row["realized_daily_max"] = observation.daily_max
        row["winning_outcome"] = infer_winning_label(spec, observation.daily_max)
        return row

    @staticmethod
    def _convert_features(features: dict[str, float], unit: str) -> dict[str, float]:
        """Convert Celsius NWP features to the market's unit if needed."""
        if unit != "F":
            return features
        # Suffixes that are absolute temperatures (need full C→F: ×1.8 + 32)
        temp_suffixes = ("_max", "_mean", "_min", "_midday_temp", "midday_temp", "_dew_point_mean", "dew_point_mean")
        # Suffixes that are temperature differences (need scale only: ×1.8)
        diff_suffixes = ("diurnal_amplitude",)
        converted = {}
        for key, value in features.items():
            if any(key.endswith(s) or key == s for s in temp_suffixes):
                converted[key] = value * 9.0 / 5.0 + 32.0
            elif any(key.endswith(s) or key == s for s in diff_suffixes):
                converted[key] = value * 9.0 / 5.0
            else:
                converted[key] = value
        return converted

    def _populate_weather_features(
        self,
        row: dict[str, object],
        latitude: float,
        longitude: float,
        timezone: str,
        target_date: dt.date,
        *,
        historical: bool,
        city: str,
        forecast_days: int = 1,
        unit: str = "C",
    ) -> None:
        for model in (self.models or DEFAULT_MODELS):
            if historical:
                payload = self._historical_payload(
                    city=city,
                    latitude=latitude,
                    longitude=longitude,
                    model=model,
                    timezone=timezone,
                    target_date=target_date,
                )
            else:
                try:
                    payload = self.openmeteo.forecast(
                        latitude=latitude,
                        longitude=longitude,
                        model=model,
                        hourly=HOURLY,
                        forecast_days=forecast_days,
                        timezone=timezone,
                    )
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Forecast unavailable for model %s at (%s, %s), skipping", model, latitude, longitude)
                    continue
            package = build_hourly_feature_frame(payload)
            raw_features = target_day_features(package, target_date)
            features = self._convert_features(raw_features, unit)
            for key, value in features.items():
                row[f"{model}_{key}"] = value
            if "model_daily_max" in features and "model_daily_max" not in row:
                row["model_daily_max"] = features["model_daily_max"]

        row["neighbor_mean_temp"] = _as_float(
            row.get("ecmwf_ifs025_model_daily_mean", row.get("model_daily_max", 0.0))
        )
        row["neighbor_spread"] = abs(
            _as_float(row.get("ecmwf_ifs025_model_daily_max", 0.0))
            - _as_float(row.get("ecmwf_aifs025_single_model_daily_max", 0.0))
        )

    def _decision_point(self, spec: MarketSpec, horizon: str) -> DecisionPoint:
        if horizon not in HORIZON_OFFSETS:
            msg = f"Unsupported decision horizon: {horizon}"
            raise ValueError(msg)
        day_offset, hour = HORIZON_OFFSETS[horizon]
        timezone = ZoneInfo(spec.timezone)
        target_local_midnight = dt.datetime.combine(spec.target_local_date, dt.time(0, 0), tzinfo=timezone)
        decision_local = target_local_midnight + dt.timedelta(days=day_offset, hours=hour)
        decision_utc = decision_local.astimezone(dt.UTC)
        issue_utc = decision_utc - dt.timedelta(hours=6)
        lead_hours = max((target_local_midnight.astimezone(dt.UTC) - decision_utc).total_seconds() / 3600.0, 1.0)
        return DecisionPoint(
            horizon=horizon,
            decision_time_utc=decision_utc,
            issue_time_utc=issue_utc,
            lead_hours=lead_hours,
        )

    def _station_coords(self, spec: MarketSpec) -> tuple[float, float, str]:
        if spec.station_lat is not None and spec.station_lon is not None:
            return spec.station_lat, spec.station_lon, spec.timezone
        if definition := lookup_station(spec.city):
            return definition.lat, definition.lon, definition.timezone
        msg = f"Missing station coordinates for {spec.city}"
        raise ValueError(msg)

    def _historical_payload(
        self,
        *,
        city: str,
        latitude: float,
        longitude: float,
        model: str,
        timezone: str,
        target_date: dt.date,
    ) -> dict:
        try:
            return self.openmeteo.historical_forecast(
                latitude=latitude,
                longitude=longitude,
                model=model,
                hourly=HOURLY,
                start_date=target_date.isoformat(),
                end_date=target_date.isoformat(),
                timezone=timezone,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "historical_forecast_fallback",
                extra={"city": city, "model": model, "error": str(exc)},
            )
            if self.fixture_dir is None:
                raise
            fixture_name = city.lower().replace(" ", "_") + "_daily.json"
            fixture_path = self.fixture_dir / fixture_name
            if not fixture_path.exists():
                raise
            return cast(dict, json.loads(fixture_path.read_text()))
