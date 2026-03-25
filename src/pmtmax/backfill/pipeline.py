"""Bronze/silver/gold backfill orchestration."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from pmtmax.backtest.dataset_builder import HORIZON_OFFSETS
from pmtmax.http import CachedHttpClient
from pmtmax.logging_utils import get_logger
from pmtmax.markets.clob_read_client import ClobReadClient
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.markets.station_registry import lookup_station
from pmtmax.modeling.bin_mapper import infer_winning_label
from pmtmax.storage.schemas import MarketSnapshot
from pmtmax.storage.warehouse import DataWarehouse
from pmtmax.utils import stable_hash
from pmtmax.weather.features import build_hourly_feature_frame, target_day_features
from pmtmax.weather.openmeteo_client import OpenMeteoClient
from pmtmax.weather.truth_sources import make_truth_source
from pmtmax.weather.truth_sources.base import TruthSourceLagError

LOGGER = get_logger(__name__)

DEFAULT_HOURLY = [
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
]
PROBE_HOURLY = ["temperature_2m"]
VARIABLE_FALLBACKS: list[list[str]] = [
    DEFAULT_HOURLY,
    ["temperature_2m", "dew_point_2m", "relative_humidity_2m", "wind_speed_10m"],
    ["temperature_2m", "dew_point_2m", "relative_humidity_2m"],
    PROBE_HOURLY,
]


def _convert_celsius_features(features: dict[str, float], unit: str) -> dict[str, float]:
    """Convert Celsius NWP features to market unit if needed."""
    if unit != "F":
        return features
    temp_suffixes = ("_max", "_daily_mean", "_min", "_midday_temp", "midday_temp", "_dew_point_mean", "dew_point_mean")
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


def _coerce_float(value: object) -> float:
    """Convert mixed row values to floats for feature aggregation."""

    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _coerce_optional_float(value: object) -> float | None:
    """Convert mixed row values to an optional float for sparse diagnostics."""

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def _rows_for_market(frame: pd.DataFrame, market_id: str) -> pd.DataFrame:
    """Return rows for one market or an empty frame when the key column is absent."""

    if "market_id" not in frame.columns:
        return pd.DataFrame()
    return frame.loc[frame["market_id"] == market_id].copy()


def _normalize_price_history_points(payload: dict[str, Any]) -> list[dict[str, object]]:
    """Normalize Polymarket price history payloads into timestamp/price pairs."""

    normalized: list[dict[str, object]] = []
    for point in cast(list[dict[str, object]], payload.get("history", [])):
        timestamp = pd.to_datetime(point.get("t"), unit="s", errors="coerce", utc=True)
        price = pd.to_numeric(point.get("p"), errors="coerce")
        if pd.isna(timestamp) or pd.isna(price):
            continue
        normalized.append({"timestamp": pd.Timestamp(timestamp), "price": float(price)})
    return normalized


def _extract_last_trade_diagnostics(payload: dict[str, Any]) -> dict[str, object]:
    """Normalize last-trade diagnostics from mixed CLOB payload shapes."""

    last_trade_price: float | None = None
    for key in ("price", "last_trade_price", "lastTradePrice"):
        if key not in payload:
            continue
        last_trade_price = _coerce_optional_float(payload.get(key))
        if last_trade_price is not None:
            break

    last_trade_side: str | None = None
    for key in ("side", "last_trade_side", "lastTradeSide"):
        value = payload.get(key)
        if value in (None, ""):
            continue
        last_trade_side = str(value)
        break

    return {
        "last_trade_price": last_trade_price,
        "last_trade_side": last_trade_side,
        "last_trade_present": last_trade_price is not None,
    }


@dataclass
class BackfillPipeline:
    """Backfill bronze/silver tables and materialize gold datasets."""

    http: CachedHttpClient
    openmeteo: OpenMeteoClient
    warehouse: DataWarehouse
    models: list[str]
    truth_snapshot_dir: Path | None = None
    forecast_fixture_dir: Path | None = None
    run_id: str | None = None
    data_version: str = "v1"

    def backfill_markets(self, snapshots: list[MarketSnapshot], source_name: str = "snapshot") -> dict[str, pd.DataFrame]:
        """Persist raw market payloads and normalized market specs."""

        bronze_rows: list[dict[str, object]] = []
        silver_rows: list[dict[str, object]] = []
        station_rows: list[dict[str, object]] = []
        source_rows: list[dict[str, object]] = []
        for snapshot in snapshots:
            market_id = str(snapshot.market.get("id") or (snapshot.spec.market_id if snapshot.spec else "unknown"))
            captured_at = pd.Timestamp(snapshot.captured_at)
            relative_path = (
                f"markets/{source_name}/{captured_at.strftime('%Y%m%dT%H%M%SZ')}_{market_id}.json"
            )
            artifact = self.warehouse.raw_store.write_json(relative_path, snapshot.market)
            bronze_rows.append(
                {
                    "market_id": market_id,
                    "event_id": snapshot.market.get("eventId"),
                    "slug": snapshot.market.get("slug"),
                    "question": snapshot.market.get("question"),
                    "captured_at": captured_at,
                    "source_name": source_name,
                    "raw_path": artifact.relative_path,
                    "raw_hash": artifact.content_hash,
                    "parse_status": "ok" if snapshot.spec is not None else "failed",
                    "parse_error": snapshot.parse_error,
                    "outcome_prices_json": json.dumps(snapshot.outcome_prices, sort_keys=True),
                    "token_ids_json": json.dumps(snapshot.clob_token_ids),
                    "city": snapshot.spec.city if snapshot.spec else None,
                    "target_local_date": pd.Timestamp(snapshot.spec.target_local_date) if snapshot.spec else pd.NaT,
                    "official_source_name": snapshot.spec.official_source_name if snapshot.spec else None,
                    **self._metadata_fields(
                        created_at=captured_at.to_pydatetime(),
                        source_priority=self._source_priority(source_name),
                    ),
                }
            )
            if snapshot.spec is None:
                continue
            spec = snapshot.spec
            silver_rows.append(
                {
                    "market_id": spec.market_id,
                    "event_id": spec.event_id,
                    "slug": spec.slug,
                    "question": spec.question,
                    "city": spec.city,
                    "country": spec.country,
                    "target_local_date": pd.Timestamp(spec.target_local_date),
                    "timezone": spec.timezone,
                    "official_source_name": spec.official_source_name,
                    "official_source_url": spec.official_source_url,
                    "station_id": spec.station_id,
                    "station_name": spec.station_name,
                    "station_lat": spec.station_lat,
                    "station_lon": spec.station_lon,
                    "truth_track": spec.truth_track,
                    "settlement_eligible": spec.settlement_eligible,
                    "public_truth_source_name": spec.public_truth_source_name,
                    "public_truth_station_id": spec.public_truth_station_id,
                    "research_priority": spec.research_priority,
                    "unit": spec.unit,
                    "precision_rule_json": spec.precision_rule.model_dump_json(),
                    "outcome_schema_json": json.dumps(
                        [outcome.model_dump(mode="json") for outcome in spec.outcome_schema],
                        sort_keys=True,
                    ),
                    "finalization_policy_json": spec.finalization_policy.model_dump_json(),
                    "token_ids_json": json.dumps(spec.token_ids),
                    "notes": spec.notes,
                    "spec_json": spec.model_dump_json(),
                    **self._metadata_fields(source_priority=100),
                }
            )
            station_rows.append(
                {
                    "station_id": spec.station_id,
                    "station_name": spec.station_name,
                    "city": spec.city,
                    "country": spec.country,
                    "timezone": spec.timezone,
                    "station_lat": spec.station_lat,
                    "station_lon": spec.station_lon,
                    "official_source_name": spec.official_source_name,
                    "truth_track": spec.truth_track,
                    "settlement_eligible": spec.settlement_eligible,
                    "public_truth_source_name": spec.public_truth_source_name,
                    "public_truth_station_id": spec.public_truth_station_id,
                    "research_priority": spec.research_priority,
                    **self._metadata_fields(source_priority=100),
                }
            )
            source_rows.append(
                {
                    "source_key": stable_hash(f"{spec.official_source_name}|{spec.station_id}")[:16],
                    "source_name": spec.official_source_name,
                    "source_url": spec.official_source_url,
                    "station_id": spec.station_id,
                    "city": spec.city,
                    "source_kind": spec.adapter_key(),
                    "truth_track": spec.truth_track,
                    "settlement_eligible": spec.settlement_eligible,
                    **self._metadata_fields(source_priority=100),
                }
            )

        bronze = self.warehouse.upsert_table("bronze_market_snapshots", pd.DataFrame(bronze_rows))
        silver = self.warehouse.upsert_table("silver_market_specs", pd.DataFrame(silver_rows))
        if station_rows:
            self.warehouse.upsert_table("dim_station", pd.DataFrame(station_rows))
        if source_rows:
            self.warehouse.upsert_table("dim_source", pd.DataFrame(source_rows))
        self.warehouse.write_manifest()
        return {"bronze_market_snapshots": bronze, "silver_market_specs": silver}

    def backfill_forecasts(
        self,
        snapshots: list[MarketSnapshot],
        *,
        models: list[str] | None = None,
        allow_fixture_fallback: bool = False,
        strict_archive: bool = True,
        single_run_horizons: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Persist raw forecast payloads and normalize hourly forecast rows."""

        selected_models = models or self.models
        bronze_history = self.warehouse.read_table("bronze_forecast_requests")
        negative_cache = self._negative_cache_signatures(bronze_history)
        bronze_rows: list[dict[str, object]] = []
        silver_rows: list[dict[str, object]] = []
        model_rows: list[dict[str, object]] = []
        source_rows: list[dict[str, object]] = []
        for snapshot in snapshots:
            spec = snapshot.spec
            if spec is None:
                continue
            latitude, longitude, timezone = self._station_coords(spec)
            for model in selected_models:
                model_rows.append(
                    {
                        "model_name": model,
                        "provider": "open-meteo",
                        "city": spec.city,
                        "station_id": spec.station_id,
                        "endpoint_kind": self._forecast_endpoint_kind(spec),
                        **self._metadata_fields(source_priority=70),
                    }
                )
                fetched_at = dt.datetime.now(tz=dt.UTC)
                endpoint_kind = self._forecast_endpoint_kind(spec)
                if endpoint_kind == "historical_forecast":
                    probe_signature = self._forecast_query_signature(
                        spec=spec,
                        model=model,
                        endpoint_kind=endpoint_kind,
                        hourly=PROBE_HOURLY,
                        request_kind="probe",
                        decision_horizon=None,
                        latitude=latitude,
                        longitude=longitude,
                    )
                    if probe_signature in negative_cache:
                        bronze_rows.append(
                            self._forecast_request_row(
                                spec=spec,
                                model=model,
                                endpoint_kind=endpoint_kind,
                                request_kind="full",
                                decision_horizon=None,
                                requested_at=fetched_at,
                                latitude=latitude,
                                longitude=longitude,
                                timezone=timezone,
                                forecast_days=1,
                                status="skipped_negative_cache",
                                availability_status="unavailable",
                                variables=DEFAULT_HOURLY,
                                error_message="Skipped due to prior 404/422 availability probe.",
                            )
                        )
                        continue

                    probe_ok, probe_row = self._probe_forecast_availability(
                        spec=spec,
                        model=model,
                        latitude=latitude,
                        longitude=longitude,
                        timezone=timezone,
                        endpoint_kind=endpoint_kind,
                        fetched_at=fetched_at,
                    )
                    bronze_rows.append(probe_row)
                    if not probe_ok:
                        if probe_row.get("query_signature"):
                            negative_cache.add(str(probe_row["query_signature"]))
                        if strict_archive or not allow_fixture_fallback:
                            bronze_rows.append(
                                self._forecast_request_row(
                                    spec=spec,
                                    model=model,
                                    endpoint_kind=endpoint_kind,
                                    request_kind="full",
                                    decision_horizon=None,
                                    requested_at=fetched_at,
                                    latitude=latitude,
                                    longitude=longitude,
                                    timezone=timezone,
                                    forecast_days=1,
                                    status="skipped_probe_failure",
                                    availability_status="unavailable",
                                    variables=DEFAULT_HOURLY,
                                    error_message="Skipped full request after failed availability probe.",
                                )
                            )
                            continue

                try:
                    payload, variables = self._fetch_forecast_with_variable_fallback(
                        spec=spec,
                        model=model,
                        latitude=latitude,
                        longitude=longitude,
                        timezone=timezone,
                        endpoint_kind=endpoint_kind,
                    )
                    status = "ok"
                    availability_status = "available"
                    error_message = None
                    http_status: int | None = 200
                    reason = None
                    query_signature = self._forecast_query_signature(
                        spec=spec,
                        model=model,
                        endpoint_kind=endpoint_kind,
                        hourly=variables,
                        request_kind="full",
                        decision_horizon=None,
                        latitude=latitude,
                        longitude=longitude,
                    )
                except Exception as exc:  # noqa: BLE001
                    payload = None
                    variables = DEFAULT_HOURLY
                    status = "error"
                    availability_status = "error"
                    http_status, reason = self._error_details(exc)
                    error_message = str(exc)
                    query_signature = self._forecast_query_signature(
                        spec=spec,
                        model=model,
                        endpoint_kind=endpoint_kind,
                        hourly=variables,
                        request_kind="full",
                        decision_horizon=None,
                        latitude=latitude,
                        longitude=longitude,
                    )
                    if availability_status == "unavailable":
                        negative_cache.add(query_signature)
                    if allow_fixture_fallback and not strict_archive:
                        fixture = self._load_forecast_fixture(spec.city)
                        if fixture is not None:
                            payload = fixture
                            status = "fixture"
                            availability_status = "demo_fixture"
                            endpoint_kind = "fixture"
                            http_status = None
                            reason = "demo_fixture_fallback"
                            error_message = None
                    if payload is None:
                        bronze_rows.append(
                            self._forecast_request_row(
                                spec=spec,
                                model=model,
                                endpoint_kind=endpoint_kind,
                                request_kind="full",
                                decision_horizon=None,
                                requested_at=fetched_at,
                                latitude=latitude,
                                longitude=longitude,
                                timezone=timezone,
                                forecast_days=1,
                                status=status,
                                availability_status=availability_status,
                                variables=variables,
                                error_message=error_message,
                                http_status=http_status,
                                reason=reason,
                                query_signature=query_signature,
                            )
                        )
                        continue

                relative_path = (
                    f"forecasts/{endpoint_kind}/{spec.city.lower().replace(' ', '_')}/"
                    f"{spec.target_local_date.isoformat()}/{spec.market_id}_{model}.json"
                )
                artifact = self.warehouse.raw_store.write_json(relative_path, payload)
                bronze_rows.append(
                    self._forecast_request_row(
                        spec=spec,
                        model=model,
                        endpoint_kind=endpoint_kind,
                        request_kind="full",
                        decision_horizon=None,
                        requested_at=fetched_at,
                        latitude=latitude,
                        longitude=longitude,
                        timezone=timezone,
                        forecast_days=self._forecast_days(spec, timezone),
                        status=status,
                        availability_status=availability_status,
                        variables=variables,
                        raw_path=artifact.relative_path,
                        raw_hash=artifact.content_hash,
                        http_status=http_status,
                        reason=reason,
                        query_signature=query_signature,
                    )
                )
                silver_rows.extend(
                    self._normalize_forecast_rows(
                        spec=spec,
                        model_name=model,
                        endpoint_kind=endpoint_kind,
                        payload=cast(dict[str, Any], payload),
                        raw_path=artifact.relative_path,
                        raw_hash=artifact.content_hash,
                        availability_status=availability_status,
                        source_variables=variables,
                        retrieved_at=fetched_at,
                        decision_horizon=None,
                        issue_time_utc=None,
                        requested_run_time_utc=None,
                    )
                )
                if endpoint_kind == "historical_forecast" and single_run_horizons:
                    single_run_bronze, single_run_silver = self._backfill_single_run_requests(
                        spec=spec,
                        model=model,
                        latitude=latitude,
                        longitude=longitude,
                        timezone=timezone,
                        horizons=single_run_horizons,
                        strict_archive=strict_archive,
                    )
                    bronze_rows.extend(single_run_bronze)
                    silver_rows.extend(single_run_silver)
                source_rows.append(
                    {
                        "source_key": stable_hash(f"open-meteo|{endpoint_kind}|{model}")[:16],
                        "source_name": "open-meteo",
                        "source_url": self._forecast_endpoint_url(
                            spec=spec,
                            model=model,
                            endpoint_kind=endpoint_kind,
                            hourly=variables,
                            forecast_days=self._forecast_days(spec, timezone),
                            request_kind="full",
                            latitude=latitude,
                            longitude=longitude,
                            decision_horizon=None,
                        ),
                        "station_id": spec.station_id,
                        "city": spec.city,
                        "source_kind": endpoint_kind,
                        **self._metadata_fields(source_priority=self._source_priority(endpoint_kind)),
                    }
                )

        bronze = self.warehouse.upsert_table("bronze_forecast_requests", pd.DataFrame(bronze_rows))
        silver = self.warehouse.upsert_table("silver_forecast_runs_hourly", pd.DataFrame(silver_rows))
        if model_rows:
            self.warehouse.upsert_table("dim_model", pd.DataFrame(model_rows))
        if source_rows:
            self.warehouse.upsert_table("dim_source", pd.DataFrame(source_rows))
        self.warehouse.write_manifest()
        return {"bronze_forecast_requests": bronze, "silver_forecast_runs_hourly": silver}

    def backfill_truth(self, snapshots: list[MarketSnapshot]) -> dict[str, pd.DataFrame]:
        """Persist raw official truth payloads and normalized daily observations."""

        bronze_rows: list[dict[str, object]] = []
        silver_rows: list[dict[str, object]] = []
        for snapshot in snapshots:
            spec = snapshot.spec
            if spec is None:
                continue
            truth_source = make_truth_source(spec, self.http, snapshot_dir=self.truth_snapshot_dir)
            fetched_at = dt.datetime.now(tz=dt.UTC)
            try:
                bundle = truth_source.fetch_observation_bundle(spec, spec.target_local_date)
            except Exception as exc:  # noqa: BLE001
                status, latest_available_date = self._truth_failure_details(exc)
                bronze_rows.append(
                    {
                        "market_id": spec.market_id,
                        "station_id": spec.station_id,
                        "city": spec.city,
                        "official_source_name": spec.official_source_name,
                        "public_truth_source_name": spec.public_truth_source_name,
                        "public_truth_station_id": spec.public_truth_station_id,
                        "truth_track": spec.truth_track,
                        "settlement_eligible": spec.settlement_eligible,
                        "target_local_date": pd.Timestamp(spec.target_local_date),
                        "fetched_at": pd.Timestamp(fetched_at),
                        "status": status,
                        "raw_path": None,
                        "raw_hash": None,
                        "media_type": None,
                        "latest_available_date": (
                            pd.Timestamp(latest_available_date) if latest_available_date is not None else pd.NaT
                        ),
                        "source_url": spec.official_source_url,
                        "error_message": str(exc),
                        **self._metadata_fields(source_priority=100),
                    }
                )
                continue
            extension = "json" if bundle.media_type == "application/json" else "html"
            relative_path = (
                f"truth/{spec.adapter_key()}/{spec.station_id}/{spec.target_local_date.strftime('%Y%m')}/"
                f"{spec.market_id}_{spec.target_local_date.isoformat()}.{extension}"
            )
            if bundle.media_type == "application/json":
                artifact = self.warehouse.raw_store.write_json(relative_path, bundle.raw_payload)
            else:
                artifact = self.warehouse.raw_store.write_text(relative_path, str(bundle.raw_payload), bundle.media_type)
            bronze_rows.append(
                {
                    "market_id": spec.market_id,
                    "station_id": spec.station_id,
                    "city": spec.city,
                    "official_source_name": spec.official_source_name,
                    "public_truth_source_name": spec.public_truth_source_name,
                    "public_truth_station_id": spec.public_truth_station_id,
                    "truth_track": spec.truth_track,
                    "settlement_eligible": spec.settlement_eligible,
                    "target_local_date": pd.Timestamp(spec.target_local_date),
                    "fetched_at": pd.Timestamp(fetched_at),
                    "status": "ok",
                    "raw_path": artifact.relative_path,
                    "raw_hash": artifact.content_hash,
                    "media_type": artifact.media_type,
                    "latest_available_date": pd.NaT,
                    "source_url": bundle.source_url,
                    "error_message": None,
                    **self._metadata_fields(created_at=fetched_at, source_priority=100),
                }
            )
            silver_rows.append(
                {
                    "market_id": spec.market_id,
                    "station_id": bundle.observation.station_id,
                    "city": spec.city,
                    "source": bundle.observation.source,
                    "official_source_name": spec.official_source_name,
                    "truth_track": spec.truth_track,
                    "settlement_eligible": spec.settlement_eligible,
                    "target_local_date": pd.Timestamp(bundle.observation.local_date),
                    "daily_max": bundle.observation.daily_max,
                    "unit": bundle.observation.unit,
                    "finalized_at": pd.Timestamp(bundle.observation.finalized_at),
                    "revision_status": bundle.observation.revision_status,
                    "raw_path": artifact.relative_path,
                    "raw_hash": artifact.content_hash,
                    **self._metadata_fields(created_at=fetched_at, source_priority=100),
                }
            )

        bronze = self.warehouse.upsert_table("bronze_truth_snapshots", pd.DataFrame(bronze_rows))
        silver = self.warehouse.upsert_table("silver_observations_daily", pd.DataFrame(silver_rows))
        self.warehouse.write_manifest()
        return {"bronze_truth_snapshots": bronze, "silver_observations_daily": silver}

    def summarize_truth_coverage(self) -> dict[str, pd.DataFrame]:
        """Summarize truth fetch outcomes, lagged markets, and archive-ready statuses."""

        bronze = self.warehouse.read_table("bronze_truth_snapshots")
        if bronze.empty:
            return {"summary": pd.DataFrame(), "details": pd.DataFrame()}

        frame = bronze.copy()
        for column in ("target_local_date", "latest_available_date", "fetched_at"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        if {"target_local_date", "latest_available_date"}.issubset(frame.columns):
            frame["lag_days"] = (
                frame["target_local_date"].dt.normalize() - frame["latest_available_date"].dt.normalize()
            ).dt.days
        else:
            frame["lag_days"] = pd.NA

        summary = (
            frame.groupby(["status", "truth_track", "city", "official_source_name"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["status", "truth_track", "city", "official_source_name"], ignore_index=True)
        )
        detail_columns = [
            "market_id",
            "city",
            "station_id",
            "public_truth_station_id",
            "official_source_name",
            "public_truth_source_name",
            "truth_track",
            "settlement_eligible",
            "target_local_date",
            "latest_available_date",
            "lag_days",
            "status",
            "source_url",
            "error_message",
        ]
        details = frame.loc[:, [column for column in detail_columns if column in frame.columns]].sort_values(
            ["status", "city", "target_local_date", "market_id"],
            ignore_index=True,
        )
        return {"summary": summary, "details": details}

    def summarize_dataset_readiness(self, snapshots: list[MarketSnapshot]) -> dict[str, pd.DataFrame]:
        """Summarize forecast/truth/gold readiness for a target snapshot set."""

        valid_snapshots = [snapshot for snapshot in snapshots if snapshot.spec is not None]
        if not valid_snapshots:
            return {"summary": pd.DataFrame(), "details": pd.DataFrame()}

        base = pd.DataFrame(
            [
                {
                    "market_id": snapshot.spec.market_id,
                    "city": snapshot.spec.city,
                    "station_id": snapshot.spec.station_id,
                    "target_local_date": pd.Timestamp(snapshot.spec.target_local_date),
                    "truth_track": snapshot.spec.truth_track,
                    "settlement_eligible": snapshot.spec.settlement_eligible,
                    "official_source_name": snapshot.spec.official_source_name,
                    "public_truth_source_name": snapshot.spec.public_truth_source_name,
                }
                for snapshot in valid_snapshots
            ]
        )
        market_ids = set(base["market_id"].astype(str))

        forecast = self.warehouse.read_table("silver_forecast_runs_hourly")
        forecast_counts = pd.DataFrame(columns=["market_id", "forecast_row_count", "forecast_model_count"])
        if not forecast.empty and "market_id" in forecast.columns:
            filtered_forecast = forecast.loc[forecast["market_id"].astype(str).isin(market_ids)].copy()
            if not filtered_forecast.empty:
                forecast_counts = (
                    filtered_forecast.groupby("market_id", dropna=False)
                    .agg(
                        forecast_row_count=("market_id", "size"),
                        forecast_model_count=("model_name", lambda series: series.dropna().astype(str).nunique()),
                    )
                    .reset_index()
                )

        truth_bronze = self.warehouse.read_table("bronze_truth_snapshots")
        truth_latest = pd.DataFrame(
            columns=[
                "market_id",
                "truth_status",
                "latest_available_date",
                "truth_error_message",
            ]
        )
        if not truth_bronze.empty and "market_id" in truth_bronze.columns:
            filtered_truth = truth_bronze.loc[truth_bronze["market_id"].astype(str).isin(market_ids)].copy()
            if not filtered_truth.empty:
                sort_columns = [column for column in ("fetched_at", "created_at") if column in filtered_truth.columns]
                if sort_columns:
                    filtered_truth = filtered_truth.sort_values(["market_id", *sort_columns], ignore_index=True)
                truth_latest = (
                    filtered_truth.drop_duplicates(subset=["market_id"], keep="last")
                    .loc[
                        :,
                        [
                            column
                            for column in ("market_id", "status", "latest_available_date", "error_message")
                            if column in filtered_truth.columns
                        ],
                    ]
                    .rename(
                        columns={
                            "status": "truth_status",
                            "error_message": "truth_error_message",
                        }
                    )
                )

        truth_silver = self.warehouse.read_table("silver_observations_daily")
        truth_counts = pd.DataFrame(columns=["market_id", "truth_daily_count"])
        if not truth_silver.empty and "market_id" in truth_silver.columns:
            filtered_truth_daily = truth_silver.loc[truth_silver["market_id"].astype(str).isin(market_ids)].copy()
            if not filtered_truth_daily.empty:
                truth_counts = (
                    filtered_truth_daily.groupby("market_id", dropna=False)
                    .size()
                    .reset_index(name="truth_daily_count")
                )

        gold = self.warehouse.read_table("gold_training_examples_tabular")
        if gold.empty:
            gold = self.warehouse.read_table("gold_training_examples")
        gold_counts = pd.DataFrame(columns=["market_id", "gold_row_count", "gold_horizon_count"])
        if not gold.empty and "market_id" in gold.columns:
            filtered_gold = gold.loc[gold["market_id"].astype(str).isin(market_ids)].copy()
            if not filtered_gold.empty:
                gold_counts = (
                    filtered_gold.groupby("market_id", dropna=False)
                    .agg(
                        gold_row_count=("market_id", "size"),
                        gold_horizon_count=("decision_horizon", lambda series: series.dropna().astype(str).nunique()),
                    )
                    .reset_index()
                )

        details = (
            base.merge(forecast_counts, on="market_id", how="left")
            .merge(truth_latest, on="market_id", how="left")
            .merge(truth_counts, on="market_id", how="left")
            .merge(gold_counts, on="market_id", how="left")
        )
        for numeric_column in (
            "forecast_row_count",
            "forecast_model_count",
            "truth_daily_count",
            "gold_row_count",
            "gold_horizon_count",
        ):
            if numeric_column in details.columns:
                details[numeric_column] = details[numeric_column].fillna(0).astype(int)
        details["forecast_ready"] = details["forecast_row_count"] > 0
        details["truth_ready"] = details["truth_daily_count"] > 0
        details["gold_ready"] = details["gold_row_count"] > 0

        def _readiness_status(row: pd.Series) -> str:
            truth_status = str(row.get("truth_status") or "")
            if bool(row.get("gold_ready")):
                return "ready"
            if truth_status == "lag":
                return "truth_lag"
            if truth_status == "error":
                return "truth_error"
            if not bool(row.get("forecast_ready")):
                return "missing_forecast"
            if not bool(row.get("truth_ready")):
                return "missing_truth"
            return "gold_missing"

        details["readiness_status"] = details.apply(_readiness_status, axis=1)
        details = details.sort_values(["city", "target_local_date", "market_id"], ignore_index=True)

        summary = (
            details.groupby("city", dropna=False)
            .agg(
                snapshot_count=("market_id", "size"),
                forecast_ready_count=("forecast_ready", "sum"),
                truth_ok_count=("truth_ready", "sum"),
                truth_lag_count=("truth_status", lambda series: (series.fillna("").astype(str) == "lag").sum()),
                truth_error_count=("truth_status", lambda series: (series.fillna("").astype(str) == "error").sum()),
                gold_market_count=("gold_ready", "sum"),
                gold_row_count=("gold_row_count", "sum"),
            )
            .reset_index()
            .sort_values("city", ignore_index=True)
        )
        return {"summary": summary, "details": details}

    def backfill_price_history(
        self,
        snapshots: list[MarketSnapshot],
        *,
        clob: ClobReadClient,
        interval: str = "max",
        fidelity: int = 60,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Persist Polymarket official token price history for a historical market set."""

        bronze_rows: list[dict[str, object]] = []
        silver_rows: list[dict[str, object]] = []
        for snapshot in snapshots:
            spec = snapshot.spec
            if spec is None:
                continue
            token_ids = spec.token_ids or snapshot.clob_token_ids
            for outcome_label, token_id in zip(spec.outcome_labels(), token_ids, strict=False):
                requested_at = dt.datetime.now(tz=dt.UTC)
                raw_path: str | None = None
                raw_hash: str | None = None
                error_message: str | None = None
                http_status_code: int | None = None
                history: list[dict[str, object]] = []
                status = "empty"
                empty_reason: str | None = None
                last_trade_price: float | None = None
                last_trade_side: str | None = None
                last_trade_present: bool | None = None
                last_trade_http_status_code: int | None = None
                last_trade_error_message: str | None = None
                try:
                    payload = clob.get_prices_history(
                        token_id,
                        interval=interval,
                        fidelity=fidelity,
                        use_cache=use_cache,
                    )
                    history = _normalize_price_history_points(payload)
                    artifact = self.warehouse.raw_store.write_json(
                        (
                            "prices/history/"
                            f"{spec.city.lower().replace(' ', '_')}/"
                            f"{requested_at.strftime('%Y%m%dT%H%M%SZ')}_{spec.market_id}_{token_id}.json"
                        ),
                        payload,
                    )
                    raw_path = artifact.relative_path
                    raw_hash = artifact.content_hash
                    status = "ok" if history else "empty"
                    if status == "empty":
                        try:
                            last_trade_payload = clob.get_last_trade_price(token_id)
                            diagnostics = _extract_last_trade_diagnostics(last_trade_payload)
                            last_trade_price = cast(float | None, diagnostics["last_trade_price"])
                            last_trade_side = cast(str | None, diagnostics["last_trade_side"])
                            last_trade_present = cast(bool, diagnostics["last_trade_present"])
                            empty_reason = (
                                "history_unavailable_last_trade_present"
                                if last_trade_present
                                else "history_unavailable_last_trade_missing"
                            )
                        except httpx.HTTPStatusError as exc:
                            last_trade_http_status_code = exc.response.status_code
                            last_trade_error_message = str(exc)
                            last_trade_present = None
                            empty_reason = "history_unavailable_last_trade_http_error"
                        except Exception as exc:  # noqa: BLE001
                            last_trade_error_message = str(exc)
                            last_trade_present = None
                            empty_reason = "history_unavailable_last_trade_error"
                except httpx.HTTPStatusError as exc:
                    http_status_code = exc.response.status_code
                    error_message = str(exc)
                    status = "http_error"
                except Exception as exc:  # noqa: BLE001
                    error_message = str(exc)
                    status = "error"

                earliest_ts = pd.NaT
                latest_ts = pd.NaT
                if history:
                    timestamps = [pd.Timestamp(cast(pd.Timestamp, row["timestamp"])) for row in history]
                    earliest_ts = min(timestamps)
                    latest_ts = max(timestamps)
                    for point in history:
                        silver_rows.append(
                            {
                                "market_id": spec.market_id,
                                "token_id": token_id,
                                "outcome_label": outcome_label,
                                "city": spec.city,
                                "station_id": spec.station_id,
                                "target_local_date": pd.Timestamp(spec.target_local_date),
                                "timestamp": pd.Timestamp(cast(pd.Timestamp, point["timestamp"])),
                                "price": float(point["price"]),
                                "interval_used": interval,
                                "source": "polymarket_official",
                                **self._metadata_fields(created_at=requested_at, source_priority=100),
                            }
                        )

                bronze_rows.append(
                    {
                        "market_id": spec.market_id,
                        "event_id": spec.event_id,
                        "city": spec.city,
                        "station_id": spec.station_id,
                        "target_local_date": pd.Timestamp(spec.target_local_date),
                        "token_id": token_id,
                        "outcome_label": outcome_label,
                        "requested_at": pd.Timestamp(requested_at),
                        "status": status,
                        "interval_requested": interval,
                        "interval_used": interval,
                        "fidelity": fidelity,
                        "point_count": len(history),
                        "first_price_ts": earliest_ts,
                        "last_price_ts": latest_ts,
                        "empty_reason": empty_reason,
                        "last_trade_price": last_trade_price,
                        "last_trade_side": last_trade_side,
                        "last_trade_present": last_trade_present,
                        "last_trade_http_status_code": last_trade_http_status_code,
                        "last_trade_error_message": last_trade_error_message,
                        "http_status_code": http_status_code,
                        "error_message": error_message,
                        "raw_path": raw_path,
                        "raw_hash": raw_hash,
                        **self._metadata_fields(created_at=requested_at, source_priority=100),
                    }
                )

        bronze = self.warehouse.upsert_table("bronze_price_history_requests", pd.DataFrame(bronze_rows))
        silver = self.warehouse.upsert_table("silver_price_timeseries", pd.DataFrame(silver_rows))
        self.warehouse.write_manifest()
        return {"bronze_price_history_requests": bronze, "silver_price_timeseries": silver}

    def materialize_backtest_panel(
        self,
        dataset: pd.DataFrame,
        *,
        output_name: str = "historical_backtest_panel",
        max_price_age_minutes: int = 720,
    ) -> pd.DataFrame:
        """Build a token-level decision-time market-price panel from official history."""

        if dataset.empty:
            raise ValueError("Dataset is empty; nothing to materialize.")
        if "market_spec_json" not in dataset.columns or "decision_time_utc" not in dataset.columns:
            msg = "Dataset must include market_spec_json and decision_time_utc columns."
            raise ValueError(msg)

        timeseries = self.warehouse.read_table("silver_price_timeseries")
        if not timeseries.empty:
            timeseries = timeseries.copy()
            timeseries["timestamp"] = pd.to_datetime(timeseries["timestamp"], errors="coerce", utc=True)
            if "price" in timeseries.columns:
                timeseries["price"] = pd.to_numeric(timeseries["price"], errors="coerce")
        else:
            # Preserve the expected schema so empty price history fails closed as
            # `coverage_status="missing"` instead of raising on missing columns.
            timeseries = pd.DataFrame(columns=["market_id", "token_id", "timestamp", "price"])
        max_age_seconds = float(max_price_age_minutes * 60)
        rows: list[dict[str, object]] = []
        for _, dataset_row in dataset.iterrows():
            spec = MarketSpec.model_validate_json(str(dataset_row["market_spec_json"]))
            decision_ts = pd.to_datetime(dataset_row["decision_time_utc"], errors="coerce", utc=True)
            if pd.isna(decision_ts):
                continue
            market_timeseries = _rows_for_market(timeseries, spec.market_id).sort_values("timestamp", ignore_index=True)
            token_ids = spec.token_ids
            for outcome_label, token_id in zip(spec.outcome_labels(), token_ids, strict=False):
                panel_row: dict[str, object] = {
                    "market_id": spec.market_id,
                    "token_id": token_id,
                    "outcome_label": outcome_label,
                    "city": spec.city,
                    "station_id": spec.station_id,
                    "target_date": pd.Timestamp(spec.target_local_date),
                    "decision_horizon": str(dataset_row.get("decision_horizon")),
                    "decision_time_utc": decision_ts,
                    "winning_outcome": dataset_row.get("winning_outcome"),
                    "realized_daily_max": dataset_row.get("realized_daily_max"),
                    "market_price": pd.NA,
                    "price_ts": pd.NaT,
                    "price_age_seconds": pd.NA,
                    "interval_used": None,
                    "coverage_status": "missing",
                    **self._metadata_fields(source_priority=100),
                }
                if not market_timeseries.empty and "token_id" in market_timeseries.columns:
                    token_history = market_timeseries.loc[
                        market_timeseries["token_id"].astype(str) == token_id
                    ].copy()
                else:
                    token_history = pd.DataFrame()
                if token_history.empty:
                    rows.append(panel_row)
                    continue
                prior = token_history.loc[token_history["timestamp"] <= decision_ts].copy()
                if prior.empty:
                    rows.append(panel_row)
                    continue
                latest = prior.sort_values("timestamp", ignore_index=True).iloc[-1]
                price_ts = pd.to_datetime(latest.get("timestamp"), errors="coerce", utc=True)
                price = pd.to_numeric(latest.get("price"), errors="coerce")
                if pd.isna(price_ts) or pd.isna(price):
                    rows.append(panel_row)
                    continue
                age_seconds = max((decision_ts - price_ts).total_seconds(), 0.0)
                panel_row["market_price"] = float(price)
                panel_row["price_ts"] = price_ts
                panel_row["price_age_seconds"] = float(age_seconds)
                panel_row["interval_used"] = latest.get("interval_used")
                panel_row["coverage_status"] = "ok" if age_seconds <= max_age_seconds else "stale"
                rows.append(panel_row)

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError("No backtest panel rows materialized.")
        self.warehouse.write_gold_table(
            "gold_backtest_panel",
            frame,
            relative_path=f"gold/{output_name}.parquet",
        )
        self.warehouse.write_manifest()
        return frame

    def summarize_price_history_coverage(self, market_ids: set[str] | None = None) -> dict[str, pd.DataFrame]:
        """Summarize official price-history fetch coverage and decision-time panel readiness."""

        bronze = self.warehouse.read_table("bronze_price_history_requests")
        panel = self.warehouse.read_table("gold_backtest_panel")
        if not bronze.empty and {"market_id", "token_id", "requested_at"}.issubset(bronze.columns):
            bronze = bronze.copy()
            bronze["requested_at"] = pd.to_datetime(bronze["requested_at"], errors="coerce", utc=True)
            bronze = (
                bronze.sort_values(["market_id", "token_id", "requested_at"], ignore_index=True)
                .drop_duplicates(subset=["market_id", "token_id"], keep="last")
                .reset_index(drop=True)
            )
        if market_ids:
            wanted = {str(market_id) for market_id in market_ids}
            if not bronze.empty and "market_id" in bronze.columns:
                bronze = bronze.loc[bronze["market_id"].astype(str).isin(wanted)].copy()
            if not panel.empty and "market_id" in panel.columns:
                panel = panel.loc[panel["market_id"].astype(str).isin(wanted)].copy()

        request_summary = pd.DataFrame()
        request_details = pd.DataFrame()
        if not bronze.empty:
            summary_agg: dict[str, tuple[str, object]] = {
                "request_count": ("token_id", "size"),
                "market_count": ("market_id", lambda series: series.astype(str).nunique()),
                "total_points": ("point_count", "sum"),
            }
            if "last_trade_present" in bronze.columns:
                summary_agg["last_trade_probe_count"] = ("last_trade_present", lambda series: series.notna().sum())
                summary_agg["last_trade_present_count"] = (
                    "last_trade_present",
                    lambda series: series.fillna(False).astype(bool).sum(),
                )
            if "empty_reason" in bronze.columns:
                summary_agg["history_empty_with_last_trade_count"] = (
                    "empty_reason",
                    lambda series: (
                        series.fillna("").astype(str) == "history_unavailable_last_trade_present"
                    ).sum(),
                )
                summary_agg["history_empty_without_last_trade_count"] = (
                    "empty_reason",
                    lambda series: (
                        series.fillna("").astype(str) == "history_unavailable_last_trade_missing"
                    ).sum(),
                )
                summary_agg["history_empty_last_trade_error_count"] = (
                    "empty_reason",
                    lambda series: (
                        series.fillna("").astype(str).str.startswith("history_unavailable_last_trade_")
                        & ~series.fillna("").astype(str).isin(
                            [
                                "history_unavailable_last_trade_present",
                                "history_unavailable_last_trade_missing",
                            ]
                        )
                    ).sum(),
                )
            request_summary = (
                bronze.groupby(["status", "city"], dropna=False)
                .agg(**summary_agg)
                .reset_index()
                .sort_values(["status", "city"], ignore_index=True)
            )
            request_details = bronze.loc[
                :,
                [
                    column
                    for column in (
                        "market_id",
                        "city",
                        "target_local_date",
                        "token_id",
                        "outcome_label",
                        "status",
                        "point_count",
                        "first_price_ts",
                        "last_price_ts",
                        "empty_reason",
                        "last_trade_price",
                        "last_trade_side",
                        "last_trade_present",
                        "last_trade_http_status_code",
                        "last_trade_error_message",
                        "http_status_code",
                        "error_message",
                    )
                    if column in bronze.columns
                ],
            ].sort_values(["city", "target_local_date", "market_id", "outcome_label"], ignore_index=True)

        panel_summary = pd.DataFrame()
        market_summary = pd.DataFrame()
        panel_details = pd.DataFrame()
        if not panel.empty:
            panel_summary = (
                panel.groupby(["city", "decision_horizon", "coverage_status"], dropna=False)
                .agg(
                    token_count=("token_id", "size"),
                    market_count=("market_id", lambda series: series.astype(str).nunique()),
                )
                .reset_index()
                .sort_values(["city", "decision_horizon", "coverage_status"], ignore_index=True)
            )
            market_summary = (
                panel.groupby(["city", "market_id", "decision_horizon"], dropna=False)
                .agg(
                    token_count=("token_id", "size"),
                    ok_token_count=("coverage_status", lambda series: (series.astype(str) == "ok").sum()),
                    stale_token_count=("coverage_status", lambda series: (series.astype(str) == "stale").sum()),
                    missing_token_count=("coverage_status", lambda series: (series.astype(str) == "missing").sum()),
                )
                .reset_index()
                .sort_values(["city", "decision_horizon", "market_id"], ignore_index=True)
            )
            market_summary["market_ready"] = market_summary["ok_token_count"] > 0
            panel_details = panel.loc[
                :,
                [
                    column
                    for column in (
                        "market_id",
                        "city",
                        "target_date",
                        "decision_horizon",
                        "token_id",
                        "outcome_label",
                        "coverage_status",
                        "market_price",
                        "price_ts",
                        "price_age_seconds",
                        "interval_used",
                    )
                    if column in panel.columns
                ],
            ].sort_values(["city", "target_date", "decision_horizon", "outcome_label"], ignore_index=True)

        return {
            "request_summary": request_summary,
            "request_details": request_details,
            "panel_summary": panel_summary,
            "market_summary": market_summary,
            "details": panel_details,
        }

    def materialize_training_set(
        self,
        snapshots: list[MarketSnapshot],
        *,
        output_name: str = "historical_training_set",
        decision_horizons: list[str] | None = None,
        contract: str = "both",
    ) -> pd.DataFrame:
        """Build the gold training dataset from normalized bronze/silver tables."""

        if contract not in {"tabular", "sequence", "both"}:
            msg = f"Unsupported materialization contract: {contract}"
            raise ValueError(msg)
        horizons = decision_horizons or ["market_open", "previous_evening", "morning_of"]
        forecast_frame = self.warehouse.read_table("silver_forecast_runs_hourly")
        truth_frame = self.warehouse.read_table("silver_observations_daily")
        rows: list[dict[str, object]] = []
        sequence_rows: list[dict[str, object]] = []
        for snapshot in snapshots:
            spec = snapshot.spec
            if spec is None:
                continue
            market_forecasts = _rows_for_market(forecast_frame, spec.market_id)
            market_truth = _rows_for_market(truth_frame, spec.market_id)
            if market_forecasts.empty or market_truth.empty:
                LOGGER.warning(
                    "materialize_training_set_skip",
                    extra={"market_id": spec.market_id, "reason": "missing_forecast_or_truth"},
                )
                continue
            truth_row = market_truth.iloc[-1]
            for horizon in horizons:
                decision_point = self._decision_point(spec, horizon)
                selected_forecasts = self._select_forecasts_for_horizon(market_forecasts, horizon)
                if selected_forecasts.empty:
                    LOGGER.warning(
                        "materialize_training_set_skip_horizon",
                        extra={"market_id": spec.market_id, "horizon": horizon, "reason": "missing_horizon_forecasts"},
                    )
                    continue
                selected_models = sorted(selected_forecasts["model_name"].dropna().astype(str).unique().tolist())
                issue_time = self._materialized_issue_time(selected_forecasts, decision_point["issue_time_utc"])
                winning_outcome = infer_winning_label(spec, float(truth_row["daily_max"]))
                row: dict[str, object] = {
                    "market_id": spec.market_id,
                    "station_id": spec.station_id,
                    "city": spec.city,
                    "truth_track": spec.truth_track,
                    "settlement_eligible": spec.settlement_eligible,
                    "target_date": pd.Timestamp(spec.target_local_date),
                    "decision_horizon": horizon,
                    "decision_time_utc": pd.Timestamp(decision_point["decision_time_utc"]),
                    "issue_time_utc": issue_time,
                    "lead_hours": decision_point["lead_hours"],
                    "market_spec_json": spec.model_dump_json(),
                    "market_prices_json": json.dumps(snapshot.outcome_prices, sort_keys=True),
                    "realized_daily_max": float(truth_row["daily_max"]),
                    "winning_outcome": winning_outcome,
                    "available_models_json": json.dumps(selected_models),
                    "selected_models_json": json.dumps(selected_models),
                    "forecast_source_kind": self._forecast_source_kind(selected_forecasts),
                    **self._metadata_fields(source_priority=self._source_priority(self._forecast_source_kind(selected_forecasts))),
                }
                self._populate_feature_row(row, selected_forecasts, spec.target_local_date, spec.timezone, unit=spec.unit)
                rows.append(row)
                sequence_rows.extend(
                    self._build_sequence_rows(
                        selected_forecasts=selected_forecasts,
                        spec=spec,
                        horizon=horizon,
                        decision_point=decision_point,
                        issue_time=issue_time,
                        snapshot=snapshot,
                        winning_outcome=winning_outcome,
                        realized_daily_max=float(truth_row["daily_max"]),
                    )
                )

        frame = pd.DataFrame(rows)
        if frame.empty:
            msg = self._empty_materialization_message(snapshots, forecast_frame, truth_frame)
            raise ValueError(msg)
        if not frame.empty:
            numeric_columns = frame.select_dtypes(include=["number"]).columns
            frame.loc[:, numeric_columns] = frame.loc[:, numeric_columns].fillna(0.0)
        if contract in {"tabular", "both"} and not frame.empty:
            self.warehouse.write_gold_table(
                "gold_training_examples_tabular",
                frame,
                relative_path=f"gold/{output_name}.parquet",
            )
            self.warehouse.write_gold_table(
                "gold_training_examples",
                frame,
                relative_path=f"gold/{output_name}_compat.parquet",
            )
        if contract in {"sequence", "both"} and sequence_rows:
            sequence_frame = pd.DataFrame(sequence_rows)
            self.warehouse.write_gold_table(
                "gold_training_examples_sequence",
                sequence_frame,
                relative_path=f"gold/{output_name}_sequence.parquet",
            )
        self.warehouse.write_manifest()
        return frame

    def _empty_materialization_message(
        self,
        snapshots: list[MarketSnapshot],
        forecast_frame: pd.DataFrame,
        truth_frame: pd.DataFrame,
    ) -> str:
        """Build an actionable error when no gold rows can be materialized."""

        missing_inputs: list[str] = []
        if forecast_frame.empty or "market_id" not in forecast_frame.columns:
            missing_inputs.append("silver_forecast_runs_hourly")
        if truth_frame.empty or "market_id" not in truth_frame.columns:
            missing_inputs.append("silver_observations_daily")

        message = "No training rows materialized."
        if missing_inputs:
            message += f" Missing usable rows in: {', '.join(missing_inputs)}."
        if any(snapshot.spec is not None and snapshot.spec.adapter_key() == "wunderground" for snapshot in snapshots):
            message += (
                " Wunderground-family markets require a documented same-airport public truth response "
                "(for example, AMO AIR_CALP for Seoul/RKSI, the Wunderground public historical API for "
                "core Wunderground airport cities, or NOAA Global Hourly for station-catalog cities mapped there) "
                "or a local station snapshot for truth backfill."
            )
        truth_bronze = self.warehouse.read_table("bronze_truth_snapshots")
        if not truth_bronze.empty and "status" in truth_bronze.columns:
            market_ids = {snapshot.spec.market_id for snapshot in snapshots if snapshot.spec is not None}
            lagged = truth_bronze.loc[truth_bronze["status"].astype(str) == "lag"].copy()
            if market_ids and "market_id" in lagged.columns:
                lagged = lagged.loc[lagged["market_id"].astype(str).isin(market_ids)].copy()
            if not lagged.empty:
                details: list[str] = []
                for _, row in lagged.head(3).iterrows():
                    city = str(row.get("city") or row.get("market_id") or "unknown")
                    station_id = str(row.get("station_id") or "unknown")
                    latest = pd.to_datetime(row.get("latest_available_date"), errors="coerce")
                    if pd.notna(latest):
                        details.append(f"{city}/{station_id}: latest {latest.date().isoformat()}")
                    else:
                        details.append(f"{city}/{station_id}: archive lag")
                message += f" Public archive lag detected: {'; '.join(details)}."
                message += " Run `uv run pmtmax summarize-truth-coverage` for per-market lag details."
        return message

    @staticmethod
    def _truth_failure_details(exc: Exception) -> tuple[str, dt.date | None]:
        if isinstance(exc, TruthSourceLagError):
            return "lag", exc.latest_available_date
        return "error", None

    def _populate_feature_row(
        self,
        row: dict[str, object],
        market_forecasts: pd.DataFrame,
        target_date: dt.date,
        timezone: str,
        unit: str = "C",
    ) -> None:
        for model in self.models:
            subset = market_forecasts.loc[market_forecasts["model_name"] == model].copy()
            raw_features = self._features_from_hourly_rows(subset, target_date, timezone)
            features = _convert_celsius_features(raw_features, unit)
            for key, value in features.items():
                row[f"{model}_{key}"] = value
            if "model_daily_max" in features and "model_daily_max" not in row:
                row["model_daily_max"] = features["model_daily_max"]
        row["neighbor_mean_temp"] = float(
            _coerce_float(row.get("ecmwf_ifs025_model_daily_mean", row.get("model_daily_max", 0.0)))
        )
        row["neighbor_spread"] = abs(
            _coerce_float(row.get("ecmwf_ifs025_model_daily_max", 0.0))
            - _coerce_float(row.get("ecmwf_aifs025_single_model_daily_max", 0.0))
        )

    def _features_from_hourly_rows(
        self,
        frame: pd.DataFrame,
        target_date: dt.date,
        timezone: str,
    ) -> dict[str, float]:
        if frame.empty:
            return self._empty_feature_map()
        local_time = pd.to_datetime(frame["forecast_time_local"], utc=True).dt.tz_convert(timezone)
        payload_frame = pd.DataFrame(
            {
                "time": local_time,
                "temperature_2m": frame["temperature_2m"].astype(float),
                "dew_point_2m": frame["dew_point_2m"].astype(float),
                "relative_humidity_2m": frame["relative_humidity_2m"].astype(float),
                "wind_speed_10m": frame["wind_speed_10m"].astype(float),
                "cloud_cover": frame["cloud_cover"].astype(float),
            }
        )
        package = build_hourly_feature_frame({"hourly": payload_frame.to_dict(orient="list")})
        features = target_day_features(package, target_date)
        return features or self._empty_feature_map()

    def _build_sequence_rows(
        self,
        *,
        selected_forecasts: pd.DataFrame,
        spec: MarketSpec,
        horizon: str,
        decision_point: dict[str, object],
        issue_time: pd.Timestamp,
        snapshot: MarketSnapshot,
        winning_outcome: str,
        realized_daily_max: float,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for model_name in sorted(selected_forecasts["model_name"].dropna().astype(str).unique().tolist()):
            subset = (
                selected_forecasts.loc[selected_forecasts["model_name"].astype(str) == model_name]
                .sort_values("forecast_time_local")
                .reset_index(drop=True)
            )
            if subset.empty:
                continue
            rows.append(
                {
                    "market_id": spec.market_id,
                    "station_id": spec.station_id,
                    "city": spec.city,
                    "truth_track": spec.truth_track,
                    "settlement_eligible": spec.settlement_eligible,
                    "target_date": pd.Timestamp(spec.target_local_date),
                    "decision_horizon": horizon,
                    "decision_time_utc": pd.Timestamp(decision_point["decision_time_utc"]),
                    "issue_time_utc": issue_time,
                    "lead_hours": decision_point["lead_hours"],
                    "model_name": model_name,
                    "sequence_length": len(subset),
                    "time_index_json": json.dumps(subset["forecast_time_local"].astype(str).tolist()),
                    "temperature_2m_json": json.dumps(subset["temperature_2m"].astype(float).tolist()),
                    "dew_point_2m_json": json.dumps(subset["dew_point_2m"].astype(float).tolist()),
                    "relative_humidity_2m_json": json.dumps(
                        subset["relative_humidity_2m"].astype(float).tolist()
                    ),
                    "wind_speed_10m_json": json.dumps(subset["wind_speed_10m"].astype(float).tolist()),
                    "cloud_cover_json": json.dumps(subset["cloud_cover"].astype(float).tolist()),
                    "valid_mask_json": json.dumps([1] * len(subset)),
                    "market_spec_json": spec.model_dump_json(),
                    "market_prices_json": json.dumps(snapshot.outcome_prices, sort_keys=True),
                    "realized_daily_max": realized_daily_max,
                    "winning_outcome": winning_outcome,
                    "forecast_source_kind": self._forecast_source_kind(subset),
                    **self._metadata_fields(
                        source_priority=self._source_priority(self._forecast_source_kind(subset))
                    ),
                }
            )
        return rows

    @staticmethod
    def _select_forecasts_for_horizon(market_forecasts: pd.DataFrame, horizon: str) -> pd.DataFrame:
        if market_forecasts.empty or "decision_horizon" not in market_forecasts.columns:
            return market_forecasts
        single_run = market_forecasts.loc[
            (market_forecasts["endpoint_kind"].astype(str) == "single_run")
            & (market_forecasts["decision_horizon"].astype(str) == horizon)
        ].copy()
        if not single_run.empty and "temperature_2m" in single_run.columns and (single_run["temperature_2m"] != 0.0).any():
            return single_run
        historical = market_forecasts.loc[
            market_forecasts["endpoint_kind"].astype(str) == "historical_forecast"
        ].copy()
        if not historical.empty:
            return historical
        generic = market_forecasts.loc[market_forecasts["decision_horizon"].isna()].copy()
        return generic if not generic.empty else market_forecasts

    @staticmethod
    def _materialized_issue_time(selected_forecasts: pd.DataFrame, fallback_issue_time: object) -> pd.Timestamp:
        if "issue_time_utc" in selected_forecasts.columns:
            available = selected_forecasts["issue_time_utc"].dropna()
            if not available.empty:
                return pd.Timestamp(available.iloc[0])
        return pd.Timestamp(fallback_issue_time)

    @staticmethod
    def _forecast_source_kind(selected_forecasts: pd.DataFrame) -> str:
        if selected_forecasts.empty or "endpoint_kind" not in selected_forecasts.columns:
            return "unknown"
        return str(selected_forecasts["endpoint_kind"].dropna().astype(str).iloc[0])

    @staticmethod
    def _empty_feature_map() -> dict[str, float]:
        return {
            "model_daily_max": 0.0,
            "model_daily_mean": 0.0,
            "model_daily_min": 0.0,
            "diurnal_amplitude": 0.0,
            "midday_temp": 0.0,
            "cloud_cover_mean": 0.0,
            "wind_speed_mean": 0.0,
            "humidity_mean": 0.0,
            "dew_point_mean": 0.0,
            "num_hours": 0.0,
        }

    def _metadata_fields(
        self,
        *,
        created_at: dt.datetime | None = None,
        source_priority: int,
    ) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "data_version": self.data_version,
            "created_at": pd.Timestamp(created_at or dt.datetime.now(tz=dt.UTC)),
            "source_priority": source_priority,
        }

    @staticmethod
    def _source_priority(source_kind: str) -> int:
        priorities = {
            "fixture": 5,
            "forecast": 60,
            "historical_forecast": 80,
            "single_run": 90,
            "snapshot": 95,
            "build_dataset": 95,
            "scan": 95,
            "wunderground": 100,
            "hko": 100,
            "cwa": 100,
        }
        return priorities.get(source_kind, 50)

    def _forecast_endpoint_kind(self, spec: MarketSpec) -> str:
        local_now = dt.datetime.now(tz=ZoneInfo(spec.timezone)).date()
        return "forecast" if spec.target_local_date >= local_now else "historical_forecast"

    @staticmethod
    def _negative_cache_signatures(bronze_history: pd.DataFrame) -> set[str]:
        required = {"request_kind", "query_signature", "availability_status", "http_status"}
        if bronze_history.empty or not required.issubset(bronze_history.columns):
            return set()
        subset = bronze_history.loc[
            bronze_history["request_kind"].astype(str) == "probe"
        ].copy()
        if subset.empty:
            return set()
        negative = subset.loc[
            subset["http_status"].isin([400, 404, 422]) | subset["availability_status"].astype(str).isin(["unavailable"])
        ]
        return set(negative["query_signature"].dropna().astype(str).tolist())

    def _forecast_days(self, spec: MarketSpec, timezone: str) -> int:
        local_now = dt.datetime.now(tz=ZoneInfo(timezone)).date()
        return max((spec.target_local_date - local_now).days + 1, 1)

    def _probe_forecast_availability(
        self,
        *,
        spec: MarketSpec,
        model: str,
        latitude: float,
        longitude: float,
        timezone: str,
        endpoint_kind: str,
        fetched_at: dt.datetime,
    ) -> tuple[bool, dict[str, object]]:
        try:
            self._fetch_forecast_payload(
                spec=spec,
                model=model,
                latitude=latitude,
                longitude=longitude,
                timezone=timezone,
                endpoint_kind=endpoint_kind,
                hourly=PROBE_HOURLY,
            )
            return True, self._forecast_request_row(
                spec=spec,
                model=model,
                endpoint_kind=endpoint_kind,
                request_kind="probe",
                decision_horizon=None,
                requested_at=fetched_at,
                latitude=latitude,
                longitude=longitude,
                timezone=timezone,
                forecast_days=1,
                status="ok",
                availability_status="available",
                variables=PROBE_HOURLY,
                http_status=200,
                query_signature=self._forecast_query_signature(
                    spec=spec,
                    model=model,
                    endpoint_kind=endpoint_kind,
                    hourly=PROBE_HOURLY,
                    request_kind="probe",
                    decision_horizon=None,
                    latitude=latitude,
                    longitude=longitude,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            http_status, reason = self._error_details(exc)
            return False, self._forecast_request_row(
                spec=spec,
                model=model,
                endpoint_kind=endpoint_kind,
                request_kind="probe",
                decision_horizon=None,
                requested_at=fetched_at,
                latitude=latitude,
                longitude=longitude,
                timezone=timezone,
                forecast_days=1,
                status="error",
                availability_status=self._availability_status(http_status, reason),
                variables=PROBE_HOURLY,
                error_message=str(exc),
                http_status=http_status,
                reason=reason,
                query_signature=self._forecast_query_signature(
                    spec=spec,
                    model=model,
                    endpoint_kind=endpoint_kind,
                    hourly=PROBE_HOURLY,
                    request_kind="probe",
                    decision_horizon=None,
                    latitude=latitude,
                    longitude=longitude,
                ),
            )

    def _fetch_forecast_with_variable_fallback(
        self,
        *,
        spec: MarketSpec,
        model: str,
        latitude: float,
        longitude: float,
        timezone: str,
        endpoint_kind: str,
    ) -> tuple[dict[str, Any], list[str]]:
        last_error: Exception | None = None
        for hourly in VARIABLE_FALLBACKS:
            try:
                payload = self._fetch_forecast_payload(
                    spec=spec,
                    model=model,
                    latitude=latitude,
                    longitude=longitude,
                    timezone=timezone,
                    endpoint_kind=endpoint_kind,
                    hourly=hourly,
                )
                return payload, hourly
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                http_status, reason = self._error_details(exc)
                if http_status == 400 and self._availability_status(http_status, reason) != "unavailable":
                    raise
                if http_status not in {400, 404, 422}:
                    raise
        if last_error is None:
            msg = "Open-Meteo request failed without an exception"
            raise RuntimeError(msg)
        raise last_error

    def _backfill_single_run_requests(
        self,
        *,
        spec: MarketSpec,
        model: str,
        latitude: float,
        longitude: float,
        timezone: str,
        horizons: list[str],
        strict_archive: bool,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        bronze_rows: list[dict[str, object]] = []
        silver_rows: list[dict[str, object]] = []
        for horizon in horizons:
            decision_point = self._decision_point(spec, horizon)
            requested_run_time = pd.Timestamp(decision_point["issue_time_utc"])
            fetched_at = dt.datetime.now(tz=dt.UTC)
            try:
                payload, variables = self._fetch_single_run_with_variable_fallback(
                    spec=spec,
                    model=model,
                    latitude=latitude,
                    longitude=longitude,
                    timezone=timezone,
                    forecast_days=self._single_run_forecast_days(spec, requested_run_time),
                    run_time=requested_run_time.to_pydatetime(),
                )
                availability_status = "available"
                status = "ok"
                http_status: int | None = 200
                reason = None
                error_message = None
            except Exception as exc:  # noqa: BLE001
                http_status, reason = self._error_details(exc)
                availability_status = self._availability_status(http_status, reason)
                status = "error"
                error_message = str(exc)
                bronze_rows.append(
                    self._forecast_request_row(
                        spec=spec,
                        model=model,
                        endpoint_kind="single_run",
                        request_kind="single_run",
                        decision_horizon=horizon,
                        requested_at=fetched_at,
                        latitude=latitude,
                        longitude=longitude,
                        timezone=timezone,
                        forecast_days=self._single_run_forecast_days(spec, requested_run_time),
                        status=status,
                        availability_status=availability_status,
                        variables=DEFAULT_HOURLY,
                        error_message=error_message,
                        http_status=http_status,
                        reason=reason,
                        query_signature=self._forecast_query_signature(
                            spec=spec,
                            model=model,
                            endpoint_kind="single_run",
                            hourly=DEFAULT_HOURLY,
                            request_kind="single_run",
                            decision_horizon=horizon,
                            latitude=latitude,
                            longitude=longitude,
                        ),
                    )
                )
                continue

            relative_path = (
                f"forecasts/single_run/{spec.city.lower().replace(' ', '_')}/"
                f"{spec.target_local_date.isoformat()}/{spec.market_id}_{model}_{horizon}.json"
            )
            artifact = self.warehouse.raw_store.write_json(relative_path, payload)
            bronze_rows.append(
                self._forecast_request_row(
                    spec=spec,
                    model=model,
                    endpoint_kind="single_run",
                    request_kind="single_run",
                    decision_horizon=horizon,
                    requested_at=fetched_at,
                    latitude=latitude,
                    longitude=longitude,
                    timezone=timezone,
                    forecast_days=self._single_run_forecast_days(spec, requested_run_time),
                    status=status,
                    availability_status=availability_status,
                    variables=variables,
                    raw_path=artifact.relative_path,
                    raw_hash=artifact.content_hash,
                    http_status=http_status,
                    reason=reason,
                    query_signature=self._forecast_query_signature(
                        spec=spec,
                        model=model,
                        endpoint_kind="single_run",
                        hourly=variables,
                        request_kind="single_run",
                        decision_horizon=horizon,
                        latitude=latitude,
                        longitude=longitude,
                    ),
                )
            )
            silver_rows.extend(
                self._normalize_forecast_rows(
                    spec=spec,
                    model_name=model,
                    endpoint_kind="single_run",
                    payload=cast(dict[str, Any], payload),
                    raw_path=artifact.relative_path,
                    raw_hash=artifact.content_hash,
                    availability_status=availability_status,
                    source_variables=variables,
                    retrieved_at=fetched_at,
                    decision_horizon=horizon,
                    issue_time_utc=requested_run_time.to_pydatetime(),
                    requested_run_time_utc=requested_run_time.to_pydatetime(),
                )
            )
        return bronze_rows, silver_rows

    def _forecast_request_row(
        self,
        *,
        spec: MarketSpec,
        model: str,
        endpoint_kind: str,
        request_kind: str,
        requested_at: dt.datetime,
        latitude: float,
        longitude: float,
        timezone: str,
        forecast_days: int,
        status: str,
        availability_status: str,
        variables: list[str],
        decision_horizon: str | None = None,
        raw_path: str | None = None,
        raw_hash: str | None = None,
        error_message: str | None = None,
        http_status: int | None = None,
        reason: str | None = None,
        query_signature: str | None = None,
    ) -> dict[str, object]:
        signature = query_signature or self._forecast_query_signature(
            spec=spec,
            model=model,
            endpoint_kind=endpoint_kind,
            hourly=variables,
            request_kind=request_kind,
            decision_horizon=decision_horizon,
            latitude=latitude,
            longitude=longitude,
        )
        return {
            "market_id": spec.market_id,
            "station_id": spec.station_id,
            "city": spec.city,
            "model_name": model,
            "endpoint_kind": endpoint_kind,
            "request_kind": request_kind,
            "decision_horizon": decision_horizon,
            "target_local_date": pd.Timestamp(spec.target_local_date),
            "timezone": timezone,
            "requested_at": pd.Timestamp(requested_at),
            "status": status,
            "availability_status": availability_status,
            "raw_path": raw_path,
            "raw_hash": raw_hash,
            "error_message": error_message,
            "requested_start_date": spec.target_local_date.isoformat(),
            "requested_end_date": spec.target_local_date.isoformat(),
            "forecast_days": forecast_days,
            "latitude": latitude,
            "longitude": longitude,
            "variables_json": json.dumps(variables),
            "http_status": http_status,
            "reason": reason,
            "query_signature": signature,
            "endpoint_url": self._forecast_endpoint_url(
                spec=spec,
                model=model,
                endpoint_kind=endpoint_kind,
                hourly=variables,
                forecast_days=forecast_days,
                request_kind=request_kind,
                latitude=latitude,
                longitude=longitude,
                decision_horizon=decision_horizon,
            ),
            "retrieved_at": pd.Timestamp(requested_at),
            **self._metadata_fields(created_at=requested_at, source_priority=self._source_priority(endpoint_kind)),
        }

    def _forecast_query_signature(
        self,
        *,
        spec: MarketSpec,
        model: str,
        endpoint_kind: str,
        hourly: list[str],
        request_kind: str,
        latitude: float,
        longitude: float,
        decision_horizon: str | None = None,
    ) -> str:
        payload = {
            "market_id": spec.market_id,
            "model": model,
            "endpoint_kind": endpoint_kind,
            "request_kind": request_kind,
            "decision_horizon": decision_horizon,
            "hourly": hourly,
            "target_date": spec.target_local_date.isoformat(),
            "latitude": latitude,
            "longitude": longitude,
            "timezone": spec.timezone,
        }
        return stable_hash(json.dumps(payload, sort_keys=True))

    def _forecast_endpoint_url(
        self,
        *,
        spec: MarketSpec,
        model: str,
        endpoint_kind: str,
        hourly: list[str],
        forecast_days: int,
        request_kind: str,
        latitude: float,
        longitude: float,
        decision_horizon: str | None = None,
    ) -> str:
        if request_kind == "single_run":
            single_runs_base_url = getattr(self.openmeteo, "single_runs_base_url", "https://single-runs-api.open-meteo.com")
            base_url = f"{single_runs_base_url}/v1/forecast"
            single_run_params: dict[str, str | int | float | bool | None] = {
                "latitude": latitude,
                "longitude": longitude,
                "models": model,
                "hourly": ",".join(hourly),
                "run": self._single_run_time(spec, decision_horizon or "morning_of"),
                "forecast_days": forecast_days,
                "timezone": spec.timezone,
            }
            return f"{base_url}?{httpx.QueryParams(single_run_params)}"
        if endpoint_kind == "historical_forecast":
            archive_base_url = getattr(self.openmeteo, "archive_base_url", "https://historical-forecast-api.open-meteo.com")
            base_url = f"{archive_base_url}/v1/forecast"
            historical_params: dict[str, str | int | float | bool | None] = {
                "latitude": latitude,
                "longitude": longitude,
                "models": model,
                "hourly": ",".join(hourly),
                "start_date": spec.target_local_date.isoformat(),
                "end_date": spec.target_local_date.isoformat(),
                "timezone": spec.timezone,
            }
        elif endpoint_kind == "fixture":
            return "fixture://bundled-openmeteo"
        else:
            forecast_base_url = getattr(self.openmeteo, "base_url", "https://api.open-meteo.com")
            base_url = f"{forecast_base_url}/v1/forecast"
            forecast_params: dict[str, str | int | float | bool | None] = {
                "latitude": latitude,
                "longitude": longitude,
                "models": model,
                "hourly": ",".join(hourly),
                "forecast_days": forecast_days,
                "timezone": spec.timezone,
            }
            return f"{base_url}?{httpx.QueryParams(forecast_params)}"
        return f"{base_url}?{httpx.QueryParams(historical_params)}"

    @staticmethod
    def _error_details(exc: Exception) -> tuple[int | None, str | None]:
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                payload = exc.response.json()
                reason = payload.get("reason")
            except Exception:  # noqa: BLE001
                reason = exc.response.text
            return exc.response.status_code, str(reason) if reason is not None else None
        return None, None

    @staticmethod
    def _availability_status(http_status: int | None, reason: str | None) -> str:
        if http_status in {404, 422}:
            return "unavailable"
        if http_status == 400 and reason and "no data is available for this location" in reason.lower():
            return "unavailable"
        return "error"

    @staticmethod
    def _single_run_time(spec: MarketSpec, horizon: str) -> str:
        decision_point = BackfillPipeline._decision_point(spec, horizon)
        issue_time = cast(dt.datetime, decision_point["issue_time_utc"])
        return issue_time.strftime("%Y-%m-%dT%H:%M")

    @staticmethod
    def _single_run_forecast_days(spec: MarketSpec, requested_run_time: pd.Timestamp) -> int:
        run_time = requested_run_time.to_pydatetime()
        if run_time.tzinfo is None:
            local_run_date = run_time.date()
        else:
            local_run_date = run_time.astimezone(ZoneInfo(spec.timezone)).date()
        return int(max((spec.target_local_date - local_run_date).days + 1, 1))

    def _fetch_forecast_payload(
        self,
        *,
        spec: MarketSpec,
        model: str,
        latitude: float,
        longitude: float,
        timezone: str,
        endpoint_kind: str,
        hourly: list[str],
    ) -> dict[str, Any]:
        if endpoint_kind == "forecast":
            return self.openmeteo.forecast(
                latitude=latitude,
                longitude=longitude,
                model=model,
                hourly=hourly,
                forecast_days=self._forecast_days(spec, timezone),
                timezone=timezone,
            )
        return self.openmeteo.historical_forecast(
            latitude=latitude,
            longitude=longitude,
            model=model,
            hourly=hourly,
            start_date=spec.target_local_date.isoformat(),
            end_date=spec.target_local_date.isoformat(),
            timezone=timezone,
        )

    def _fetch_single_run_with_variable_fallback(
        self,
        *,
        spec: MarketSpec,
        model: str,
        latitude: float,
        longitude: float,
        timezone: str,
        forecast_days: int,
        run_time: dt.datetime,
    ) -> tuple[dict[str, Any], list[str]]:
        last_error: Exception | None = None
        run_string = pd.Timestamp(run_time).strftime("%Y-%m-%dT%H:%M")
        for hourly in VARIABLE_FALLBACKS:
            try:
                payload = self.openmeteo.single_run(
                    latitude=latitude,
                    longitude=longitude,
                    model=model,
                    hourly=hourly,
                    run=run_string,
                    forecast_days=forecast_days,
                    timezone=timezone,
                )
                return payload, hourly
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                http_status, reason = self._error_details(exc)
                if http_status == 400 and self._availability_status(http_status, reason) != "unavailable":
                    raise
                if http_status not in {400, 404, 422, 500}:
                    raise
        if last_error is None:
            msg = "Open-Meteo single-run request failed without an exception"
            raise RuntimeError(msg)
        raise last_error

    def _load_forecast_fixture(self, city: str) -> dict[str, Any] | None:
        if self.forecast_fixture_dir is None:
            return None
        fixture_name = city.lower().replace(" ", "_") + "_daily.json"
        fixture_path = self.forecast_fixture_dir / fixture_name
        if not fixture_path.exists():
            return None
        return cast(dict[str, Any], json.loads(fixture_path.read_text()))

    def _normalize_forecast_rows(
        self,
        *,
        spec: MarketSpec,
        model_name: str,
        endpoint_kind: str,
        payload: dict[str, Any],
        raw_path: str,
        raw_hash: str,
        availability_status: str,
        source_variables: list[str],
        retrieved_at: dt.datetime,
        decision_horizon: str | None = None,
        issue_time_utc: dt.datetime | None = None,
        requested_run_time_utc: dt.datetime | None = None,
    ) -> list[dict[str, object]]:
        package = build_hourly_feature_frame(payload)
        frame: pd.DataFrame = package["frame"]
        if frame.empty:
            return []
        local_time = pd.to_datetime(frame["time"])
        if getattr(local_time.dt, "tz", None) is None:
            local_time = local_time.dt.tz_localize(
                spec.timezone,
                ambiguous="infer",
                nonexistent="shift_forward",
            )
        utc_time = local_time.dt.tz_convert("UTC")
        rows: list[dict[str, object]] = []
        for idx, local_timestamp in enumerate(local_time):
            rows.append(
                {
                    "market_id": spec.market_id,
                    "station_id": spec.station_id,
                    "city": spec.city,
                    "target_local_date": pd.Timestamp(spec.target_local_date),
                    "timezone": spec.timezone,
                    "provider": "open-meteo",
                    "model_name": model_name,
                    "endpoint_kind": endpoint_kind,
                    "decision_horizon": decision_horizon,
                    "availability_status": availability_status,
                    "issue_time_utc": pd.Timestamp(issue_time_utc) if issue_time_utc is not None else pd.NaT,
                    "requested_run_time_utc": pd.Timestamp(requested_run_time_utc)
                    if requested_run_time_utc is not None
                    else pd.NaT,
                    "retrieved_at_utc": pd.Timestamp(retrieved_at),
                    "forecast_time_local": local_timestamp.isoformat(),
                    "forecast_time_utc": utc_time.iloc[idx].isoformat(),
                    "local_date": pd.Timestamp(local_timestamp.date()),
                    "hour": int(local_timestamp.hour),
                    "source_variables_json": json.dumps(source_variables),
                    "temperature_2m": _coerce_float(frame.iloc[idx].get("temperature_2m", 0.0)),
                    "dew_point_2m": _coerce_float(frame.iloc[idx].get("dew_point_2m", 0.0)),
                    "relative_humidity_2m": _coerce_float(frame.iloc[idx].get("relative_humidity_2m", 0.0)),
                    "wind_speed_10m": _coerce_float(frame.iloc[idx].get("wind_speed_10m", 0.0)),
                    "cloud_cover": _coerce_float(frame.iloc[idx].get("cloud_cover", 0.0)),
                    "raw_path": raw_path,
                    "raw_hash": raw_hash,
                    **self._metadata_fields(
                        created_at=retrieved_at,
                        source_priority=self._source_priority(endpoint_kind),
                    ),
                }
            )
        return rows

    @staticmethod
    def _decision_point(spec: MarketSpec, horizon: str) -> dict[str, object]:
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
        return {
            "decision_time_utc": decision_utc,
            "issue_time_utc": issue_utc,
            "lead_hours": lead_hours,
        }

    @staticmethod
    def _station_coords(spec: MarketSpec) -> tuple[float, float, str]:
        if spec.station_lat is not None and spec.station_lon is not None:
            return spec.station_lat, spec.station_lon, spec.timezone
        if definition := lookup_station(spec.city):
            return definition.lat, definition.lon, definition.timezone
        msg = f"Missing station coordinates for {spec.city}"
        raise ValueError(msg)

    def summarize_forecast_availability(
        self,
        *,
        top_k: int = 3,
    ) -> dict[str, pd.DataFrame]:
        """Summarize forecast archive coverage and top available models."""

        bronze = self.warehouse.read_table("bronze_forecast_requests")
        if bronze.empty:
            return {"summary": pd.DataFrame(), "recommended": pd.DataFrame()}

        frame = bronze.copy()
        frame["decision_horizon"] = frame["decision_horizon"].fillna("generic")
        frame["availability_status"] = frame["availability_status"].fillna("unknown").astype(str)
        frame["is_available"] = frame["availability_status"].isin(["available", "demo_fixture"]).astype(int)
        frame["is_unavailable"] = frame["availability_status"].eq("unavailable").astype(int)
        frame["is_error"] = frame["availability_status"].eq("error").astype(int)

        summary = (
            frame.groupby(
                ["city", "model_name", "endpoint_kind", "request_kind", "decision_horizon"],
                dropna=False,
            )
            .agg(
                requests=("market_id", "count"),
                available_requests=("is_available", "sum"),
                unavailable_requests=("is_unavailable", "sum"),
                error_requests=("is_error", "sum"),
                last_requested_at=("requested_at", "max"),
            )
            .reset_index()
        )
        summary["availability_ratio"] = summary["available_requests"] / summary["requests"].clip(lower=1)

        recommended = summary.loc[summary["request_kind"].isin(["full", "single_run"])].copy()
        recommended = recommended.loc[recommended["available_requests"] > 0].copy()
        recommended = recommended.sort_values(
            by=["city", "decision_horizon", "availability_ratio", "available_requests", "model_name"],
            ascending=[True, True, False, False, True],
        )
        recommended = recommended.groupby(["city", "decision_horizon"], dropna=False).head(top_k).reset_index(drop=True)

        return {"summary": summary, "recommended": recommended}
