"""Public airport truth adapter backed by NOAA Global Hourly."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from pmtmax.http import CachedHttpClient
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.storage.schemas import ObservationRecord
from pmtmax.weather.truth_sources.base import (
    TruthSourceLagError,
    TruthFetchBundle,
    TruthSource,
    celsius_to_fahrenheit,
)

NOAA_ACCESS_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
NOAA_SEARCH_URL = "https://www.ncei.noaa.gov/access/services/search/v1/data"


class NoaaGlobalHourlyTruthSource(TruthSource):
    """Compute local-date daily maxima from NOAA hourly station observations."""

    def __init__(self, http: CachedHttpClient, snapshot_dir: Path | None = None) -> None:
        self.http = http
        self.snapshot_dir = snapshot_dir

    def fetch_observation_bundle(self, spec: MarketSpec, target_date: date) -> TruthFetchBundle:
        payload = self._load_snapshot(spec, target_date)
        params = self._query_params(spec, target_date)
        source_url = f"{NOAA_ACCESS_URL}?{urlencode(params)}"
        if payload is None:
            payload = cast(list[dict[str, Any]], self.http.get_json(NOAA_ACCESS_URL, params=params, use_cache=True))
        hourly_values = self._local_day_values(payload, spec, target_date)
        if not hourly_values:
            coverage = self.coverage_status(spec, target_date)
            latest_available_date = coverage.get("latest_available_date")
            if coverage["status"] == "lag" and isinstance(latest_available_date, date):
                msg = (
                    f"No NOAA Global Hourly rows for {spec.station_id} on {target_date.isoformat()}; "
                    f"latest available date for public station {coverage['query_station_id']} "
                    f"is {latest_available_date.isoformat()}"
                )
                raise TruthSourceLagError(msg, latest_available_date=latest_available_date)
            msg = f"No NOAA Global Hourly rows for {spec.station_id} on {target_date.isoformat()}"
            raise ValueError(msg)
        if spec.unit == "F":
            converted = [celsius_to_fahrenheit(value) for value in hourly_values]
        else:
            converted = hourly_values
        observation = ObservationRecord(
            source="NOAA Global Hourly",
            station_id=spec.station_id,
            local_date=target_date,
            hourly_temp=converted,
            daily_max=max(converted),
            unit=spec.unit,
            finalized_at=datetime.now(tz=UTC),
        )
        return TruthFetchBundle(
            observation=observation,
            raw_payload=payload,
            media_type="application/json",
            source_url=source_url,
        )

    def _load_snapshot(self, spec: MarketSpec, target_date: date) -> list[dict[str, Any]] | None:
        if self.snapshot_dir is None:
            return None
        candidates = [
            self.snapshot_dir / f"{spec.station_id}_{target_date:%Y%m%d}.json",
        ]
        if spec.public_truth_station_id:
            candidates.append(self.snapshot_dir / f"{spec.public_truth_station_id}_{target_date:%Y%m%d}.json")
        for candidate in candidates:
            if candidate.exists():
                return cast(list[dict[str, Any]], json.loads(candidate.read_text()))
        return None

    def coverage_status(self, spec: MarketSpec, target_date: date) -> dict[str, object]:
        """Return whether the public archive is likely ready for the target date."""

        query_station_id = spec.public_truth_station_id or spec.station_id
        if self._load_snapshot(spec, target_date) is not None:
            return {
                "status": "local_snapshot",
                "query_station_id": query_station_id,
                "latest_available_date": target_date,
            }
        latest_available_date = self.latest_available_date(spec)
        if latest_available_date is None:
            return {
                "status": "unknown",
                "query_station_id": query_station_id,
                "latest_available_date": None,
            }
        if latest_available_date >= target_date:
            return {
                "status": "ready",
                "query_station_id": query_station_id,
                "latest_available_date": latest_available_date,
            }
        return {
            "status": "lag",
            "query_station_id": query_station_id,
            "latest_available_date": latest_available_date,
        }

    def latest_available_date(self, spec: MarketSpec) -> date | None:
        """Return the latest UTC day advertised by NOAA for the queried station."""

        payload = self.http.get_json(NOAA_SEARCH_URL, params=self._search_params(spec), use_cache=True)
        if not isinstance(payload, dict):
            return None
        end_date = payload.get("endDate")
        if not isinstance(end_date, str) or not end_date:
            return None
        try:
            return datetime.fromisoformat(end_date.replace("Z", "+00:00")).date()
        except ValueError:
            return None

    @staticmethod
    def _query_params(spec: MarketSpec, target_date: date) -> dict[str, str]:
        station_id = spec.public_truth_station_id or spec.station_id
        start_date = target_date - timedelta(days=1)
        end_date = target_date + timedelta(days=1)
        return {
            "dataset": "global-hourly",
            "stations": station_id,
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "format": "json",
            "includeAttributes": "false",
        }

    @staticmethod
    def _search_params(spec: MarketSpec) -> dict[str, str | int]:
        return {
            "dataset": "global-hourly",
            "stations": spec.public_truth_station_id or spec.station_id,
            "limit": 1,
        }

    @staticmethod
    def _parse_tmp(raw_value: Any) -> float | None:
        if not isinstance(raw_value, str):
            return None
        value = raw_value.split(",", 1)[0].strip()
        if not value or value in {"+9999", "-9999", "9999"}:
            return None
        return int(value) / 10.0

    def _local_day_values(
        self,
        payload: list[dict[str, Any]],
        spec: MarketSpec,
        target_date: date,
    ) -> list[float]:
        timezone = ZoneInfo(spec.timezone)
        values: list[float] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            timestamp = row.get("DATE")
            if not isinstance(timestamp, str):
                continue
            try:
                observed_utc = datetime.fromisoformat(timestamp).replace(tzinfo=UTC)
            except ValueError:
                continue
            if observed_utc.astimezone(timezone).date() != target_date:
                continue
            parsed = self._parse_tmp(row.get("TMP"))
            if parsed is not None:
                values.append(parsed)
        return values
