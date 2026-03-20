"""Aviation Meteorological Office AIR_CALP truth adapter for RKSI daily extremes."""

from __future__ import annotations

import csv
import io
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

from pmtmax.http import CachedHttpClient
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.storage.schemas import ObservationRecord
from pmtmax.weather.truth_sources.base import (
    TruthSourceLagError,
    TruthFetchBundle,
    TruthSource,
    celsius_to_fahrenheit,
)

AMO_AIR_CALP_URL = "http://amoapi.kma.go.kr/amoApi/air_calp"


class AmoAirCalpTruthSource(TruthSource):
    """Fetch daily extreme temperatures from the AMO AIR_CALP CSV feed."""

    def __init__(self, http: CachedHttpClient, snapshot_dir: Path | None = None) -> None:
        self.http = http
        self.snapshot_dir = snapshot_dir

    def fetch_observation_bundle(self, spec: MarketSpec, target_date: date) -> TruthFetchBundle:
        params = self._query_params(spec, target_date)
        source_url = f"{AMO_AIR_CALP_URL}?{urlencode(params)}"
        payload = self._load_snapshot(spec, target_date)
        if payload is None:
            payload = self.http.get_text(AMO_AIR_CALP_URL, params=params, use_cache=True)
        rows = self._parse_csv(payload)
        row = self._row_for_date(rows, target_date)
        if row is None:
            latest = self._latest_available_date(rows)
            if latest is None or latest < target_date:
                previous_month = target_date.replace(day=1) - timedelta(days=1)
                previous_payload = self._load_snapshot(spec, previous_month)
                if previous_payload is None:
                    previous_payload = self.http.get_text(
                        AMO_AIR_CALP_URL,
                        params=self._query_params(spec, previous_month),
                        use_cache=True,
                    )
                previous_latest = self._latest_available_date(self._parse_csv(previous_payload))
                if previous_latest is not None and (latest is None or previous_latest > latest):
                    latest = previous_latest
            if latest is not None and latest < target_date:
                msg = (
                    f"No AMO AIR_CALP row for {spec.station_id} on {target_date.isoformat()}; "
                    f"latest available date for public station {params['icao']} is {latest.isoformat()}"
                )
                raise TruthSourceLagError(msg, latest_available_date=latest)
            msg = f"No AMO AIR_CALP row for {spec.station_id} on {target_date.isoformat()}"
            raise ValueError(msg)

        value_c = self._parse_tenths_celsius(row.get("TMP_MAX"))
        if value_c is None:
            msg = f"AMO AIR_CALP did not contain TMP_MAX for {spec.station_id} on {target_date.isoformat()}"
            raise ValueError(msg)
        value = celsius_to_fahrenheit(value_c) if spec.unit == "F" else value_c
        observation = ObservationRecord(
            source="Aviation Meteorological Office AIR_CALP",
            station_id=spec.station_id,
            local_date=target_date,
            hourly_temp=[],
            daily_max=value,
            unit=spec.unit,
            finalized_at=datetime.now(tz=UTC),
        )
        return TruthFetchBundle(
            observation=observation,
            raw_payload=payload,
            media_type="text/csv",
            source_url=source_url,
        )

    def _load_snapshot(self, spec: MarketSpec, target_date: date) -> str | None:
        if self.snapshot_dir is None:
            return None
        for station_id in {spec.station_id, spec.public_truth_station_id or spec.station_id}:
            candidate = self.snapshot_dir / f"{station_id}_{target_date:%Y%m}.csv"
            if candidate.exists():
                return candidate.read_text()
        return None

    @staticmethod
    def _query_params(spec: MarketSpec, target_date: date) -> dict[str, str]:
        return {
            "icao": spec.public_truth_station_id or spec.station_id,
            "yyyymm": target_date.strftime("%Y%m"),
        }

    @staticmethod
    def _parse_csv(payload: str) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(payload)))

    @staticmethod
    def _row_for_date(rows: list[dict[str, str]], target_date: date) -> dict[str, str] | None:
        target = target_date.strftime("%Y%m%d")
        for row in rows:
            if row.get("TM") == target:
                return row
        return None

    @staticmethod
    def _latest_available_date(rows: list[dict[str, str]]) -> date | None:
        dates: list[date] = []
        for row in rows:
            value = row.get("TM", "")
            try:
                dates.append(datetime.strptime(value, "%Y%m%d").date())
            except ValueError:
                continue
        return max(dates) if dates else None

    @staticmethod
    def _parse_tenths_celsius(raw_value: str | None) -> float | None:
        if raw_value is None:
            return None
        value = raw_value.strip().strip('"')
        if not value:
            return None
        return int(value) / 10.0
