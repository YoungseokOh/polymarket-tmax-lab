"""Wunderground truth adapter."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast
from urllib.parse import urlencode, urlparse

from pmtmax.config.settings import EnvSettings
from pmtmax.http import CachedHttpClient
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.storage.schemas import ObservationRecord
from pmtmax.weather.truth_sources.base import TruthFetchBundle, TruthSource

TEMP_MAX_PATTERNS = [
    re.compile(r'"temperatureMax"\s*:\s*\{"value"\s*:\s*(?P<value>-?\d+(?:\.\d+)?)', re.I),
    re.compile(r'"temperatureMax"\s*:\s*(?P<value>-?\d+(?:\.\d+)?)', re.I),
]
WU_HISTORICAL_URL = "https://api.weather.com/v1/location/{location_id}/observations/historical.json"
WU_API_KEY_ENV = "PMTMAX_WU_API_KEY"


class WundergroundTruthSource(TruthSource):
    """Retrieve daily max temperature from Wunderground history pages."""

    def __init__(
        self,
        http: CachedHttpClient,
        snapshot_dir: Path | None = None,
        api_key: str | None = None,
    ) -> None:
        self.http = http
        self.snapshot_dir = snapshot_dir
        self.api_key = EnvSettings().wu_api_key if api_key is None else api_key

    def fetch_observation_bundle(self, spec: MarketSpec, target_date: date) -> TruthFetchBundle:
        url = f"{spec.official_source_url.rstrip('/')}/date/{target_date.isoformat()}"
        html = self._load_snapshot(spec.station_id, target_date)
        if html is not None:
            value = self._parse_daily_max(html)
            observation = ObservationRecord(
                source="Wunderground",
                station_id=spec.station_id,
                local_date=target_date,
                daily_max=value,
                unit=spec.unit,
                finalized_at=datetime.now(tz=UTC),
            )
            return TruthFetchBundle(
                observation=observation,
                raw_payload=html,
                media_type="text/html",
                source_url=url,
            )

        try:
            payload, source_url = self._fetch_historical_payload(spec, target_date)
            value = self._parse_historical_max(payload)
            raw_payload: dict[str, object] | str = payload
            media_type = "application/json"
        except Exception:
            html = self.http.get_text(url, use_cache=True)
            value = self._parse_daily_max(html)
            raw_payload = html
            media_type = "text/html"
            source_url = url

        observation = ObservationRecord(
            source="Wunderground",
            station_id=spec.station_id,
            local_date=target_date,
            daily_max=value,
            unit=spec.unit,
            finalized_at=datetime.now(tz=UTC),
        )
        return TruthFetchBundle(
            observation=observation,
            raw_payload=raw_payload,
            media_type=media_type,
            source_url=source_url,
        )

    def _load_snapshot(self, station_id: str, target_date: date) -> str | None:
        if self.snapshot_dir is None:
            return None
        candidate = self.snapshot_dir / f"{station_id}_{target_date:%Y%m%d}.html"
        if candidate.exists():
            return candidate.read_text()
        return None

    def _fetch_historical_payload(self, spec: MarketSpec, target_date: date) -> tuple[dict[str, object], str]:
        if not self.api_key:
            msg = f"Missing {WU_API_KEY_ENV} for Weather.com historical API"
            raise RuntimeError(msg)
        location_id = self._location_id(spec)
        base_url = WU_HISTORICAL_URL.format(location_id=location_id)
        params = {
            "apiKey": self.api_key,
            "units": "m" if spec.unit == "C" else "e",
            "startDate": target_date.strftime("%Y%m%d"),
            "endDate": target_date.strftime("%Y%m%d"),
        }
        payload = cast(dict[str, object], self.http.get_json(base_url, params=params, use_cache=True))
        source_url = f"{base_url}?{urlencode(params)}"
        return payload, source_url

    @staticmethod
    def _location_id(spec: MarketSpec) -> str:
        parts = [part for part in urlparse(spec.official_source_url).path.split("/") if part]
        if len(parts) < 4 or parts[0] != "history" or parts[1] != "daily":
            msg = f"Unsupported Wunderground source URL: {spec.official_source_url}"
            raise ValueError(msg)
        country_code = parts[2].upper()
        return f"{spec.station_id}:9:{country_code}"

    @staticmethod
    def _parse_daily_max(html: str) -> float:
        for pattern in TEMP_MAX_PATTERNS:
            match = pattern.search(html)
            if match:
                return float(match.group("value"))
        msg = "Could not parse Wunderground daily max"
        raise ValueError(msg)

    @staticmethod
    def _parse_historical_max(payload: dict[str, object]) -> float:
        observations = payload.get("observations")
        if not isinstance(observations, list) or not observations:
            msg = "No Wunderground historical observations found"
            raise ValueError(msg)

        values: list[float] = []
        for row in observations:
            if not isinstance(row, dict):
                continue
            value = row.get("temp")
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, str):
                try:
                    values.append(float(value))
                except ValueError:
                    continue

        if not values:
            msg = "No Wunderground historical temperature values found"
            raise ValueError(msg)
        return max(values)
