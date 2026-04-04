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
from pmtmax.weather.truth_sources.base import (
    TruthFetchBundle,
    TruthSource,
    TruthSourceLagError,
    TruthSourceParseError,
)

TEMP_MAX_PATTERNS = [
    re.compile(r'"temperatureMax"\s*:\s*\{"value"\s*:\s*(?P<value>-?\d+(?:\.\d+)?)', re.I),
    re.compile(r'"temperatureMax"\s*:\s*(?P<value>-?\d+(?:\.\d+)?)', re.I),
]
PUBLIC_API_KEY_PATTERNS = [
    re.compile(r'apiKey=(?P<value>[a-f0-9]{32})', re.I),
    re.compile(r'"apiKey"\s*:\s*"(?P<value>[a-f0-9]{32})"', re.I),
]
WU_HISTORICAL_URL = "https://api.weather.com/v1/location/{location_id}/observations/historical.json"
WU_API_KEY_ENV = "PMTMAX_WU_API_KEY"
NO_DATA_RECORDED_MARKERS = ("No Data Recorded",)
COUNTRY_CODE_BY_NAME = {
    "argentina": "AR",
    "brazil": "BR",
    "canada": "CA",
    "china": "CN",
    "france": "FR",
    "germany": "DE",
    "hong kong": "HK",
    "india": "IN",
    "israel": "IL",
    "italy": "IT",
    "japan": "JP",
    "new zealand": "NZ",
    "poland": "PL",
    "singapore": "SG",
    "south korea": "KR",
    "spain": "ES",
    "taiwan": "TW",
    "turkey": "TR",
    "uk": "GB",
    "united kingdom": "GB",
    "usa": "US",
    "united states": "US",
}


class WundergroundTruthSource(TruthSource):
    """Retrieve daily max temperature from Wunderground history pages."""

    def __init__(
        self,
        http: CachedHttpClient,
        snapshot_dir: Path | None = None,
        api_key: str | None = None,
        use_cache: bool = True,
    ) -> None:
        self.http = http
        self.snapshot_dir = snapshot_dir
        resolved_api_key = EnvSettings().wu_api_key if api_key is None else api_key
        self.api_key = resolved_api_key or None
        self._public_api_key_cache: dict[str, str] = {}
        self.use_cache = use_cache

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

        historical_error: Exception | None = None
        runtime_api_key = self.api_key
        if not runtime_api_key:
            runtime_api_key = self._public_api_key_cache.get(spec.station_id)
            if not runtime_api_key:
                html = self.http.get_text(url, use_cache=self.use_cache)
                runtime_api_key = self._extract_public_api_key(html)
                if runtime_api_key is not None:
                    self._public_api_key_cache[spec.station_id] = runtime_api_key
        try:
            payload, source_url = self._fetch_historical_payload(spec, target_date, api_key=runtime_api_key)
            value = self._parse_historical_max(payload)
            raw_payload: dict[str, object] | str = payload
            media_type = "application/json"
        except Exception as exc:
            historical_error = exc
            if html is None:
                html = self.http.get_text(url, use_cache=self.use_cache)
            try:
                value = self._parse_daily_max(html)
            except ValueError as parse_exc:
                if not runtime_api_key:
                    msg = (
                        f"{parse_exc}. Set {WU_API_KEY_ENV} for Weather.com historical API access "
                        "or provide a same-source local snapshot."
                    )
                    raise RuntimeError(msg) from parse_exc
                if historical_error is not None:
                    raise historical_error from parse_exc
                raise
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

    def _fetch_historical_payload(
        self,
        spec: MarketSpec,
        target_date: date,
        *,
        api_key: str | None = None,
    ) -> tuple[dict[str, object], str]:
        runtime_api_key = api_key or self.api_key
        if not runtime_api_key:
            msg = f"Missing {WU_API_KEY_ENV} for Weather.com historical API"
            raise RuntimeError(msg)
        location_id = self._location_id(spec)
        base_url = WU_HISTORICAL_URL.format(location_id=location_id)
        params = {
            "apiKey": runtime_api_key,
            "units": "m" if spec.unit == "C" else "e",
            "startDate": target_date.strftime("%Y%m%d"),
            "endDate": target_date.strftime("%Y%m%d"),
        }
        payload = cast(dict[str, object], self.http.get_json(base_url, params=params, use_cache=self.use_cache))
        source_url = f"{base_url}?{urlencode(params)}"
        return payload, source_url

    def _extract_public_api_key(self, html: str) -> str | None:
        for pattern in PUBLIC_API_KEY_PATTERNS:
            match = pattern.search(html)
            if match:
                return match.group("value")
        return None

    @staticmethod
    def _location_id(spec: MarketSpec) -> str:
        parts = [part for part in urlparse(spec.official_source_url).path.split("/") if part]
        if len(parts) >= 4 and parts[0] == "history" and parts[1] == "daily":
            country_code = parts[2].upper()
            return f"{spec.station_id}:9:{country_code}"
        if len(parts) >= 2 and parts[0] == "weather":
            country_code = COUNTRY_CODE_BY_NAME.get((spec.country or "").strip().lower())
            if not country_code:
                msg = f"Unsupported Wunderground source URL: {spec.official_source_url}"
                raise ValueError(msg)
            return f"{spec.station_id}:9:{country_code}"
        if len(parts) < 4 or parts[0] != "history" or parts[1] != "daily":
            msg = f"Unsupported Wunderground source URL: {spec.official_source_url}"
            raise ValueError(msg)
        country_code = parts[2].upper()
        return f"{spec.station_id}:9:{country_code}"

    @staticmethod
    def _parse_daily_max(html: str) -> float:
        if any(marker in html for marker in NO_DATA_RECORDED_MARKERS):
            msg = "Could not parse Wunderground daily max"
            raise TruthSourceParseError(msg)
        for pattern in TEMP_MAX_PATTERNS:
            match = pattern.search(html)
            if match:
                return float(match.group("value"))
        msg = "Could not parse Wunderground daily max"
        raise TruthSourceParseError(msg)

    @staticmethod
    def _parse_historical_max(payload: dict[str, object]) -> float:
        observations = payload.get("observations")
        if not isinstance(observations, list) or not observations:
            msg = "No Wunderground historical observations found"
            raise TruthSourceLagError(msg)

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
            raise TruthSourceLagError(msg)
        return max(values)
