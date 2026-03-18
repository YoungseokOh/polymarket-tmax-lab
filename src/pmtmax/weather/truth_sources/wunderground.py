"""Wunderground truth adapter."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path

from pmtmax.http import CachedHttpClient
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.storage.schemas import ObservationRecord
from pmtmax.weather.truth_sources.base import TruthFetchBundle, TruthSource

TEMP_MAX_PATTERNS = [
    re.compile(r'"temperatureMax"\s*:\s*\{"value"\s*:\s*(?P<value>-?\d+(?:\.\d+)?)', re.I),
    re.compile(r'"temperatureMax"\s*:\s*(?P<value>-?\d+(?:\.\d+)?)', re.I),
]


class WundergroundTruthSource(TruthSource):
    """Retrieve daily max temperature from Wunderground history pages."""

    def __init__(self, http: CachedHttpClient, snapshot_dir: Path | None = None) -> None:
        self.http = http
        self.snapshot_dir = snapshot_dir

    def fetch_observation_bundle(self, spec: MarketSpec, target_date: date) -> TruthFetchBundle:
        html = self._load_snapshot(spec.station_id, target_date)
        url = f"{spec.official_source_url.rstrip('/')}/date/{target_date.isoformat()}"
        if html is None:
            html = self.http.get_text(url, use_cache=True)
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

    def _load_snapshot(self, station_id: str, target_date: date) -> str | None:
        if self.snapshot_dir is None:
            return None
        candidate = self.snapshot_dir / f"{station_id}_{target_date:%Y%m%d}.html"
        if candidate.exists():
            return candidate.read_text()
        return None

    @staticmethod
    def _parse_daily_max(html: str) -> float:
        for pattern in TEMP_MAX_PATTERNS:
            match = pattern.search(html)
            if match:
                return float(match.group("value"))
        msg = "Could not parse Wunderground daily max"
        raise ValueError(msg)
