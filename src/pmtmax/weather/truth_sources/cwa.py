"""Central Weather Administration truth adapter."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path

from pmtmax.http import CachedHttpClient
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.storage.schemas import ObservationRecord
from pmtmax.weather.truth_sources.base import TruthFetchBundle, TruthSource


class CwaTruthSource(TruthSource):
    """Best-effort CWA adapter using cached official exports or supplied snapshots."""

    def __init__(self, http: CachedHttpClient, snapshot_dir: Path | None = None) -> None:
        self.http = http
        self.snapshot_dir = snapshot_dir

    def fetch_observation_bundle(self, spec: MarketSpec, target_date: date) -> TruthFetchBundle:
        snapshot = self._load_snapshot(spec.station_id, target_date)
        if snapshot is None:
            msg = (
                "CWA exact-source retrieval requires a cached official snapshot or an adapter override. "
                "This repo fails closed instead of substituting another station."
            )
            raise RuntimeError(msg)
        value = self._parse_daily_max(snapshot, target_date)
        observation = ObservationRecord(
            source="Central Weather Administration",
            station_id=spec.station_id,
            local_date=target_date,
            daily_max=value,
            unit="C",
            finalized_at=datetime.now(tz=UTC),
        )
        return TruthFetchBundle(
            observation=observation,
            raw_payload=snapshot,
            media_type="text/html",
            source_url=spec.official_source_url,
        )

    def _load_snapshot(self, station_id: str, target_date: date) -> str | None:
        if self.snapshot_dir is None:
            return None
        candidate = self.snapshot_dir / f"{station_id}_{target_date:%Y%m}.html"
        if candidate.exists():
            return candidate.read_text()
        return None

    @staticmethod
    def _parse_daily_max(html: str, target_date: date) -> float:
        pattern = re.compile(
            rf"{target_date.year}\D+{target_date.month}\D+{target_date.day}\D+(?P<value>-?\d+(?:\.\d+)?)"
        )
        match = pattern.search(html)
        if not match:
            msg = "Could not parse CWA daily max from supplied snapshot"
            raise ValueError(msg)
        return float(match.group("value"))
