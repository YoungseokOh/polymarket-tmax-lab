"""Hong Kong Observatory truth adapter."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

from pmtmax.http import CachedHttpClient
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.storage.schemas import ObservationRecord
from pmtmax.weather.truth_sources.base import TruthFetchBundle, TruthSource


class HkoTruthSource(TruthSource):
    """Retrieve daily maximum temperatures from HKO open data."""

    def __init__(self, http: CachedHttpClient, snapshot_dir: Path | None = None) -> None:
        self.http = http
        self.snapshot_dir = snapshot_dir

    def fetch_observation_bundle(self, spec: MarketSpec, target_date: date) -> TruthFetchBundle:
        url = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
        payload = self._load_snapshot(spec.station_id, target_date)
        if payload is None:
            payload = self.http.get_json(
                url,
                params={
                    "dataType": "CLMMAXT",
                    "station": spec.station_id,
                    "year": target_date.year,
                    "month": target_date.month,
                    "rformat": "json",
                    "lang": "en",
                },
                use_cache=True,
            )
        value = self._parse_value(payload, target_date)
        observation = ObservationRecord(
            source="Hong Kong Observatory",
            station_id=spec.station_id,
            local_date=target_date,
            daily_max=value,
            unit="C",
            finalized_at=datetime.now(tz=UTC),
        )
        return TruthFetchBundle(
            observation=observation,
            raw_payload=payload,
            media_type="application/json",
            source_url=url,
        )

    def _load_snapshot(self, station_id: str, target_date: date) -> dict | None:
        if self.snapshot_dir is None:
            return None
        candidate = self.snapshot_dir / f"{station_id}_{target_date:%Y%m}.json"
        if candidate.exists():
            return cast(dict, json.loads(candidate.read_text()))
        return None

    @staticmethod
    def _parse_value(payload: dict, target_date: date) -> float:
        for row in payload.get("data", []):
            if int(row[0]) == target_date.year and int(row[1]) == target_date.month and int(row[2]) == target_date.day:
                return float(row[3])
        msg = f"No HKO record for {target_date.isoformat()}"
        raise ValueError(msg)
