"""Central Weather Administration truth adapter."""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

from pmtmax.http import CachedHttpClient
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.storage.schemas import ObservationRecord
from pmtmax.weather.truth_sources.base import TruthFetchBundle, TruthSource

CODIS_STATION_API_URL = "https://codis.cwa.gov.tw/api/station"


class CwaTruthSource(TruthSource):
    """Best-effort CWA adapter using cached official exports or supplied snapshots."""

    def __init__(self, http: CachedHttpClient, snapshot_dir: Path | None = None) -> None:
        self.http = http
        self.snapshot_dir = snapshot_dir

    def fetch_observation_bundle(self, spec: MarketSpec, target_date: date) -> TruthFetchBundle:
        snapshot = self._load_snapshot(spec.station_id, target_date)
        if snapshot is not None:
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

        payload = self._fetch_codis_payload(spec.station_id, target_date)
        value = self._parse_codis_daily_max(payload["response"], target_date)
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
            raw_payload=payload,
            media_type="application/json",
            source_url=CODIS_STATION_API_URL,
        )

    def _load_snapshot(self, station_id: str, target_date: date) -> str | None:
        if self.snapshot_dir is None:
            return None
        candidate = self.snapshot_dir / f"{station_id}_{target_date:%Y%m}.html"
        if candidate.exists():
            return candidate.read_text()
        return None

    def _fetch_codis_payload(self, station_id: str, target_date: date) -> dict[str, object]:
        month_end = monthrange(target_date.year, target_date.month)[1]
        request_payload = {
            "type": "report_month",
            "stn_type": "cwb",
            "stn_ID": station_id,
            "more": "",
            "start": f"{target_date:%Y-%m}-01T00:00:00",
            "end": f"{target_date:%Y-%m}-{month_end:02d}T00:00:00",
        }
        response_payload = cast(dict[str, object], self.http.post_json(CODIS_STATION_API_URL, data=request_payload, use_cache=True))
        return {"request": request_payload, "response": response_payload}

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

    @staticmethod
    def _parse_codis_daily_max(payload: dict[str, object], target_date: date) -> float:
        if payload.get("code") != 200:
            msg = f"CWA CODiS request failed: {payload.get('message')}"
            raise RuntimeError(msg)

        for station_payload in payload.get("data", []):
            if not isinstance(station_payload, dict):
                continue
            for row in station_payload.get("dts", []):
                if not isinstance(row, dict):
                    continue
                data_date = row.get("DataDate")
                if not isinstance(data_date, str) or not data_date.startswith(target_date.isoformat()):
                    continue
                air_temperature = row.get("AirTemperature")
                if not isinstance(air_temperature, dict) or air_temperature.get("Maximum") is None:
                    msg = "CWA CODiS daily row is missing AirTemperature.Maximum"
                    raise ValueError(msg)
                return float(cast(float, air_temperature["Maximum"]))

        msg = f"No CWA CODiS daily row for {target_date.isoformat()}"
        raise ValueError(msg)
