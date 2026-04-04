"""Hong Kong Observatory truth adapter."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast
from urllib.parse import urlencode

from pmtmax.http import CachedHttpClient
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.storage.schemas import ObservationRecord
from pmtmax.weather.truth_sources.base import TruthFetchBundle, TruthSource, TruthSourceLagError

HKO_OPEN_DATA_URL = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
WAYBACK_CDX_URL = "https://web.archive.org/cdx/search/cdx"
WAYBACK_REPLAY_URL = "https://web.archive.org/web/{timestamp}if_/{url}"


class HkoTruthSource(TruthSource):
    """Retrieve daily maximum temperatures from HKO open data."""

    def __init__(
        self,
        http: CachedHttpClient,
        snapshot_dir: Path | None = None,
        use_cache: bool = True,
    ) -> None:
        self.http = http
        self.snapshot_dir = snapshot_dir
        self.use_cache = use_cache

    def fetch_observation_bundle(self, spec: MarketSpec, target_date: date) -> TruthFetchBundle:
        month_params = self._month_params(spec.station_id, target_date)
        month_source_url = self._request_url(HKO_OPEN_DATA_URL, month_params)
        payload = self._load_snapshot(spec, target_date)
        if payload is not None:
            value = self._parse_value(payload, target_date)
            return self._bundle(spec, target_date, value=value, payload=payload, source_url=month_source_url)

        month_payload = self._fetch_month_payload(HKO_OPEN_DATA_URL, spec.station_id, target_date)
        month_value = self._extract_value(month_payload, target_date)
        if month_value is not None:
            return self._bundle(
                spec,
                target_date,
                value=month_value,
                payload=month_payload,
                source_url=month_source_url,
            )

        full_station_payload = self._fetch_full_station_payload(HKO_OPEN_DATA_URL, spec.station_id)
        full_station_value = self._extract_value(full_station_payload, target_date)
        if full_station_value is not None:
            return self._bundle(
                spec,
                target_date,
                value=full_station_value,
                payload=full_station_payload,
                source_url=self._request_url(HKO_OPEN_DATA_URL, self._full_station_params(spec.station_id)),
            )

        archived = self.restore_archived_month_payload(spec.station_id, target_date)
        if archived is not None:
            payload, archive_source_url = archived
            archived_value = self._extract_value(payload, target_date)
            if archived_value is not None:
                return self._bundle(
                    spec,
                    target_date,
                    value=archived_value,
                    payload=payload,
                    source_url=month_source_url,
                    archive_source_url=archive_source_url,
                    source_provenance="official_archive",
                )

        latest_available_date = self._latest_available_date(month_payload) or self._latest_available_date(full_station_payload)
        msg = f"No HKO record for {target_date.isoformat()}"
        raise TruthSourceLagError(msg, latest_available_date=latest_available_date)

    def restore_archived_month_payload(self, station_id: str, target_date: date) -> tuple[dict, str] | None:
        """Return an archived copy of the exact official monthly HKO payload when available."""

        params = self._month_params(station_id, target_date)
        official_url = self._request_url(HKO_OPEN_DATA_URL, params)
        captures = cast(
            list[list[str]] | list[object],
            self.http.get_json(
                WAYBACK_CDX_URL,
                params={
                    "url": official_url,
                    "output": "json",
                    "fl": "timestamp,original,statuscode,mimetype",
                    "filter": "statuscode:200",
                    "limit": "10",
                    "from": target_date.strftime("%Y%m"),
                },
                use_cache=self.use_cache,
            ),
        )
        if not isinstance(captures, list) or len(captures) <= 1:
            return None

        # CDX JSON responses include a header row first; use the latest capture available.
        for row in reversed(captures[1:]):
            if not isinstance(row, list) or not row:
                continue
            timestamp = str(row[0]).strip()
            if not timestamp:
                continue
            archive_url = WAYBACK_REPLAY_URL.format(timestamp=timestamp, url=official_url)
            payload = cast(dict, self.http.get_json(archive_url, use_cache=self.use_cache))
            if self._extract_value(payload, target_date) is not None:
                return payload, archive_url
        return None

    def _bundle(
        self,
        spec: MarketSpec,
        target_date: date,
        *,
        value: float,
        payload: dict,
        source_url: str,
        archive_source_url: str | None = None,
        source_provenance: str = "live",
    ) -> TruthFetchBundle:
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
            source_url=source_url,
            archive_source_url=archive_source_url,
            source_provenance=source_provenance,
        )

    def _load_snapshot(self, spec: MarketSpec, target_date: date) -> dict | None:
        if self.snapshot_dir is None:
            return None
        fixture_candidate = self.snapshot_dir / f"{spec.station_id}_{target_date:%Y%m}.json"
        if fixture_candidate.exists():
            return cast(dict, json.loads(fixture_candidate.read_text()))
        bronze_candidate = (
            self.snapshot_dir
            / "truth"
            / "hko"
            / spec.station_id
            / target_date.strftime("%Y%m")
            / f"{spec.market_id}_{target_date.isoformat()}.json"
        )
        if bronze_candidate.exists():
            return cast(dict, json.loads(bronze_candidate.read_text()))
        return None

    def _fetch_month_payload(self, url: str, station_id: str, target_date: date) -> dict:
        return cast(
            dict,
            self.http.get_json(url, params=self._month_params(station_id, target_date), use_cache=self.use_cache),
        )

    def _fetch_full_station_payload(self, url: str, station_id: str) -> dict:
        return cast(
            dict,
            self.http.get_json(url, params=self._full_station_params(station_id), use_cache=self.use_cache),
        )

    @staticmethod
    def _month_params(station_id: str, target_date: date) -> dict[str, object]:
        return {
            "dataType": "CLMMAXT",
            "station": station_id,
            "year": target_date.year,
            "month": target_date.month,
            "rformat": "json",
            "lang": "en",
        }

    @staticmethod
    def _full_station_params(station_id: str) -> dict[str, object]:
        return {
            "dataType": "CLMMAXT",
            "station": station_id,
            "rformat": "json",
            "lang": "en",
        }

    @staticmethod
    def _request_url(url: str, params: dict[str, object]) -> str:
        return f"{url}?{urlencode(params)}"

    @staticmethod
    def _extract_value(payload: dict, target_date: date) -> float | None:
        for row in payload.get("data", []):
            try:
                row_date = date(int(row[0]), int(row[1]), int(row[2]))
            except (TypeError, ValueError, IndexError):
                continue
            if row_date == target_date:
                return float(row[3])
        return None

    @classmethod
    def _parse_value(cls, payload: dict, target_date: date) -> float:
        value = cls._extract_value(payload, target_date)
        if value is None:
            msg = f"No HKO record for {target_date.isoformat()}"
            raise ValueError(msg)
        return value

    @staticmethod
    def _latest_available_date(payload: dict) -> date | None:
        latest: date | None = None
        for row in payload.get("data", []):
            try:
                row_date = date(int(row[0]), int(row[1]), int(row[2]))
            except (TypeError, ValueError, IndexError):
                continue
            latest = row_date if latest is None or row_date > latest else latest
        return latest
