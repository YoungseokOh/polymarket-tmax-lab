"""METAR real-time observation client via Aviation Weather API."""

from __future__ import annotations

from datetime import UTC, datetime

from pmtmax.http import CachedHttpClient
from pmtmax.logging_utils import get_logger
from pmtmax.storage.schemas import MetarObservation

LOGGER = get_logger(__name__)


class MetarClient:
    """Fetch decoded METAR observations from aviationweather.gov."""

    def __init__(self, http: CachedHttpClient, base_url: str) -> None:
        self.http = http
        self.base_url = base_url.rstrip("/")

    def fetch_latest(self, icao_code: str) -> MetarObservation | None:
        """Return the single most recent METAR for *icao_code*, or ``None``."""

        observations = self._fetch(icao_code, hours=1)
        return observations[0] if observations else None

    def fetch_recent(self, icao_code: str, hours: int = 24) -> list[MetarObservation]:
        """Return up to *hours* worth of decoded METARs."""

        return self._fetch(icao_code, hours=hours)

    def _fetch(self, icao_code: str, hours: int) -> list[MetarObservation]:
        url = f"{self.base_url}/metar"
        params = {"ids": icao_code, "format": "json", "hours": hours}
        try:
            payload = self.http.get_json(url, params=params, use_cache=False)
        except Exception:
            LOGGER.warning("METAR fetch failed for %s", icao_code)
            return []

        if not isinstance(payload, list):
            return []

        results: list[MetarObservation] = []
        for item in payload:
            obs = _parse_metar_item(item)
            if obs is not None:
                results.append(obs)
        return results


def _parse_metar_item(item: dict) -> MetarObservation | None:
    """Parse a single decoded METAR JSON object."""

    station_id = item.get("icaoId") or item.get("stationId", "")
    if not station_id:
        return None

    temp_c = item.get("temp")
    if temp_c is None:
        return None

    obs_time = item.get("obsTime") or item.get("reportTime")
    if obs_time is None:
        observed_at = datetime.now(tz=UTC)
    elif isinstance(obs_time, int | float):
        observed_at = datetime.fromtimestamp(obs_time, tz=UTC)
    else:
        try:
            observed_at = datetime.fromisoformat(str(obs_time))
        except ValueError:
            observed_at = datetime.now(tz=UTC)

    return MetarObservation(
        station_id=station_id,
        observed_at=observed_at,
        temp_c=float(temp_c),
        dew_point_c=_opt_float(item.get("dewp")),
        wind_speed_kt=_opt_float(item.get("wspd")),
        raw_metar=item.get("rawOb", ""),
    )


def _opt_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
