"""Live intraday temperature observations for observation-driven trading."""

from __future__ import annotations

import csv
import io
import re
from calendar import monthrange
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from pmtmax.http import CachedHttpClient
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.storage.schemas import LiveTemperatureObservation
from pmtmax.weather.truth_sources.amo_air_calp import AMO_AIR_CALP_URL
from pmtmax.weather.truth_sources.cwa import CODIS_STATION_API_URL

HKO_TEXT_READINGS_URL = "https://www.weather.gov.hk/textonly/v2/forecast/text_readings_v2_e.htm"
HKO_STATION_LABELS = {
    "HKA": "Chek Lap Kok",
}

_HKO_UPDATED_AT_PATTERN = re.compile(
    r"Latest readings recorded at\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})\s+Hong Kong Time\s+"
    r"(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)\s+(?P<year>\d{4})"
)
_HKO_STATION_ROW_PATTERN = re.compile(
    r"^(?P<label>.+?)\s+(?P<current>-?\d+(?:\.\d+)?)\s+(?P<humidity>N/A|\d+)\s+"
    r"(?P<maximum>-?\d+(?:\.\d+)?)\s*/\s*(?P<minimum>-?\d+(?:\.\d+)?)"
)


def fetch_intraday_observations(
    spec: MarketSpec,
    *,
    http: CachedHttpClient,
    observed_at: datetime,
) -> list[LiveTemperatureObservation]:
    """Return documented intraday lower-bound observations for one market spec."""

    candidates: list[LiveTemperatureObservation] = []
    adapter_key = spec.adapter_key()
    truth_source_key = spec.truth_source_key()

    if adapter_key == "hko":
        observation = _fetch_hko_intraday_observation(spec, http=http)
        if observation is not None:
            candidates.append(observation)

    if adapter_key == "cwa":
        observation = _fetch_cwa_intraday_observation(spec, http=http, observed_at=observed_at)
        if observation is not None:
            candidates.append(observation)

    if truth_source_key == "amo_air_calp":
        observation = _fetch_air_calp_intraday_observation(spec, http=http, observed_at=observed_at)
        if observation is not None:
            candidates.append(observation)

    return candidates


def _fetch_hko_intraday_observation(
    spec: MarketSpec,
    *,
    http: CachedHttpClient,
) -> LiveTemperatureObservation | None:
    payload = http.get_text(HKO_TEXT_READINGS_URL, use_cache=False)
    station_label = HKO_STATION_LABELS.get(spec.station_id, spec.station_name)
    observed_at = _parse_hko_updated_at(payload)
    row = _parse_hko_station_row(payload, station_label)
    if observed_at is None or row is None:
        return None
    return LiveTemperatureObservation(
        source_family="official_intraday",
        observation_source="hko_text_readings_v2",
        station_id=spec.station_id,
        observed_at=observed_at,
        lower_bound_temp_c=row["maximum"],
        current_temp_c=row["current"],
        daily_high_so_far_c=row["maximum"],
        source_confidence="exact_public_intraday",
    )


def _fetch_cwa_intraday_observation(
    spec: MarketSpec,
    *,
    http: CachedHttpClient,
    observed_at: datetime,
) -> LiveTemperatureObservation | None:
    target_date = spec.target_local_date
    month_end = monthrange(target_date.year, target_date.month)[1]
    request_payload = {
        "type": "report_month",
        "stn_type": "cwb",
        "stn_ID": spec.station_id,
        "more": "",
        "start": f"{target_date:%Y-%m}-01T00:00:00",
        "end": f"{target_date:%Y-%m}-{month_end:02d}T00:00:00",
    }
    response_payload = http.post_json(CODIS_STATION_API_URL, data=request_payload, use_cache=False)
    row = _parse_cwa_daily_row(response_payload, target_date)
    if row is None:
        return None
    air_temperature = row.get("AirTemperature")
    if not isinstance(air_temperature, dict):
        return None
    maximum = air_temperature.get("Maximum")
    if maximum is None:
        return None
    maximum_time = _parse_local_iso_datetime(
        air_temperature.get("MaximumTime"),
        timezone=spec.timezone,
        fallback=observed_at,
    )
    return LiveTemperatureObservation(
        source_family="official_intraday",
        observation_source="cwa_codis_report_month",
        station_id=spec.station_id,
        observed_at=maximum_time,
        lower_bound_temp_c=float(maximum),
        current_temp_c=None,
        daily_high_so_far_c=float(maximum),
        source_confidence="exact_public_intraday",
    )


def _fetch_air_calp_intraday_observation(
    spec: MarketSpec,
    *,
    http: CachedHttpClient,
    observed_at: datetime,
) -> LiveTemperatureObservation | None:
    params = {
        "icao": spec.public_truth_station_id or spec.station_id,
        "yyyymm": spec.target_local_date.strftime("%Y%m"),
    }
    payload = http.get_text(AMO_AIR_CALP_URL, params=params, use_cache=False)
    rows = list(csv.DictReader(io.StringIO(payload)))
    row = _air_calp_row_for_date(rows, spec.target_local_date)
    if row is None:
        return None
    maximum = _parse_air_calp_tenths_celsius(row.get("TMP_MAX"))
    if maximum is None:
        return None
    maximum_time = _parse_air_calp_timestamp(
        target_date=spec.target_local_date,
        raw_time=row.get("TMP_MAX_TM"),
        timezone=spec.timezone,
        fallback=observed_at,
    )
    return LiveTemperatureObservation(
        source_family="research_intraday",
        observation_source="amo_air_calp_intraday",
        station_id=str(spec.public_truth_station_id or spec.station_id),
        observed_at=maximum_time,
        lower_bound_temp_c=maximum,
        current_temp_c=None,
        daily_high_so_far_c=maximum,
        source_confidence="research_intraday_same_airport",
    )


def _parse_hko_updated_at(payload: str) -> datetime | None:
    match = _HKO_UPDATED_AT_PATTERN.search(payload)
    if match is None:
        return None
    try:
        local_dt = datetime.strptime(
            (
                f"{match.group('day')} {match.group('month')} {match.group('year')} "
                f"{match.group('hour')}:{match.group('minute')}"
            ),
            "%d %B %Y %H:%M",
        ).replace(tzinfo=ZoneInfo("Asia/Hong_Kong"))
    except ValueError:
        return None
    return local_dt.astimezone(UTC)


def _parse_hko_station_row(payload: str, station_label: str) -> dict[str, float] | None:
    normalized_label = station_label.strip()
    for line in payload.splitlines():
        compact = " ".join(line.split())
        if not compact.startswith(normalized_label):
            continue
        match = _HKO_STATION_ROW_PATTERN.match(compact)
        if match is None or match.group("label").strip() != normalized_label:
            continue
        return {
            "current": float(match.group("current")),
            "maximum": float(match.group("maximum")),
            "minimum": float(match.group("minimum")),
        }
    return None


def _parse_cwa_daily_row(payload: object, target_date: date) -> dict[str, object] | None:
    if not isinstance(payload, dict) or payload.get("code") != 200:
        return None
    for station_payload in payload.get("data", []):
        if not isinstance(station_payload, dict):
            continue
        for row in station_payload.get("dts", []):
            if not isinstance(row, dict):
                continue
            data_date = row.get("DataDate")
            if isinstance(data_date, str) and data_date.startswith(target_date.isoformat()):
                return row
    return None


def _air_calp_row_for_date(rows: list[dict[str, str]], target_date: date) -> dict[str, str] | None:
    target = target_date.strftime("%Y%m%d")
    for row in rows:
        if row.get("TM") == target:
            return row
    return None


def _parse_air_calp_tenths_celsius(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    value = raw_value.strip().strip('"')
    if not value:
        return None
    try:
        return int(value) / 10.0
    except ValueError:
        return None


def _parse_air_calp_timestamp(
    *,
    target_date: date,
    raw_time: str | None,
    timezone: str,
    fallback: datetime,
) -> datetime:
    if raw_time is None:
        return _normalize_utc(fallback)
    value = raw_time.strip().strip('"')
    if not value:
        return _normalize_utc(fallback)
    if not value.isdigit():
        return _normalize_utc(fallback)
    padded = value.zfill(4)
    hour = int(padded[:2])
    minute = int(padded[2:])
    if hour >= 24 or minute >= 60:
        return _normalize_utc(fallback)
    local_dt = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        hour,
        minute,
        tzinfo=ZoneInfo(timezone),
    )
    return local_dt.astimezone(UTC)


def _parse_local_iso_datetime(
    raw_value: object,
    *,
    timezone: str,
    fallback: datetime,
) -> datetime:
    if not isinstance(raw_value, str) or not raw_value:
        return _normalize_utc(fallback)
    try:
        local_dt = datetime.fromisoformat(raw_value)
    except ValueError:
        return _normalize_utc(fallback)
    if local_dt.tzinfo is None:
        local_dt = local_dt.replace(tzinfo=ZoneInfo(timezone))
    return local_dt.astimezone(UTC)


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
