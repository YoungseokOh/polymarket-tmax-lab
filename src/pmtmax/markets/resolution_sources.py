"""Resolution source metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolutionSource:
    key: str
    display_name: str
    url_hint: str
    requires_finalized_snapshot: bool = True


RESOLUTION_SOURCES: dict[str, ResolutionSource] = {
    "wunderground": ResolutionSource(
        key="wunderground",
        display_name="Wunderground",
        url_hint="https://www.wunderground.com/history/daily/",
    ),
    "hko": ResolutionSource(
        key="hko",
        display_name="Hong Kong Observatory Daily Extract",
        url_hint="https://data.weather.gov.hk/weatherAPI/opendata/opendata.php",
    ),
    "cwa": ResolutionSource(
        key="cwa",
        display_name="Central Weather Administration",
        url_hint="https://codis.cwa.gov.tw/StationData",
    ),
}

