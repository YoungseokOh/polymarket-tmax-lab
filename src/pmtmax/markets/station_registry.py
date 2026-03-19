"""Canonical station metadata for supported city templates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StationDefinition:
    city: str
    country: str
    timezone: str
    official_source_name: str
    station_id: str
    station_name: str
    lat: float
    lon: float
    aliases: tuple[str, ...] = ()


STATIONS: dict[str, StationDefinition] = {
    "Seoul": StationDefinition(
        city="Seoul",
        country="South Korea",
        timezone="Asia/Seoul",
        official_source_name="Wunderground",
        station_id="RKSI",
        station_name="Incheon Intl Airport",
        lat=37.4602,
        lon=126.4407,
        aliases=("incheon", "incheon intl airport", "rksi"),
    ),
    "NYC": StationDefinition(
        city="NYC",
        country="USA",
        timezone="America/New_York",
        official_source_name="Wunderground",
        station_id="KLGA",
        station_name="LaGuardia Airport",
        lat=40.7769,
        lon=-73.8740,
        aliases=("new york city", "laguardia", "laguardia airport", "klga"),
    ),
    "London": StationDefinition(
        city="London",
        country="United Kingdom",
        timezone="Europe/London",
        official_source_name="Wunderground",
        station_id="EGLC",
        station_name="London City Airport",
        lat=51.51,
        lon=0.028,
        aliases=("london city airport", "eglc"),
    ),
    "Hong Kong": StationDefinition(
        city="Hong Kong",
        country="Hong Kong",
        timezone="Asia/Hong_Kong",
        official_source_name="Hong Kong Observatory Daily Extract",
        station_id="HKA",
        station_name="Hong Kong International Airport",
        lat=22.3080,
        lon=113.9185,
        aliases=("hong kong international airport", "hka"),
    ),
    "Taipei": StationDefinition(
        city="Taipei",
        country="Taiwan",
        timezone="Asia/Taipei",
        official_source_name="Central Weather Administration",
        station_id="466920",
        station_name="Taipei",
        lat=25.0377,
        lon=121.5149,
        aliases=("taipei station", "466920"),
    ),
}


def canonical_city(city: str) -> str:
    """Return the canonical supported city label when known."""

    lowered = city.strip().lower()
    for definition in STATIONS.values():
        if lowered == definition.city.lower() or lowered in definition.aliases:
            return definition.city
    if lowered == "new york city":
        return "NYC"
    return city.strip()


def lookup_station(city: str) -> StationDefinition | None:
    """Lookup canonical station metadata for a city or alias."""

    canonical = canonical_city(city)
    return STATIONS.get(canonical)
