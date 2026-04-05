"""Canonical station metadata for supported city templates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


CATALOG_PATH = Path(__file__).resolve().parents[3] / "configs" / "market_inventory" / "station_catalog.json"


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
    truth_track: str = "exact_public"
    settlement_eligible: bool = True
    public_truth_source_name: str | None = None
    public_truth_station_id: str | None = None
    research_priority: str = "expansion"


@lru_cache(maxsize=1)
def _station_maps() -> tuple[dict[str, StationDefinition], dict[str, StationDefinition], dict[str, str]]:
    payload = json.loads(CATALOG_PATH.read_text())
    by_city: dict[str, StationDefinition] = {}
    by_station_id: dict[str, StationDefinition] = {}
    aliases: dict[str, str] = {}
    for key, raw in payload.items():
        city = str(raw.get("city") or key)
        definition = StationDefinition(
            city=city,
            country=str(raw["country"]),
            timezone=str(raw["timezone"]),
            official_source_name=str(raw["official_source_name"]),
            station_id=str(raw["station_id"]),
            station_name=str(raw["station_name"]),
            lat=float(raw["lat"]),
            lon=float(raw["lon"]),
            aliases=tuple(str(item).lower() for item in raw.get("aliases", [])),
            truth_track=str(raw.get("truth_track", "exact_public")),
            settlement_eligible=bool(raw.get("settlement_eligible", True)),
            public_truth_source_name=str(raw.get("public_truth_source_name") or "") or None,
            public_truth_station_id=str(raw.get("public_truth_station_id") or "") or None,
            research_priority=str(raw.get("research_priority", "expansion")),
        )
        by_city.setdefault(city, definition)
        by_station_id[definition.station_id.upper()] = definition
        aliases.setdefault(city.lower(), city)
        aliases[definition.station_id.lower()] = city
        aliases[definition.station_name.lower()] = city
        for alias in definition.aliases:
            aliases[alias] = city
    return by_city, by_station_id, aliases


def supported_cities() -> list[str]:
    """Return canonical cities from the checked-in station catalog."""

    by_city, _, _ = _station_maps()
    return list(by_city)


def canonical_city(city: str) -> str:
    """Return the canonical supported city label when known."""

    lowered = city.strip().lower()
    _, _, aliases = _station_maps()
    return aliases.get(lowered, city.strip())


def lookup_station(city: str) -> StationDefinition | None:
    """Lookup canonical station metadata for a city or alias."""

    by_city, _, _ = _station_maps()
    return by_city.get(canonical_city(city))


def lookup_station_by_station_id(station_id: str) -> StationDefinition | None:
    """Lookup station metadata by the official station id from market rules."""

    _, by_station_id, _ = _station_maps()
    return by_station_id.get(station_id.strip().upper())


def lookup_city_stations(city: str) -> list[StationDefinition]:
    """Return all known station definitions for one canonical city."""

    canonical = canonical_city(city)
    by_city, by_station_id, _ = _station_maps()
    primary = by_city.get(canonical)
    matches = [definition for definition in by_station_id.values() if definition.city == canonical]
    if primary is None:
        return matches
    ordered = [primary]
    ordered.extend(definition for definition in matches if definition.station_id != primary.station_id)
    return ordered
