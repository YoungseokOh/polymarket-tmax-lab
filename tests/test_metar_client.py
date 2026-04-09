from __future__ import annotations

import httpx
import respx

from pmtmax.http import CachedHttpClient
from pmtmax.weather.metar_client import MetarClient

SAMPLE_METAR_RESPONSE = [
    {
        "icaoId": "KJFK",
        "obsTime": 1700000000,
        "temp": 5.0,
        "dewp": 1.0,
        "wspd": 12,
        "rawOb": "KJFK 141956Z 31012KT 10SM FEW250 05/01 A3012",
    },
    {
        "icaoId": "KJFK",
        "obsTime": 1699996400,
        "temp": 4.5,
        "dewp": 0.5,
        "wspd": 10,
        "rawOb": "KJFK 141856Z 32010KT 10SM SCT250 05/01 A3010",
    },
]


@respx.mock
def test_fetch_latest_returns_single_observation(tmp_path) -> None:
    route = respx.get("https://aviationweather.gov/api/data/metar").mock(
        return_value=httpx.Response(200, json=SAMPLE_METAR_RESPONSE),
    )
    http = CachedHttpClient(tmp_path / "cache")
    client = MetarClient(http, "https://aviationweather.gov/api/data")

    obs = client.fetch_latest("KJFK")

    assert route.called
    assert obs is not None
    assert obs.station_id == "KJFK"
    assert obs.temp_c == 5.0
    assert obs.dew_point_c == 1.0
    assert obs.wind_speed_kt == 12.0


@respx.mock
def test_fetch_recent_returns_all_observations(tmp_path) -> None:
    respx.get("https://aviationweather.gov/api/data/metar").mock(
        return_value=httpx.Response(200, json=SAMPLE_METAR_RESPONSE),
    )
    http = CachedHttpClient(tmp_path / "cache")
    client = MetarClient(http, "https://aviationweather.gov/api/data")

    obs_list = client.fetch_recent("KJFK", hours=24)

    assert len(obs_list) == 2


@respx.mock
def test_fetch_latest_returns_none_on_empty_response(tmp_path) -> None:
    respx.get("https://aviationweather.gov/api/data/metar").mock(
        return_value=httpx.Response(200, json=[]),
    )
    http = CachedHttpClient(tmp_path / "cache")
    client = MetarClient(http, "https://aviationweather.gov/api/data")

    obs = client.fetch_latest("XXXX")

    assert obs is None


@respx.mock
def test_fetch_latest_handles_missing_temp(tmp_path) -> None:
    respx.get("https://aviationweather.gov/api/data/metar").mock(
        return_value=httpx.Response(200, json=[{"icaoId": "KJFK", "obsTime": 1700000000}]),
    )
    http = CachedHttpClient(tmp_path / "cache")
    client = MetarClient(http, "https://aviationweather.gov/api/data")

    obs = client.fetch_latest("KJFK")

    assert obs is None
