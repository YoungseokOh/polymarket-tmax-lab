from __future__ import annotations

import httpx
import respx

from pmtmax.http import CachedHttpClient
from pmtmax.weather.openmeteo_client import OpenMeteoClient


@respx.mock
def test_historical_forecast_uses_generic_forecast_archive_endpoint(tmp_path) -> None:
    route = respx.get("https://historical-forecast-api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json={"hourly": {"time": [], "temperature_2m": []}})
    )
    client = OpenMeteoClient(CachedHttpClient(tmp_path / "cache"), "https://api.open-meteo.com", "https://historical-forecast-api.open-meteo.com")

    client.historical_forecast(
        latitude=37.4602,
        longitude=126.4407,
        model="ecmwf_ifs025",
        hourly=["temperature_2m"],
        start_date="2025-12-11",
        end_date="2025-12-11",
        timezone="Asia/Seoul",
    )

    assert route.called


@respx.mock
def test_forecast_accepts_gfs_seamless_model(tmp_path) -> None:
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json={"hourly": {"time": [], "temperature_2m": []}})
    )
    client = OpenMeteoClient(CachedHttpClient(tmp_path / "cache"), "https://api.open-meteo.com", "https://historical-forecast-api.open-meteo.com")

    client.forecast(
        latitude=40.7128,
        longitude=-74.0060,
        model="gfs_seamless",
        hourly=["temperature_2m"],
        forecast_days=2,
        timezone="America/New_York",
    )

    assert route.called
    request = route.calls[0].request
    assert "gfs_seamless" in str(request.url)


@respx.mock
def test_single_run_uses_single_runs_host(tmp_path) -> None:
    route = respx.get("https://single-runs-api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json={"hourly": {"time": [], "temperature_2m": []}})
    )
    client = OpenMeteoClient(CachedHttpClient(tmp_path / "cache"), "https://api.open-meteo.com", "https://historical-forecast-api.open-meteo.com")

    client.single_run(
        latitude=37.4602,
        longitude=126.4407,
        model="ecmwf_ifs025",
        hourly=["temperature_2m"],
        run="2025-12-10T12:00",
        forecast_days=2,
        timezone="Asia/Seoul",
    )

    assert route.called
