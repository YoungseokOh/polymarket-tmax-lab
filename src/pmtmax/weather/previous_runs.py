"""Previous run reconstruction for pseudo-ensemble inputs."""

from __future__ import annotations

from typing import Any

from pmtmax.weather.openmeteo_client import OpenMeteoClient


def fetch_previous_runs_slice(
    client: OpenMeteoClient,
    *,
    latitude: float,
    longitude: float,
    model: str,
    hourly: list[str],
    timezone: str,
) -> dict[str, Any]:
    """Fetch previous deterministic runs from Open-Meteo."""

    return client.previous_runs(
        latitude=latitude,
        longitude=longitude,
        model=model,
        hourly=hourly,
        timezone=timezone,
    )

