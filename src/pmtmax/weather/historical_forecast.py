"""Historical forecast reconstruction."""

from __future__ import annotations

from datetime import date
from typing import Any

from pmtmax.weather.openmeteo_client import OpenMeteoClient


def fetch_historical_forecast_slice(
    client: OpenMeteoClient,
    *,
    latitude: float,
    longitude: float,
    model: str,
    hourly: list[str],
    target_date: date,
    timezone: str,
) -> dict[str, Any]:
    """Fetch archived forecasts for the target settlement day."""

    day = target_date.isoformat()
    return client.historical_forecast(
        latitude=latitude,
        longitude=longitude,
        model=model,
        hourly=hourly,
        start_date=day,
        end_date=day,
        timezone=timezone,
    )

