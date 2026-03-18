"""KMA feature adapter over Open-Meteo."""

from __future__ import annotations

from pmtmax.weather.features import build_hourly_feature_frame
from pmtmax.weather.openmeteo_client import OpenMeteoClient

KMA_HOURLY_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
]


def fetch_kma_features(
    client: OpenMeteoClient,
    *,
    latitude: float,
    longitude: float,
    model: str,
    timezone: str,
) -> dict:
    """Fetch and normalize KMA-style deterministic features."""

    payload = client.forecast(
        latitude=latitude,
        longitude=longitude,
        model=model,
        hourly=KMA_HOURLY_VARS,
        timezone=timezone,
    )
    return build_hourly_feature_frame(payload)

