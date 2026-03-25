"""Open-Meteo forecast and archive client."""

from __future__ import annotations

from typing import Any, cast

from pmtmax.http import CachedHttpClient


class OpenMeteoClient:
    """Client for Open-Meteo forecast and historical forecast endpoints."""

    def __init__(
        self,
        http: CachedHttpClient,
        base_url: str,
        archive_base_url: str,
        single_runs_base_url: str = "https://single-runs-api.open-meteo.com",
    ) -> None:
        self.http = http
        self.base_url = base_url.rstrip("/")
        self.archive_base_url = archive_base_url.rstrip("/")
        self.single_runs_base_url = single_runs_base_url.rstrip("/")

    def forecast(
        self,
        *,
        latitude: float,
        longitude: float,
        model: str,
        hourly: list[str],
        forecast_days: int = 7,
        timezone: str = "UTC",
    ) -> dict[str, Any]:
        """Fetch current forecast data. Cached for 6 hours to avoid rate limits."""

        payload = self.http.get_json(
            f"{self.base_url}/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "models": model,
                "hourly": ",".join(hourly),
                "forecast_days": forecast_days,
                "timezone": timezone,
            },
            use_cache=True,
            cache_ttl_seconds=6 * 3600,
        )
        return cast(dict[str, Any], payload)

    def historical_forecast(
        self,
        *,
        latitude: float,
        longitude: float,
        model: str,
        hourly: list[str],
        start_date: str,
        end_date: str,
        timezone: str = "UTC",
    ) -> dict[str, Any]:
        """Fetch historical forecast archives. Cached indefinitely (stable data)."""

        payload = self.http.get_json(
            f"{self.archive_base_url}/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "models": model,
                "hourly": ",".join(hourly),
                "start_date": start_date,
                "end_date": end_date,
                "timezone": timezone,
            },
            use_cache=True,
        )
        return cast(dict[str, Any], payload)

    def single_run(
        self,
        *,
        latitude: float,
        longitude: float,
        model: str,
        hourly: list[str],
        run: str,
        forecast_days: int = 7,
        timezone: str = "UTC",
    ) -> dict[str, Any]:
        """Fetch an exact archived weather-model run. Cached indefinitely (immutable run)."""

        payload = self.http.get_json(
            f"{self.single_runs_base_url}/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "models": model,
                "hourly": ",".join(hourly),
                "run": run,
                "forecast_days": forecast_days,
                "timezone": timezone,
            },
            use_cache=True,
        )
        return cast(dict[str, Any], payload)

    def previous_runs(
        self,
        *,
        latitude: float,
        longitude: float,
        model: str,
        hourly: list[str],
        forecast_days: int = 7,
        timezone: str = "UTC",
    ) -> dict[str, Any]:
        """Fetch previous forecast runs for pseudo-ensemble construction. Cached 6 hours."""

        payload = self.http.get_json(
            f"{self.base_url}/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "models": model,
                "hourly": ",".join(hourly),
                "forecast_days": forecast_days,
                "previous_runs": "true",
                "timezone": timezone,
            },
            use_cache=True,
            cache_ttl_seconds=6 * 3600,
        )
        return cast(dict[str, Any], payload)

    def ensemble(
        self,
        *,
        latitude: float,
        longitude: float,
        models: list[str],
        hourly: list[str],
        timezone: str = "UTC",
    ) -> dict[str, Any]:
        """Fetch ensemble forecasts when publicly available."""

        payload = self.http.get_json(
            f"{self.base_url}/v1/ensemble",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "models": ",".join(models),
                "hourly": ",".join(hourly),
                "timezone": timezone,
            },
            use_cache=False,
        )
        return cast(dict[str, Any], payload)
