"""Forecast-change exit signal."""

from __future__ import annotations

from pmtmax.storage.schemas import ProbForecast


def should_forecast_exit(
    position_outcome_label: str,
    current_forecast: ProbForecast,
    buffer: float = 0.05,
) -> bool:
    """Return ``True`` when the forecast probability for *position_outcome_label* drops below *buffer*."""

    prob = current_forecast.outcome_probabilities.get(position_outcome_label, 0.0)
    return prob < buffer
