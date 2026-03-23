"""Execution guardrails."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def spread_ok(best_bid: float, best_ask: float, max_spread_bps: int) -> bool:
    """Check whether spread stays inside the configured threshold."""

    if best_ask <= 0:
        return False
    mid = (best_bid + best_ask) / 2.0
    spread_bps = ((best_ask - best_bid) / max(mid, 1e-6)) * 10_000.0
    return spread_bps <= max_spread_bps


def forecast_fresh(generated_at: datetime, stale_forecast_minutes: int) -> bool:
    """Check whether a forecast is still considered fresh."""

    threshold = datetime.now(tz=UTC) - timedelta(minutes=stale_forecast_minutes)
    if generated_at.tzinfo is None:
        return generated_at >= threshold.replace(tzinfo=None)
    return generated_at.astimezone(UTC) >= threshold


def exposure_ok(current_exposure: float, proposed_size: float, limit: float) -> bool:
    """Check whether exposure after the trade stays below a cap."""

    return current_exposure + proposed_size <= limit
