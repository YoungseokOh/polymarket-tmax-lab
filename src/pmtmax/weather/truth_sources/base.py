"""Truth-source interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

from pydantic import BaseModel

from pmtmax.markets.market_spec import MarketSpec
from pmtmax.storage.schemas import ObservationRecord


class TruthSourceLagError(ValueError):
    """Raised when a truth source is known to lag behind the target settlement date."""

    def __init__(self, message: str, *, latest_available_date: date | None = None) -> None:
        super().__init__(message)
        self.latest_available_date = latest_available_date


class TruthSourceParseError(ValueError):
    """Raised when a truth source responds but its payload cannot be parsed safely."""


class TruthFetchBundle(BaseModel):
    observation: ObservationRecord
    raw_payload: dict[str, Any] | list[dict[str, Any]] | str
    media_type: str
    source_url: str
    archive_source_url: str | None = None
    source_provenance: str = "live"


class TruthSource(ABC):
    """Interface for official settlement-value retrieval."""

    def fetch_daily_observation(self, spec: MarketSpec, target_date: date) -> ObservationRecord:
        """Fetch the exact-source observation for one settlement date."""

        return self.fetch_observation_bundle(spec, target_date).observation

    @abstractmethod
    def fetch_observation_bundle(self, spec: MarketSpec, target_date: date) -> TruthFetchBundle:
        """Fetch the exact-source observation plus its raw source payload."""


def fahrenheit_to_celsius(value: float) -> float:
    """Convert Fahrenheit to Celsius."""

    return (value - 32.0) * 5.0 / 9.0


def celsius_to_fahrenheit(value: float) -> float:
    """Convert Celsius to Fahrenheit."""

    return value * 9.0 / 5.0 + 32.0
