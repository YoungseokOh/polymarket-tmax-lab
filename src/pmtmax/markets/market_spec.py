"""Market domain models for temperature-max contracts."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PrecisionRule(BaseModel):
    unit: Literal["C", "F"]
    step: float
    rounding: Literal["whole_degree", "range_bin", "exact_source"] = "whole_degree"
    source_precision_text: str


class OutcomeBin(BaseModel):
    label: str
    lower: float | None = None
    upper: float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    def contains(self, value: float) -> bool:
        """Return whether a realized value resolves into this outcome."""

        if self.lower is not None:
            if self.lower_inclusive and value < self.lower:
                return False
            if not self.lower_inclusive and value <= self.lower:
                return False
        if self.upper is not None:
            if self.upper_inclusive and value > self.upper:
                return False
            if not self.upper_inclusive and value >= self.upper:
                return False
        return True


class FinalizationPolicy(BaseModel):
    wait_for_finalized_data: bool = True
    ignore_post_final_revision: bool = True
    notes: str = ""


class MarketSpec(BaseModel):
    market_id: str
    event_id: str | None = None
    slug: str
    question: str
    condition_id: str | None = None
    token_ids: list[str] = Field(default_factory=list)
    city: str
    country: str | None = None
    target_local_date: date
    timezone: str
    official_source_name: str
    official_source_url: str
    station_id: str
    station_name: str
    station_lat: float | None = None
    station_lon: float | None = None
    unit: Literal["C", "F"]
    metric: Literal["daily_max_temperature"] = "daily_max_temperature"
    precision_rule: PrecisionRule
    outcome_schema: list[OutcomeBin]
    finalization_policy: FinalizationPolicy
    notes: str = ""

    @field_validator("outcome_schema")
    @classmethod
    def _validate_bins(cls, bins: list[OutcomeBin]) -> list[OutcomeBin]:
        if not bins:
            msg = "outcome_schema must not be empty"
            raise ValueError(msg)
        return bins

    def outcome_labels(self) -> list[str]:
        """Return ordered market outcome labels."""

        return [item.label for item in self.outcome_schema]

    def adapter_key(self) -> str:
        """Return a normalized truth-source key."""

        name = self.official_source_name.lower()
        if "wunderground" in name:
            return "wunderground"
        if "hong kong observatory" in name:
            return "hko"
        if "central weather administration" in name or "cwa" in name:
            return "cwa"
        return "unknown"

