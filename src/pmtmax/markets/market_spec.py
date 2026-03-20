"""Market domain models for temperature-max contracts."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from pmtmax.markets.station_registry import lookup_station, lookup_station_by_station_id


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
    truth_track: Literal["exact_public", "research_public"] = "exact_public"
    settlement_eligible: bool = True
    public_truth_source_name: str | None = None
    public_truth_station_id: str | None = None
    research_priority: Literal["core", "expansion"] = "expansion"
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

    @model_validator(mode="after")
    def _hydrate_station_catalog_defaults(self) -> MarketSpec:
        definition = lookup_station_by_station_id(self.station_id) or lookup_station(self.city)
        if definition is None:
            return self
        self.city = definition.city
        if not self.country:
            self.country = definition.country
        if not self.timezone or self.timezone == "UTC":
            self.timezone = definition.timezone
        if self.station_lat is None:
            self.station_lat = definition.lat
        if self.station_lon is None:
            self.station_lon = definition.lon
        if self.adapter_key() == "wunderground":
            self.truth_track = "research_public"
            self.settlement_eligible = False
            self.public_truth_source_name = definition.public_truth_source_name
            self.public_truth_station_id = definition.public_truth_station_id
        else:
            self.truth_track = "exact_public"
            self.settlement_eligible = True
            if not self.public_truth_source_name:
                self.public_truth_source_name = definition.public_truth_source_name or self.official_source_name
            if not self.public_truth_station_id:
                self.public_truth_station_id = definition.public_truth_station_id or self.station_id
        self.research_priority = definition.research_priority  # type: ignore[assignment]
        return self

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
        if "noaa" in name:
            return "noaa"
        return "unknown"

    def truth_source_key(self) -> str:
        """Return the default truth adapter used for collection/materialization."""

        public_name = (self.public_truth_source_name or "").lower()
        if self.truth_track == "research_public" and "air_calp" in public_name:
            return "amo_air_calp"
        if self.truth_track == "research_public" and self.adapter_key() == "wunderground":
            return "noaa_global_hourly"
        if self.adapter_key() == "noaa":
            return "noaa_global_hourly"
        return self.adapter_key()
