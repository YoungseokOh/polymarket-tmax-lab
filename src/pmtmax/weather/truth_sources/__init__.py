"""Truth-source registry."""

from __future__ import annotations

from pathlib import Path

from pmtmax.http import CachedHttpClient
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.weather.truth_sources.base import TruthSource
from pmtmax.weather.truth_sources.cwa import CwaTruthSource
from pmtmax.weather.truth_sources.hko import HkoTruthSource
from pmtmax.weather.truth_sources.wunderground import WundergroundTruthSource


def make_truth_source(spec: MarketSpec, http: CachedHttpClient, snapshot_dir: Path | None = None) -> TruthSource:
    """Instantiate the right official truth adapter for a market."""

    key = spec.adapter_key()
    if key == "wunderground":
        return WundergroundTruthSource(http, snapshot_dir=snapshot_dir)
    if key == "hko":
        return HkoTruthSource(http, snapshot_dir=snapshot_dir)
    if key == "cwa":
        return CwaTruthSource(http, snapshot_dir=snapshot_dir)
    msg = f"Unsupported truth adapter for {spec.official_source_name}"
    raise ValueError(msg)
