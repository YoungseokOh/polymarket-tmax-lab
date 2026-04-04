"""Truth-source registry."""

from __future__ import annotations

from pathlib import Path

from pmtmax.http import CachedHttpClient
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.weather.truth_sources.amo_air_calp import AmoAirCalpTruthSource
from pmtmax.weather.truth_sources.base import TruthSource
from pmtmax.weather.truth_sources.cwa import CwaTruthSource
from pmtmax.weather.truth_sources.hko import HkoTruthSource
from pmtmax.weather.truth_sources.noaa_global_hourly import NoaaGlobalHourlyTruthSource
from pmtmax.weather.truth_sources.wunderground import WundergroundTruthSource


def make_truth_source(
    spec: MarketSpec,
    http: CachedHttpClient,
    snapshot_dir: Path | None = None,
    *,
    use_cache: bool = True,
) -> TruthSource:
    """Instantiate the right official truth adapter for a market."""

    key = spec.truth_source_key()
    if key == "amo_air_calp":
        return AmoAirCalpTruthSource(http, snapshot_dir=snapshot_dir)
    if key == "noaa_global_hourly":
        return NoaaGlobalHourlyTruthSource(http, snapshot_dir=snapshot_dir)
    if key == "wunderground":
        return WundergroundTruthSource(http, snapshot_dir=snapshot_dir, use_cache=use_cache)
    if key == "hko":
        return HkoTruthSource(http, snapshot_dir=snapshot_dir, use_cache=use_cache)
    if key == "cwa":
        return CwaTruthSource(http, snapshot_dir=snapshot_dir)
    msg = f"Unsupported truth adapter for {spec.official_source_name}"
    raise ValueError(msg)
