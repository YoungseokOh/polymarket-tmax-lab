from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pmtmax.examples import EXAMPLE_MARKETS, example_market_specs
from pmtmax.markets.rule_parser import parse_market_spec
from pmtmax.weather.truth_sources import make_truth_source
from pmtmax.weather.truth_sources.amo_air_calp import AmoAirCalpTruthSource
from pmtmax.weather.truth_sources.base import TruthSourceLagError
from pmtmax.weather.truth_sources.noaa_global_hourly import NoaaGlobalHourlyTruthSource
from pmtmax.weather.truth_sources.wunderground import WundergroundTruthSource


class _FakeHttp:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_json(self, url: str, params: dict[str, object] | None = None, use_cache: bool = True) -> list[dict[str, str]]:
        self.calls.append({"url": url, "params": params, "use_cache": use_cache})
        return [
            {"DATE": "2025-12-19T05:00:00", "TMP": "+0000,1"},
            {"DATE": "2025-12-20T11:00:00", "TMP": "+0022,1"},
            {"DATE": "2025-12-20T16:00:00", "TMP": "+0039,1"},
            {"DATE": "2025-12-20T20:00:00", "TMP": "+0031,1"},
            {"DATE": "2025-12-21T02:00:00", "TMP": "+0010,1"},
            {"DATE": "2025-12-21T05:00:00", "TMP": "+0001,1"},
        ]


class _LaggingHttp:
    def get_json(self, url: str, params: dict[str, object] | None = None, use_cache: bool = True) -> object:  # noqa: ARG002
        if "search/v1/data" in url:
            return {"endDate": "2025-08-24T21:59:59"}
        return []


def _toronto_spec():
    market = {
        **EXAMPLE_MARKETS["Seoul"],
        "id": "example-toronto",
        "slug": "highest-temperature-in-toronto-on-march-21-2026",
        "question": "Highest temperature in Toronto on March 21?",
    }
    description = (
        "This market will resolve to the temperature range that contains the highest temperature recorded at the "
        "Toronto Pearson Intl Airport Station in degrees Celsius on 21 Mar '26. "
        "The resolution source for this market will be information from Wunderground, specifically the highest "
        "temperature recorded for all times on this day by the Forecast for the Toronto Pearson Intl Airport "
        "Station once information is finalized, available here: "
        "https://www.wunderground.com/history/daily/ca/mississauga/CYYZ. "
        'This market can not resolve to "Yes" until all data for this date has been finalized. '
        "The resolution source for this market measures temperatures to whole degrees Celsius (eg, 9°C). "
        "Any revisions to temperatures recorded after data is finalized for this market's timeframe will not be considered."
    )
    return parse_market_spec(description, market=market)


def test_make_truth_source_routes_seoul_markets_to_amo_public_archive() -> None:
    spec = example_market_specs(["Seoul"])[0]
    source = make_truth_source(spec, _FakeHttp(), snapshot_dir=Path("tests/fixtures/truth"))  # type: ignore[arg-type]

    assert isinstance(source, AmoAirCalpTruthSource)


def test_make_truth_source_routes_nyc_markets_to_wunderground_public_archive() -> None:
    spec = example_market_specs(["NYC"])[0]
    source = make_truth_source(spec, _FakeHttp(), snapshot_dir=Path("tests/fixtures/truth"))  # type: ignore[arg-type]

    assert isinstance(source, WundergroundTruthSource)


def test_make_truth_source_routes_london_markets_to_wunderground_public_archive() -> None:
    market = {
        **EXAMPLE_MARKETS["Seoul"],
        "id": "example-london",
        "slug": "highest-temperature-in-london-on-march-18-2026",
        "question": "Highest temperature in London on March 18?",
    }
    description = Path("tests/fixtures/markets/london_rules.txt").read_text()
    spec = parse_market_spec(description, market=market)
    source = make_truth_source(spec, _FakeHttp(), snapshot_dir=Path("tests/fixtures/truth"))  # type: ignore[arg-type]

    assert isinstance(source, WundergroundTruthSource)


def test_make_truth_source_routes_toronto_markets_to_wunderground() -> None:
    # Toronto was migrated from NOAA Global Hourly to Wunderground Public Historical API
    spec = _toronto_spec()
    source = make_truth_source(spec, _FakeHttp(), snapshot_dir=Path("tests/fixtures/truth"))  # type: ignore[arg-type]

    assert isinstance(source, WundergroundTruthSource)


def test_noaa_truth_source_uses_local_station_snapshot_when_available() -> None:
    spec = example_market_specs(["Seoul"])[0]
    source = NoaaGlobalHourlyTruthSource(_FakeHttp(), snapshot_dir=Path("tests/fixtures/truth"))  # type: ignore[arg-type]

    bundle = source.fetch_observation_bundle(spec, date(2025, 12, 11))

    assert bundle.observation.source == "NOAA Global Hourly"
    assert bundle.observation.daily_max == pytest.approx(8.7)
    assert bundle.observation.unit == "C"


def test_noaa_truth_source_fetches_hourly_rows_and_converts_to_market_unit() -> None:
    spec = example_market_specs(["NYC"])[0].model_copy(
        update={
            "public_truth_source_name": "NOAA Global Hourly",
            "public_truth_station_id": "72503014732",
        }
    )
    http = _FakeHttp()
    source = NoaaGlobalHourlyTruthSource(http)  # type: ignore[arg-type]

    bundle = source.fetch_observation_bundle(spec, date(2025, 12, 20))

    assert bundle.observation.daily_max == pytest.approx(39.02, abs=0.05)
    assert bundle.observation.unit == "F"
    assert len(http.calls) == 1
    assert http.calls[0]["params"] == {
        "dataset": "global-hourly",
        "stations": "72503014732",
        "startDate": "2025-12-19",
        "endDate": "2025-12-21",
        "format": "json",
        "includeAttributes": "false",
    }


def test_noaa_truth_source_raises_lag_error_when_archive_lags_target_date() -> None:
    spec = example_market_specs(["Seoul"])[0].model_copy(update={"target_local_date": date(2025, 9, 1)})
    source = NoaaGlobalHourlyTruthSource(_LaggingHttp())  # type: ignore[arg-type]

    with pytest.raises(TruthSourceLagError, match="latest available date.*2025-08-24"):
        source.fetch_observation_bundle(spec, date(2025, 9, 1))
