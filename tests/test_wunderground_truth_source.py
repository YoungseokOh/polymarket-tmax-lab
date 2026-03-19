from __future__ import annotations

from datetime import date
from pathlib import Path

from pmtmax.examples import example_market_specs
from pmtmax.weather.truth_sources.wunderground import WundergroundTruthSource


class _FakeHttp:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_json(self, url: str, params: dict[str, object] | None = None, use_cache: bool = True) -> dict:
        self.calls.append({"url": url, "params": params, "use_cache": use_cache})
        return {
            "metadata": {"location_id": "RKSI:9:KR", "units": "m"},
            "observations": [
                {"valid_time_gmt": 1773673200, "temp": 3},
                {"valid_time_gmt": 1773716400, "temp": 11},
                {"valid_time_gmt": 1773723600, "temp": 12},
                {"valid_time_gmt": 1773757800, "temp": 5},
            ],
        }


def test_wunderground_truth_source_uses_cached_html_snapshot_when_available() -> None:
    spec = example_market_specs(["NYC"])[0]
    source = WundergroundTruthSource(_FakeHttp(), snapshot_dir=Path("tests/fixtures/truth"))  # type: ignore[arg-type]

    bundle = source.fetch_observation_bundle(spec, date(2025, 12, 20))

    assert bundle.observation.daily_max == 39
    assert bundle.media_type == "text/html"


def test_wunderground_truth_source_fetches_official_historical_api_for_live_data() -> None:
    spec = example_market_specs(["Seoul"])[0].model_copy(update={"target_local_date": date(2026, 3, 17)})
    http = _FakeHttp()
    source = WundergroundTruthSource(http)  # type: ignore[arg-type]

    bundle = source.fetch_observation_bundle(spec, date(2026, 3, 17))

    assert bundle.observation.daily_max == 12
    assert bundle.media_type == "application/json"
    assert len(http.calls) == 1
    assert http.calls[0]["url"] == "https://api.weather.com/v1/location/RKSI:9:KR/observations/historical.json"
    assert http.calls[0]["params"] == {
        "apiKey": "e1f10a1e78da46f5b10a1e78da96f525",
        "units": "m",
        "startDate": "20260317",
        "endDate": "20260317",
    }
