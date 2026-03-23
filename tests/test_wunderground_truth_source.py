from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

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

    def get_text(self, url: str, params: dict[str, object] | None = None, use_cache: bool = True) -> str:  # noqa: ARG002
        return "<html>fixture</html>"


class _BrokenHtmlHttp(_FakeHttp):
    def get_text(self, url: str, params: dict[str, object] | None = None, use_cache: bool = True) -> str:  # noqa: ARG002
        return "<html>missing temperatureMax</html>"


class _PublicApiKeyHttp(_FakeHttp):
    def get_text(self, url: str, params: dict[str, object] | None = None, use_cache: bool = True) -> str:  # noqa: ARG002
        return '<html><script>var url="https://api.weather.com/v3/wx/observations/current?apiKey=abcdef0123456789abcdef0123456789";</script></html>'


def test_wunderground_truth_source_uses_cached_html_snapshot_when_available() -> None:
    spec = example_market_specs(["NYC"])[0]
    source = WundergroundTruthSource(_FakeHttp(), snapshot_dir=Path("tests/fixtures/truth"))  # type: ignore[arg-type]

    bundle = source.fetch_observation_bundle(spec, date(2025, 12, 20))

    assert bundle.observation.daily_max == 39
    assert bundle.media_type == "text/html"


def test_wunderground_truth_source_fetches_official_historical_api_for_live_data() -> None:
    spec = example_market_specs(["Seoul"])[0].model_copy(update={"target_local_date": date(2026, 3, 17)})
    http = _FakeHttp()
    source = WundergroundTruthSource(http, api_key="test-weathercom-key")  # type: ignore[arg-type]

    bundle = source.fetch_observation_bundle(spec, date(2026, 3, 17))

    assert bundle.observation.daily_max == 12
    assert bundle.media_type == "application/json"
    assert len(http.calls) == 1
    assert http.calls[0]["url"] == "https://api.weather.com/v1/location/RKSI:9:KR/observations/historical.json"
    assert http.calls[0]["params"] == {
        "apiKey": "test-weathercom-key",
        "units": "m",
        "startDate": "20260317",
        "endDate": "20260317",
    }


def test_wunderground_truth_source_discovers_public_frontend_api_key() -> None:
    spec = example_market_specs(["NYC"])[0].model_copy(update={"target_local_date": date(2026, 3, 17)})
    http = _PublicApiKeyHttp()
    source = WundergroundTruthSource(http)  # type: ignore[arg-type]

    bundle = source.fetch_observation_bundle(spec, date(2026, 3, 17))

    assert bundle.observation.daily_max == 12
    assert bundle.media_type == "application/json"
    assert len(http.calls) == 1
    assert http.calls[0]["params"] == {
        "apiKey": "abcdef0123456789abcdef0123456789",
        "units": "e",
        "startDate": "20260317",
        "endDate": "20260317",
    }


def test_wunderground_truth_source_surfaces_api_key_guidance_when_html_fallback_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PMTMAX_WU_API_KEY", raising=False)
    spec = example_market_specs(["Seoul"])[0].model_copy(update={"target_local_date": date(2026, 3, 17)})
    source = WundergroundTruthSource(_BrokenHtmlHttp())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="PMTMAX_WU_API_KEY"):
        source.fetch_observation_bundle(spec, date(2026, 3, 17))
