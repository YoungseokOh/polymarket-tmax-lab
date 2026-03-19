from __future__ import annotations

from datetime import date

from pmtmax.examples import example_market_specs
from pmtmax.weather.truth_sources.hko import HkoTruthSource


class _FakeHttp:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_json(self, url: str, params: dict[str, object] | None = None, use_cache: bool = True) -> dict:
        self.calls.append({"url": url, "params": params, "use_cache": use_cache})
        if params and params.get("year") == 2026:
            return {"data": []}
        return {
            "data": [
                ["2026", "3", "16", "20.1", "C"],
                ["2026", "3", "17", "21.7", "C"],
            ]
        }


def test_hko_truth_source_falls_back_to_full_station_payload_when_month_is_empty() -> None:
    spec = example_market_specs(["Hong Kong"])[0].model_copy(update={"target_local_date": date(2026, 3, 17)})
    http = _FakeHttp()
    source = HkoTruthSource(http)  # type: ignore[arg-type]

    bundle = source.fetch_observation_bundle(spec, date(2026, 3, 17))

    assert bundle.observation.daily_max == 21.7
    assert len(http.calls) == 2
    assert http.calls[0]["params"] == {
        "dataType": "CLMMAXT",
        "station": "HKA",
        "year": 2026,
        "month": 3,
        "rformat": "json",
        "lang": "en",
    }
    assert http.calls[1]["params"] == {
        "dataType": "CLMMAXT",
        "station": "HKA",
        "rformat": "json",
        "lang": "en",
    }
