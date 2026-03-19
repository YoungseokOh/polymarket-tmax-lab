from __future__ import annotations

from datetime import date

from pmtmax.examples import example_market_specs
from pmtmax.weather.truth_sources.cwa import CwaTruthSource


class _FakeHttp:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def post_json(self, url: str, data: dict[str, object] | None = None, use_cache: bool = True) -> dict:
        self.calls.append({"url": url, "data": data, "use_cache": use_cache})
        return {
            "code": 200,
            "message": "",
            "data": [
                {
                    "StationID": "466920",
                    "dts": [
                        {
                            "DataDate": "2026-03-16T00:00:00",
                            "AirTemperature": {"Maximum": 24.8},
                        },
                        {
                            "DataDate": "2026-03-17T00:00:00",
                            "AirTemperature": {"Maximum": 26.4},
                        },
                    ],
                }
            ],
        }


def test_cwa_truth_source_uses_codis_override_when_no_snapshot_is_available() -> None:
    spec = example_market_specs(["Taipei"])[0].model_copy(update={"target_local_date": date(2026, 3, 17)})
    http = _FakeHttp()
    source = CwaTruthSource(http)  # type: ignore[arg-type]

    bundle = source.fetch_observation_bundle(spec, date(2026, 3, 17))

    assert bundle.observation.daily_max == 26.4
    assert bundle.media_type == "application/json"
    assert len(http.calls) == 1
    assert http.calls[0]["url"] == "https://codis.cwa.gov.tw/api/station"
    assert http.calls[0]["data"] == {
        "type": "report_month",
        "stn_type": "cwb",
        "stn_ID": "466920",
        "more": "",
        "start": "2026-03-01T00:00:00",
        "end": "2026-03-31T00:00:00",
    }
