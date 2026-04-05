from __future__ import annotations

from datetime import UTC, date, datetime

from pmtmax.examples import example_market_specs
from pmtmax.weather.intraday_observation import fetch_intraday_observations


class _FakeIntradayHttp:
    def __init__(self, *, text_payload: str = "", json_payload: dict | None = None) -> None:
        self.text_payload = text_payload
        self.json_payload = json_payload or {}
        self.text_calls: list[dict[str, object]] = []
        self.json_calls: list[dict[str, object]] = []

    def get_text(self, url: str, params: dict[str, object] | None = None, use_cache: bool = True) -> str:
        self.text_calls.append({"url": url, "params": params, "use_cache": use_cache})
        return self.text_payload

    def post_json(self, url: str, data: dict[str, object] | None = None, use_cache: bool = True) -> dict:
        self.json_calls.append({"url": url, "data": data, "use_cache": use_cache})
        return self.json_payload


def test_fetch_intraday_observations_parses_hko_text_readings() -> None:
    spec = example_market_specs(["Hong Kong"])[0].model_copy(update={"target_local_date": date(2026, 4, 5)})
    http = _FakeIntradayHttp(
        text_payload=(
            "Latest readings recorded at 11:50 Hong Kong Time 5 April 2026\n"
            "Chek Lap Kok                   24.5       91        25.6 / 23.7       +1.2\n"
        )
    )

    observations = fetch_intraday_observations(
        spec,
        http=http,  # type: ignore[arg-type]
        observed_at=datetime(2026, 4, 5, 4, 0, tzinfo=UTC),
    )

    assert len(observations) == 1
    observation = observations[0]
    assert observation.source_family == "official_intraday"
    assert observation.observation_source == "hko_text_readings_v2"
    assert observation.station_id == "HKA"
    assert observation.current_temp_c == 24.5
    assert observation.daily_high_so_far_c == 25.6
    assert observation.lower_bound_temp_c == 25.6
    assert observation.observed_at == datetime(2026, 4, 5, 3, 50, tzinfo=UTC)


def test_fetch_intraday_observations_parses_cwa_daily_maximum() -> None:
    spec = example_market_specs(["Taipei"])[0].model_copy(update={"target_local_date": date(2026, 4, 5)})
    http = _FakeIntradayHttp(
        json_payload={
            "code": 200,
            "data": [
                {
                    "StationID": "466920",
                    "dts": [
                        {
                            "DataDate": "2026-04-05T00:00:00",
                            "AirTemperature": {
                                "Maximum": 21.5,
                                "MaximumTime": "2026-04-05T02:50:00",
                            },
                        }
                    ],
                }
            ],
        }
    )

    observations = fetch_intraday_observations(
        spec,
        http=http,  # type: ignore[arg-type]
        observed_at=datetime(2026, 4, 5, 6, 0, tzinfo=UTC),
    )

    assert len(observations) == 1
    observation = observations[0]
    assert observation.source_family == "official_intraday"
    assert observation.observation_source == "cwa_codis_report_month"
    assert observation.station_id == "466920"
    assert observation.lower_bound_temp_c == 21.5
    assert observation.daily_high_so_far_c == 21.5
    assert observation.observed_at == datetime(2026, 4, 4, 18, 50, tzinfo=UTC)


def test_fetch_intraday_observations_parses_air_calp_current_day_maximum() -> None:
    spec = example_market_specs(["Seoul"])[0].model_copy(update={"target_local_date": date(2026, 4, 5)})
    http = _FakeIntradayHttp(
        text_payload=(
            '"TM","STN_ID","TMP_MNM_TM","TMP_MNM","TMP_MAX_TM","TMP_MAX"\n'
            '"20260405","113","0605","31","1412","121"\n'
        )
    )

    observations = fetch_intraday_observations(
        spec,
        http=http,  # type: ignore[arg-type]
        observed_at=datetime(2026, 4, 5, 6, 0, tzinfo=UTC),
    )

    assert len(observations) == 1
    observation = observations[0]
    assert observation.source_family == "research_intraday"
    assert observation.observation_source == "amo_air_calp_intraday"
    assert observation.station_id == "RKSI"
    assert observation.lower_bound_temp_c == 12.1
    assert observation.daily_high_so_far_c == 12.1
    assert observation.observed_at == datetime(2026, 4, 5, 5, 12, tzinfo=UTC)
