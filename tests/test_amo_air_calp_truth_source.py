from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pmtmax.examples import example_market_specs
from pmtmax.weather.truth_sources.amo_air_calp import AmoAirCalpTruthSource
from pmtmax.weather.truth_sources.base import TruthSourceLagError


class _FakeHttp:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def get_text(self, url: str, params: dict[str, object] | None = None, use_cache: bool = True) -> str:
        self.calls.append({"url": url, "params": params, "use_cache": use_cache})
        return self.payload


def test_amo_air_calp_truth_source_uses_monthly_snapshot_when_available() -> None:
    spec = example_market_specs(["Seoul"])[0]
    source = AmoAirCalpTruthSource(_FakeHttp("unused"), snapshot_dir=Path("tests/fixtures/truth"))  # type: ignore[arg-type]

    bundle = source.fetch_observation_bundle(spec, date(2025, 12, 11))

    assert bundle.observation.source == "Aviation Meteorological Office AIR_CALP"
    assert bundle.observation.daily_max == pytest.approx(8.7)
    assert bundle.observation.unit == "C"
    assert bundle.media_type == "text/csv"


def test_amo_air_calp_truth_source_fetches_monthly_csv_for_missing_snapshot() -> None:
    spec = example_market_specs(["Seoul"])[0]
    http = _FakeHttp(
        '"TM","STN_ID","TMP_MNM_TM","TMP_MNM","TMP_MAX_TM","TMP_MAX"\n'
        '"20260317","113","605","31","1412","121"\n'
    )
    source = AmoAirCalpTruthSource(http)  # type: ignore[arg-type]

    bundle = source.fetch_observation_bundle(spec.model_copy(update={"target_local_date": date(2026, 3, 17)}), date(2026, 3, 17))

    assert bundle.observation.daily_max == pytest.approx(12.1)
    assert http.calls == [
        {
            "url": "http://amoapi.kma.go.kr/amoApi/air_calp",
            "params": {"icao": "RKSI", "yyyymm": "202603"},
            "use_cache": True,
        }
    ]


def test_amo_air_calp_truth_source_raises_lag_error_when_monthly_csv_lags_target_date() -> None:
    spec = example_market_specs(["Seoul"])[0].model_copy(update={"target_local_date": date(2026, 3, 17)})
    http = _FakeHttp(
        '"TM","STN_ID","TMP_MNM_TM","TMP_MNM","TMP_MAX_TM","TMP_MAX"\n'
        '"20260301","113","58","30","1504","117"\n'
        '"20260302","113","2223","14","1","60"\n'
    )
    source = AmoAirCalpTruthSource(http)  # type: ignore[arg-type]

    with pytest.raises(TruthSourceLagError, match="latest available date.*2026-03-02"):
        source.fetch_observation_bundle(spec, date(2026, 3, 17))
