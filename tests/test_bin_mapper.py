from datetime import date

import numpy as np

from pmtmax.markets.market_spec import FinalizationPolicy, MarketSpec, OutcomeBin, PrecisionRule
from pmtmax.modeling.bin_mapper import (
    infer_winning_label,
    map_normal_to_outcomes,
    map_samples_to_outcomes,
)


def _spec_celsius() -> MarketSpec:
    return MarketSpec(
        market_id="1",
        slug="slug",
        question="Highest temperature in Seoul on December 11?",
        city="Seoul",
        target_local_date=date(2025, 12, 11),
        timezone="Asia/Seoul",
        official_source_name="Wunderground",
        official_source_url="https://www.wunderground.com/history/daily/kr/incheon/RKSI",
        station_id="RKSI",
        station_name="Incheon Intl Airport",
        unit="C",
        precision_rule=PrecisionRule(unit="C", step=1.0, source_precision_text="whole degree"),
        outcome_schema=[
            OutcomeBin(label="4°C or below", upper=4),
            OutcomeBin(label="5°C", lower=5, upper=5),
            OutcomeBin(label="6°C", lower=6, upper=6),
            OutcomeBin(label="7°C", lower=7, upper=7),
            OutcomeBin(label="8°C", lower=8, upper=8),
            OutcomeBin(label="9°C", lower=9, upper=9),
            OutcomeBin(label="10°C or higher", lower=10),
        ],
        finalization_policy=FinalizationPolicy(),
    )


def _spec_fahrenheit() -> MarketSpec:
    return MarketSpec(
        market_id="2",
        slug="slug",
        question="Highest temperature in NYC on December 20?",
        city="NYC",
        target_local_date=date(2025, 12, 20),
        timezone="America/New_York",
        official_source_name="Wunderground",
        official_source_url="https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA",
        station_id="KLGA",
        station_name="LaGuardia Airport",
        unit="F",
        precision_rule=PrecisionRule(unit="F", step=2.0, rounding="range_bin", source_precision_text="whole degree"),
        outcome_schema=[
            OutcomeBin(label="34°F or below", upper=34),
            OutcomeBin(label="35-36°F", lower=35, upper=36),
            OutcomeBin(label="37-38°F", lower=37, upper=38),
            OutcomeBin(label="39-40°F", lower=39, upper=40),
            OutcomeBin(label="41°F or higher", lower=41),
        ],
        finalization_policy=FinalizationPolicy(),
    )


def test_map_normal_to_celsius_bins_sums_to_one() -> None:
    probabilities = map_normal_to_outcomes(_spec_celsius(), mean=8.0, std=0.8)
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
    assert probabilities["8°C"] > probabilities["5°C"]


def test_map_samples_to_fahrenheit_ranges() -> None:
    samples = np.array([35.2, 35.7, 37.0, 39.9, 41.3])
    probabilities = map_samples_to_outcomes(_spec_fahrenheit(), samples)
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
    assert probabilities["35-36°F"] > 0.0
    assert probabilities["41°F or higher"] > 0.0


def test_infer_winning_label() -> None:
    assert infer_winning_label(_spec_celsius(), 8.0) == "8°C"

