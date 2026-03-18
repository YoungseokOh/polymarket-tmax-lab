from datetime import date

from pmtmax.markets.market_spec import FinalizationPolicy, MarketSpec, OutcomeBin, PrecisionRule


def test_market_spec_adapter_key() -> None:
    spec = MarketSpec(
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
        outcome_schema=[OutcomeBin(label="8°C", lower=8, upper=8)],
        finalization_policy=FinalizationPolicy(),
    )
    assert spec.adapter_key() == "wunderground"


def test_outcome_bin_contains() -> None:
    outcome = OutcomeBin(label="70-71°F", lower=70, upper=71)
    assert outcome.contains(70)
    assert outcome.contains(71)
    assert not outcome.contains(72)

