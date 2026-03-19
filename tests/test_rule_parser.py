from pathlib import Path

from pmtmax.examples import EXAMPLE_MARKETS
from pmtmax.markets.rule_parser import parse_market_spec


def _read_fixture(name: str) -> str:
    return Path("tests/fixtures/markets").joinpath(name).read_text()


def test_parse_seoul_rules() -> None:
    spec = parse_market_spec(_read_fixture("seoul_rules.txt"), market=EXAMPLE_MARKETS["Seoul"])
    assert spec.city == "Seoul"
    assert spec.station_name == "Incheon Intl Airport"
    assert spec.station_id == "RKSI"
    assert spec.official_source_name == "Wunderground"
    assert spec.unit == "C"
    assert spec.timezone == "Asia/Seoul"
    assert spec.station_lat is not None
    assert spec.finalization_policy.wait_for_finalized_data
    assert spec.finalization_policy.ignore_post_final_revision


def test_parse_nyc_rules() -> None:
    spec = parse_market_spec(_read_fixture("nyc_rules.txt"), market=EXAMPLE_MARKETS["NYC"])
    assert spec.city == "NYC"
    assert spec.station_name == "LaGuardia Airport"
    assert spec.station_id == "KLGA"
    assert spec.unit == "F"
    assert spec.timezone == "America/New_York"
    assert spec.precision_rule.step == 2.0


def test_parse_london_rules() -> None:
    market = {
        **EXAMPLE_MARKETS["Seoul"],
        "id": "example-london",
        "slug": "highest-temperature-in-london-on-march-18-2026",
        "question": "Highest temperature in London on March 18?",
    }
    spec = parse_market_spec(_read_fixture("london_rules.txt"), market=market)
    assert spec.city == "London"
    assert spec.station_name == "London City Airport"
    assert spec.station_id == "EGLC"
    assert spec.timezone == "Europe/London"
    assert spec.station_lat is not None
    assert spec.official_source_url == "https://www.wunderground.com/history/daily/gb/london/EGLC"


def test_parse_hko_rules() -> None:
    spec = parse_market_spec(_read_fixture("hk_rules.txt"), market=EXAMPLE_MARKETS["Hong Kong"])
    assert spec.city == "Hong Kong"
    assert spec.official_source_name == "Hong Kong Observatory Daily Extract"
    assert spec.station_id == "HKA"
    assert spec.station_name == "Hong Kong International Airport"


def test_parse_taipei_rules() -> None:
    spec = parse_market_spec(_read_fixture("taipei_rules.txt"), market=EXAMPLE_MARKETS["Taipei"])
    assert spec.city == "Taipei"
    assert spec.official_source_name == "Central Weather Administration"
    assert spec.station_id == "466920"
    assert spec.timezone == "Asia/Taipei"
