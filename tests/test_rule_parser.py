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
    assert spec.truth_track == "research_public"
    assert spec.settlement_eligible is False
    assert spec.public_truth_source_name == "Aviation Meteorological Office AIR_CALP"
    assert spec.public_truth_station_id == "RKSI"
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
    assert spec.truth_track == "exact_public"
    assert spec.settlement_eligible is True


def test_parse_taipei_rules() -> None:
    spec = parse_market_spec(_read_fixture("taipei_rules.txt"), market=EXAMPLE_MARKETS["Taipei"])
    assert spec.city == "Taipei"
    assert spec.official_source_name == "Central Weather Administration"
    assert spec.station_id == "466920"
    assert spec.timezone == "Asia/Taipei"
    assert spec.truth_track == "exact_public"


def test_parse_toronto_rules_from_inline_market_description() -> None:
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
    spec = parse_market_spec(description, market=market)

    assert spec.city == "Toronto"
    assert spec.station_id == "CYYZ"
    assert spec.timezone == "America/Toronto"
    assert spec.truth_track == "research_public"
    assert spec.public_truth_station_id == "71624099999"


def test_parse_noaa_timeseries_rules_from_inline_market_description() -> None:
    market = {
        **EXAMPLE_MARKETS["Seoul"],
        "id": "example-tel-aviv",
        "slug": "highest-temperature-in-tel-aviv-on-march-23-2026",
        "question": "Highest temperature in Tel Aviv on March 23?",
    }
    description = (
        "This market will resolve to the temperature range that contains the highest temperature recorded by NOAA "
        "at the Ben Gurion International Airport in degrees Celsius on 23 Mar '26. "
        "The resolution source for this market will be information from NOAA, specifically the highest reading "
        'under the "Temp" column on the specified date once information is finalized for all hours on that date, '
        "available here: https://www.weather.gov/wrh/timeseries?site=LLBG "
        'This market can not resolve to "Yes" until data for this date has been finalized. '
        "The resolution source for this market measures temperatures to whole degrees Celsius (eg, 9°C). "
        "Any revisions to temperatures recorded after data is finalized for this market's timeframe will not be considered."
    )
    spec = parse_market_spec(description, market=market)

    assert spec.city == "Tel Aviv"
    assert spec.official_source_name == "NOAA Timeseries"
    assert spec.station_id == "LLBG"
    assert spec.timezone == "Asia/Jerusalem"
    assert spec.truth_track == "exact_public"
    assert spec.public_truth_station_id == "40180099999"
