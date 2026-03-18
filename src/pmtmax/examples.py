"""Bundled historical market templates and helper data."""

from __future__ import annotations

from typing import Any

from pmtmax.markets.market_spec import MarketSpec
from pmtmax.markets.rule_parser import parse_market_spec

EXAMPLE_MARKETS: dict[str, dict[str, Any]] = {
    "Seoul": {
        "id": "example-seoul",
        "slug": "highest-temperature-in-seoul-on-december-11",
        "question": "Highest temperature in Seoul on December 11?",
        "conditionId": "0xseoul",
        "tokens": [
            {"outcome": "4°C or below", "token_id": "seoul-4"},
            {"outcome": "5°C", "token_id": "seoul-5"},
            {"outcome": "6°C", "token_id": "seoul-6"},
            {"outcome": "7°C", "token_id": "seoul-7"},
            {"outcome": "8°C", "token_id": "seoul-8"},
            {"outcome": "9°C", "token_id": "seoul-9"},
            {"outcome": "10°C or higher", "token_id": "seoul-10"},
        ],
        "outcomes": '["4°C or below", "5°C", "6°C", "7°C", "8°C", "9°C", "10°C or higher"]',
        "outcomePrices": '["0.03", "0.08", "0.14", "0.18", "0.22", "0.20", "0.15"]',
        "clobTokenIds": '["seoul-4", "seoul-5", "seoul-6", "seoul-7", "seoul-8", "seoul-9", "seoul-10"]',
        "description": (
            "This market will resolve to the temperature range that contains the highest temperature "
            "recorded at the Incheon Intl Airport Station in degrees Celsius on 11 Dec '25. "
            "The resolution source for this market will be information from Wunderground, specifically "
            "the highest temperature recorded for all times on this day by the Forecast for the Incheon "
            "Intl Airport Station once information is finalized, available here: "
            "https://www.wunderground.com/history/daily/kr/incheon/RKSI. "
            "This market can not resolve to \"Yes\" until all data for this date has been finalized. "
            "The resolution source for this market measures temperatures to whole degrees Celsius (eg, 9°C). "
            "Any revisions to temperatures recorded after data is finalized for this market's timeframe will not "
            "be considered for this market's resolution."
        ),
    },
    "NYC": {
        "id": "example-nyc",
        "slug": "highest-temperature-in-nyc-on-december-20",
        "question": "Highest temperature in NYC on December 20?",
        "conditionId": "0xnyc",
        "tokens": [
            {"outcome": "34°F or below", "token_id": "nyc-34"},
            {"outcome": "35-36°F", "token_id": "nyc-35"},
            {"outcome": "37-38°F", "token_id": "nyc-37"},
            {"outcome": "39-40°F", "token_id": "nyc-39"},
            {"outcome": "41°F or higher", "token_id": "nyc-41"},
        ],
        "outcomes": '["34°F or below", "35-36°F", "37-38°F", "39-40°F", "41°F or higher"]',
        "outcomePrices": '["0.06", "0.14", "0.24", "0.28", "0.18"]',
        "clobTokenIds": '["nyc-34", "nyc-35", "nyc-37", "nyc-39", "nyc-41"]',
        "description": (
            "This market will resolve to the temperature range that contains the highest temperature recorded at "
            "the LaGuardia Airport Station in degrees Fahrenheit on 20 Dec '25. "
            "The resolution source for this market will be information from Wunderground, specifically the highest "
            "temperature recorded for all times on this day by the Forecast for the LaGuardia Airport Station once "
            "information is finalized, available here: https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA. "
            "This market can not resolve to \"Yes\" until all data for this date has been finalized. "
            "The resolution source for this market measures temperatures to whole degrees Fahrenheit (eg, 39°F). "
            "Any revisions to temperatures recorded after data is finalized for this market's timeframe will not be considered."
        ),
    },
    "Hong Kong": {
        "id": "example-hk",
        "slug": "highest-temperature-in-hong-kong-on-march-13",
        "question": "Highest temperature in Hong Kong on March 13?",
        "conditionId": "0xhk",
        "tokens": [
            {"outcome": "18°C or below", "token_id": "hk-18"},
            {"outcome": "19°C", "token_id": "hk-19"},
            {"outcome": "20°C", "token_id": "hk-20"},
            {"outcome": "21°C or higher", "token_id": "hk-21"},
        ],
        "outcomes": '["18°C or below", "19°C", "20°C", "21°C or higher"]',
        "outcomePrices": '["0.15", "0.32", "0.34", "0.19"]',
        "clobTokenIds": '["hk-18", "hk-19", "hk-20", "hk-21"]',
        "description": (
            "This market will resolve to the temperature range that contains the highest temperature recorded at the "
            "Hong Kong International Airport Station in degrees Celsius on 13 Mar '25. "
            "The resolution source for this market will be the Hong Kong Observatory Daily Extract, specifically the "
            "Daily Maximum Temperature (CLMMAXT) series for station HKA, available here: "
            "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&station=HKA&rformat=json&lang=en. "
            "This market can not resolve to \"Yes\" until all data for this date has been finalized. "
            "The resolution source for this market measures temperatures to one decimal place Celsius; markets resolve "
            "to whole-degree Celsius bins using the official finalized value. Any revisions to temperatures recorded "
            "after data is finalized for this market's timeframe will not be considered."
        ),
    },
    "Taipei": {
        "id": "example-tpe",
        "slug": "highest-temperature-in-taipei-on-march-13",
        "question": "Highest temperature in Taipei on March 13?",
        "conditionId": "0xtpe",
        "tokens": [
            {"outcome": "21°C or below", "token_id": "tpe-21"},
            {"outcome": "22°C", "token_id": "tpe-22"},
            {"outcome": "23°C", "token_id": "tpe-23"},
            {"outcome": "24°C or higher", "token_id": "tpe-24"},
        ],
        "outcomes": '["21°C or below", "22°C", "23°C", "24°C or higher"]',
        "outcomePrices": '["0.18", "0.31", "0.29", "0.22"]',
        "clobTokenIds": '["tpe-21", "tpe-22", "tpe-23", "tpe-24"]',
        "description": (
            "This market will resolve to the temperature range that contains the highest temperature recorded at the "
            "Taipei Station in degrees Celsius on 13 Mar '25. "
            "The resolution source for this market will be the Central Weather Administration station observations for "
            "Taipei station 466920, available from the official CODiS / station observation pages. "
            "This market can not resolve to \"Yes\" until all data for this date has been finalized. "
            "The resolution source for this market measures temperatures to one decimal place Celsius; markets resolve "
            "to whole-degree Celsius bins using the official finalized value. Any revisions to temperatures recorded "
            "after data is finalized for this market's timeframe will not be considered."
        ),
    },
}


def example_market_specs(cities: list[str] | None = None) -> list[MarketSpec]:
    """Return parsed example specs for the requested cities."""

    selected = cities or list(EXAMPLE_MARKETS)
    return [parse_market_spec(EXAMPLE_MARKETS[city]["description"], market=EXAMPLE_MARKETS[city]) for city in selected]
