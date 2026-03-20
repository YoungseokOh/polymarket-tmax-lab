"""Deterministic rule parser for maximum-temperature markets."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Any, cast

from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from pmtmax.markets.market_spec import FinalizationPolicy, MarketSpec, PrecisionRule
from pmtmax.markets.normalization import extract_clob_token_ids, extract_outcome_labels
from pmtmax.markets.outcome_schema import infer_unit_from_label, parse_outcome_schema
from pmtmax.markets.station_registry import (
    canonical_city,
    lookup_station,
    lookup_station_by_station_id,
)

QUESTION_RE = re.compile(
    r"highest temperature in (?P<city>.+?) on (?P<date>[A-Za-z0-9 ,'-]+)\?",
    re.I,
)
RULE_DATE_RE = re.compile(r"on\s+(?P<date>\d{1,2}\s+[A-Za-z]{3}\s+'?\d{2,4})", re.I)
STATION_RE = re.compile(r"recorded (?:at|by NOAA at) the (?P<station>.+?)(?: Station|\s+in degrees)", re.I)
SOURCE_URL_RE = re.compile(r"https?://\S+")
FINALIZED_RE = re.compile(r"can not resolve to \"yes\" until all data .* finalized", re.I)
REVISION_RE = re.compile(r"revisions .* after data is finalized .* will not be considered", re.I)
WHOLE_DEGREE_RE = re.compile(r"measures temperatures to whole degrees\s*(?P<unit>Celsius|Fahrenheit)", re.I)
HKO_STATION_RE = re.compile(r"station\s+(?P<station>[A-Z]{2,4})", re.I)
CWA_STATION_RE = re.compile(r"station\s+(?P<station>\d{5,6})", re.I)
NOAA_SITE_RE = re.compile(r"[?&]site=(?P<station>[A-Z0-9]{4,6})", re.I)


def extract_rule_text(html_or_text: str) -> str:
    """Extract raw rule text from either HTML or plain text."""

    if "<html" not in html_or_text.lower() and "<body" not in html_or_text.lower():
        return html_or_text.strip()
    soup = BeautifulSoup(html_or_text, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return text


def _extract_question(raw_text: str, market: dict[str, Any] | None = None) -> str:
    if market and market.get("question"):
        return str(market["question"])
    for line in raw_text.splitlines():
        if "highest temperature in" in line.lower():
            return line.strip()
    msg = "Could not find market question"
    raise ValueError(msg)


def _parse_target_date(question: str, raw_text: str) -> date:
    match = QUESTION_RE.search(question)
    if not match:
        msg = f"Unsupported question format: {question}"
        raise ValueError(msg)

    question_date = match.group("date")
    if any(char.isdigit() for char in question_date) and any(char.isalpha() for char in question_date):
        if year_match := RULE_DATE_RE.search(raw_text):
            raw_date = year_match.group("date").replace("'", "20")
        else:
            raw_date = question_date.replace("'", "20")
    else:
        if year_match := RULE_DATE_RE.search(raw_text):
            raw_date = year_match.group("date").replace("'", "20")
        else:
            raw_date = question_date.replace("'", "20")
    parsed = cast(datetime, date_parser.parse(raw_date, fuzzy=True))
    return parsed.date()


def _parse_city(question: str) -> str:
    match = QUESTION_RE.search(question)
    if not match:
        msg = f"Unsupported question format: {question}"
        raise ValueError(msg)
    city = match.group("city").strip()
    return canonical_city(city)


def _detect_source(raw_text: str) -> tuple[str, str]:
    lowered = raw_text.lower()
    if "wunderground" in lowered:
        source_name = "Wunderground"
    elif "information from noaa" in lowered or "weather.gov/wrh/timeseries" in lowered:
        source_name = "NOAA Timeseries"
    elif "hong kong observatory" in lowered:
        source_name = "Hong Kong Observatory Daily Extract"
    elif "central weather administration" in lowered or "cwa" in lowered:
        source_name = "Central Weather Administration"
    else:
        msg = "Unsupported official source"
        raise ValueError(msg)
    url_match = SOURCE_URL_RE.search(raw_text)
    return source_name, url_match.group(0).rstrip(").") if url_match else ""


def _parse_station(raw_text: str, source_name: str, market: dict[str, Any] | None) -> str:
    if match := STATION_RE.search(raw_text):
        return match.group("station").strip()
    if source_name.startswith("Hong Kong Observatory") and (match := HKO_STATION_RE.search(raw_text)):
        return match.group("station").strip().upper()
    if source_name == "NOAA Timeseries" and (match := NOAA_SITE_RE.search(raw_text)):
        return match.group("station").strip().upper()
    if source_name == "Central Weather Administration" and (match := CWA_STATION_RE.search(raw_text)):
        return match.group("station").strip()
    if market and market.get("question"):
        city = canonical_city(str(market["question"]).split(" on ")[0].removeprefix("Highest temperature in ").strip())
        if definition := lookup_station(city):
            return definition.station_name
    msg = "Could not determine station name"
    raise ValueError(msg)


def _resolve_timezone(city: str, explicit_timezone: str | None, station_id: str | None = None) -> str:
    if explicit_timezone and explicit_timezone != "UTC":
        return explicit_timezone
    if station_id and (definition := lookup_station_by_station_id(station_id)):
        return definition.timezone
    if definition := lookup_station(city):
        return definition.timezone
    return explicit_timezone or "UTC"


def _resolve_station_id(
    raw_text: str,
    source_name: str,
    source_url: str,
    city: str,
    station_name: str,
) -> str:
    if source_name == "Wunderground" and source_url:
        return source_url.rstrip("/").split("/")[-1]
    if source_name == "NOAA Timeseries" and source_url:
        if station_match := NOAA_SITE_RE.search(source_url):
            return station_match.group("station").upper()
    if source_name.startswith("Hong Kong Observatory"):
        if source_url:
            station_match = re.search(r"station=([A-Z0-9]+)", source_url)
            if station_match:
                return station_match.group(1)
        if match := HKO_STATION_RE.search(raw_text):
            return match.group("station").upper()
    if source_name == "Central Weather Administration" and (match := CWA_STATION_RE.search(raw_text)):
        return match.group("station")
    if definition := lookup_station(city):
        return definition.station_id
    return station_name


def parse_market_spec(
    html_or_text: str,
    *,
    market: dict[str, Any] | None = None,
    timezone: str | None = None,
) -> MarketSpec:
    """Parse rule text and market metadata into a normalized MarketSpec."""

    raw_text = extract_rule_text(html_or_text)
    question = _extract_question(raw_text, market)
    city = _parse_city(question)
    target_date = _parse_target_date(question, raw_text)
    source_name, source_url = _detect_source(raw_text)
    station_name = _parse_station(raw_text, source_name, market)
    station_id = _resolve_station_id(raw_text, source_name, source_url, city, station_name)
    definition = lookup_station_by_station_id(station_id) or lookup_station(city)
    if source_name == "Wunderground" and definition is None:
        msg = f"Unknown Wunderground station metadata for {city} / {station_id}"
        raise ValueError(msg)
    if definition is not None:
        city = definition.city
        station_name = definition.station_name
    resolved_timezone = _resolve_timezone(city, timezone, station_id)

    token_ids = extract_clob_token_ids(market or {})
    outcome_labels = extract_outcome_labels(market or {})
    if not outcome_labels:
        outcome_labels = [line.strip() for line in raw_text.splitlines() if "°" in line]
    if not outcome_labels:
        msg = "Could not determine market outcomes"
        raise ValueError(msg)

    unit = infer_unit_from_label(outcome_labels[0])
    precision_text = "exact_source"
    if match := WHOLE_DEGREE_RE.search(raw_text):
        precision_text = match.group(0)

    precision_rule = PrecisionRule(
        unit=unit,
        step=2.0 if any("-" in label for label in outcome_labels) else 1.0,
        rounding="range_bin" if any("-" in label for label in outcome_labels) else "whole_degree",
        source_precision_text=precision_text,
    )
    finalization_policy = FinalizationPolicy(
        wait_for_finalized_data=bool(FINALIZED_RE.search(raw_text)),
        ignore_post_final_revision=bool(REVISION_RE.search(raw_text)),
        notes="Parsed from market rules",
    )

    country = definition.country if definition else _infer_country(city)
    station_lat = definition.lat if definition else None
    station_lon = definition.lon if definition else None
    truth_track = "research_public" if source_name == "Wunderground" else "exact_public"
    settlement_eligible = source_name != "Wunderground"
    public_truth_source_name = source_name
    public_truth_station_id = station_id
    research_priority = "expansion"
    if definition is not None:
        research_priority = definition.research_priority
        if source_name == "Wunderground":
            public_truth_source_name = definition.public_truth_source_name or source_name
            public_truth_station_id = definition.public_truth_station_id or station_id
        else:
            public_truth_source_name = definition.public_truth_source_name or source_name
            public_truth_station_id = definition.public_truth_station_id or station_id

    return MarketSpec(
        market_id=str(market.get("id")) if market else question,
        event_id=str(market.get("eventId")) if market and market.get("eventId") else None,
        slug=str(market.get("slug")) if market else question.lower().replace(" ", "-"),
        question=question,
        condition_id=str(market.get("conditionId")) if market and market.get("conditionId") else None,
        token_ids=token_ids,
        city=city,
        country=country,
        target_local_date=target_date,
        timezone=resolved_timezone,
        official_source_name=source_name,
        official_source_url=source_url,
        station_id=station_id,
        station_name=station_name,
        station_lat=station_lat,
        station_lon=station_lon,
        truth_track=cast(Any, truth_track),
        settlement_eligible=settlement_eligible,
        public_truth_source_name=public_truth_source_name,
        public_truth_station_id=public_truth_station_id,
        research_priority=cast(Any, research_priority),
        unit=unit,
        precision_rule=precision_rule,
        outcome_schema=parse_outcome_schema(outcome_labels),
        finalization_policy=finalization_policy,
        notes=f"Parsed at {datetime.now(tz=UTC).isoformat()}",
    )


def _infer_country(city: str) -> str:
    if definition := lookup_station(city):
        return definition.country
    return ""
