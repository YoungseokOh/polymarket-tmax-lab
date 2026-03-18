"""Filters for Polymarket maximum temperature contracts."""

from __future__ import annotations

import re
from typing import Any

from pmtmax.markets.normalization import extract_outcome_labels

TEMP_MARKET_RE = re.compile(r"highest temperature in .+ on .+\?", re.I)
SUPPORTED_SOURCE_HINTS = (
    "wunderground",
    "hong kong observatory",
    "central weather administration",
    "highest temperature recorded",
)


def is_temperature_max_market(market: dict[str, Any]) -> bool:
    """Return whether a market matches the supported recurring family."""

    question = (market.get("question") or "").strip()
    description = (market.get("description") or "").strip()
    outcomes = extract_outcome_labels(market)
    has_temp_pattern = bool(TEMP_MARKET_RE.search(question))
    has_temp_resolution = any(hint in description.lower() for hint in SUPPORTED_SOURCE_HINTS)
    outcome_hint = any("°c" in label.lower() or "°f" in label.lower() for label in outcomes)
    return has_temp_pattern and has_temp_resolution and outcome_hint
