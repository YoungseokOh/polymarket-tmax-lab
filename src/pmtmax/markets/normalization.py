"""Normalization helpers for Gamma-like market payloads."""

from __future__ import annotations

import json
from typing import Any


def parse_json_list(value: Any) -> list[str]:
    """Parse a list that may already be a list or a JSON-encoded string."""

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(payload, list):
            return [str(item) for item in payload]
    return []


def extract_outcome_labels(market: dict[str, Any]) -> list[str]:
    """Extract normalized outcome labels from a Gamma-like market payload."""

    if market.get("tokens"):
        return [str(token.get("outcome", "")).strip() for token in market.get("tokens", [])]
    return [label.strip() for label in parse_json_list(market.get("outcomes")) if label.strip()]


def extract_clob_token_ids(market: dict[str, Any]) -> list[str]:
    """Extract normalized token ids from a Gamma-like market payload."""

    if market.get("tokens"):
        token_ids: list[str] = []
        for token in market.get("tokens", []):
            token_id = token.get("token_id") or token.get("tokenId")
            if token_id:
                token_ids.append(str(token_id))
        return token_ids
    return parse_json_list(market.get("clobTokenIds") or market.get("tokenIds"))


def extract_outcome_prices(market: dict[str, Any]) -> dict[str, float]:
    """Extract market-implied prices keyed by outcome label."""

    labels = extract_outcome_labels(market)
    raw_prices = parse_json_list(market.get("outcomePrices"))
    prices: dict[str, float] = {}
    for label, raw_price in zip(labels, raw_prices, strict=False):
        try:
            prices[label] = float(raw_price)
        except ValueError:
            continue
    return prices
