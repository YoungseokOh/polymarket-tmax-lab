"""Helpers for bundled and scanned market snapshots."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pmtmax.examples import EXAMPLE_MARKETS
from pmtmax.markets.normalization import (
    extract_clob_token_ids,
    extract_outcome_prices,
)
from pmtmax.markets.rule_parser import parse_market_spec
from pmtmax.storage.schemas import MarketSnapshot
from pmtmax.utils import dump_json, load_json


def snapshot_from_market(market: dict[str, Any], *, captured_at: datetime | None = None) -> MarketSnapshot:
    """Build a parsed snapshot from a raw market payload."""

    captured = captured_at or datetime.now(tz=UTC)
    outcome_prices = extract_outcome_prices(market)
    clob_token_ids = extract_clob_token_ids(market)
    try:
        spec = parse_market_spec(market.get("description", ""), market=market)
        return MarketSnapshot(
            captured_at=captured,
            market=market,
            spec=spec,
            outcome_prices=outcome_prices,
            clob_token_ids=clob_token_ids,
        )
    except Exception as exc:  # noqa: BLE001
        return MarketSnapshot(
            captured_at=captured,
            market=market,
            spec=None,
            parse_error=str(exc),
            outcome_prices=outcome_prices,
            clob_token_ids=clob_token_ids,
        )


def bundled_market_snapshots(cities: list[str] | None = None) -> list[MarketSnapshot]:
    """Return reproducible historical market snapshots for bundled city templates."""

    selected = cities or list(EXAMPLE_MARKETS)
    captured_at = datetime(2026, 1, 1, tzinfo=UTC)
    return [snapshot_from_market(EXAMPLE_MARKETS[city], captured_at=captured_at) for city in selected]


def save_market_snapshots(path: Path, snapshots: list[MarketSnapshot]) -> Path:
    """Serialize market snapshots to disk."""

    dump_json(path, [snapshot.model_dump(mode="json") for snapshot in snapshots])
    return path


def load_market_snapshots(path: Path) -> list[MarketSnapshot]:
    """Load serialized market snapshots from disk."""

    payload = load_json(path)
    return [MarketSnapshot.model_validate(item) for item in payload]
