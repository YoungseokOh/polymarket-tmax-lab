"""Shared order-book utilities extracted from CLI."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pmtmax.markets.clob_read_client import ClobReadClient
from pmtmax.storage.schemas import BookLevel, BookSnapshot, MarketSnapshot


def synthetic_book(snapshot: MarketSnapshot, outcome_label: str, token_id: str) -> BookSnapshot:
    """Build a synthetic single-level book from Gamma-API outcome prices."""

    price = snapshot.outcome_prices.get(outcome_label, 0.5)
    half_spread = max(0.03, price * 0.10)
    bid = max(price - half_spread, 0.01)
    ask = min(price + half_spread, 0.99)
    liquidity = max(price * 50.0, 5.0)
    return BookSnapshot(
        market_id=snapshot.spec.market_id if snapshot.spec is not None else str(snapshot.market.get("id")),
        token_id=token_id,
        outcome_label=outcome_label,
        source="fixture",
        timestamp=snapshot.captured_at,
        bids=[BookLevel(price=bid, size=liquidity)],
        asks=[BookLevel(price=ask, size=liquidity)],
    )


def book_snapshot_from_payload(
    *,
    snapshot: MarketSnapshot,
    token_id: str,
    outcome_label: str,
    payload: dict[str, Any] | None,
) -> BookSnapshot:
    """Parse a CLOB book payload into a BookSnapshot, falling back to synthetic."""

    if payload is None:
        return synthetic_book(snapshot, outcome_label, token_id)
    bids = [BookLevel(price=float(level["price"]), size=float(level["size"])) for level in payload.get("bids", [])[:5]]
    asks = [BookLevel(price=float(level["price"]), size=float(level["size"])) for level in payload.get("asks", [])[:5]]
    timestamp = payload.get("timestamp")
    parsed_ts = None
    if timestamp:
        try:
            parsed_ts = datetime.fromtimestamp(int(str(timestamp)) / 1000.0, tz=UTC)
        except ValueError:
            parsed_ts = None
    return BookSnapshot(
        market_id=snapshot.spec.market_id if snapshot.spec is not None else str(snapshot.market.get("id")),
        token_id=token_id,
        outcome_label=outcome_label,
        source="clob",
        timestamp=parsed_ts,
        bids=bids,
        asks=asks,
    )


def fetch_book(
    clob: ClobReadClient,
    snapshot: MarketSnapshot,
    token_id: str,
    outcome_label: str,
) -> BookSnapshot:
    """Fetch a live order book, falling back to synthetic on error."""

    try:
        payload = clob.get_book(token_id)
    except Exception:  # noqa: BLE001
        payload = None
    return book_snapshot_from_payload(snapshot=snapshot, token_id=token_id, outcome_label=outcome_label, payload=payload)
