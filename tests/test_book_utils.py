from __future__ import annotations

from datetime import UTC, datetime

from pmtmax.markets.book_utils import fetch_book
from pmtmax.storage.schemas import MarketSnapshot


class _RaisingClob:
    def get_book(self, token_id: str) -> dict:
        raise RuntimeError(f"book unavailable for {token_id}")


def test_fetch_book_returns_missing_by_default_on_clob_error() -> None:
    snapshot = MarketSnapshot(captured_at=datetime.now(tz=UTC), market={"id": "market-1"})

    book = fetch_book(_RaisingClob(), snapshot, "token-1", "11°C")

    assert book.source == "missing"
    assert book.bids == []
    assert book.asks == []


def test_fetch_book_can_opt_into_synthetic_fallback() -> None:
    snapshot = MarketSnapshot(
        captured_at=datetime.now(tz=UTC),
        market={"id": "market-1"},
        outcome_prices={"11°C": 0.42},
    )

    book = fetch_book(
        _RaisingClob(),
        snapshot,
        "token-1",
        "11°C",
        allow_synthetic_fallback=True,
    )

    assert book.source == "fixture"
    assert book.best_bid() > 0.0
    assert book.best_ask() < 1.0
