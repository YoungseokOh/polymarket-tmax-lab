from __future__ import annotations

import pytest

from pmtmax.execution.paper_market_maker import PaperMarketMaker
from pmtmax.execution.quoter import Quote
from pmtmax.storage.schemas import BookLevel, BookSnapshot, RiskLimits


def test_paper_market_maker_tracks_realized_and_unrealized_pnl() -> None:
    mm = PaperMarketMaker(
        risk_limits=RiskLimits(
            max_position_per_outcome=100.0,
            max_total_exposure=500.0,
            max_loss=100.0,
        )
    )
    buy_quote = Quote(
        token_id="token-1",
        outcome_label="11°C",
        fair_value=0.5,
        bid_price=0.45,
        bid_size=10.0,
        ask_price=0.75,
        ask_size=10.0,
    )
    buy_book = BookSnapshot(
        market_id="market-1",
        token_id="token-1",
        outcome_label="11°C",
        source="clob",
        bids=[BookLevel(price=0.60, size=10.0)],
        asks=[BookLevel(price=0.40, size=10.0)],
    )

    fills = mm.simulate_quotes([buy_quote], {"11°C": buy_book})

    assert len(fills) == 1
    assert mm.inventory["token-1"] == pytest.approx(10.0)
    assert mm.realized_pnl == pytest.approx(0.0)
    assert mm.unrealized_pnl == pytest.approx(1.0)

    sell_quote = Quote(
        token_id="token-1",
        outcome_label="11°C",
        fair_value=0.5,
        bid_price=0.20,
        bid_size=10.0,
        ask_price=0.50,
        ask_size=10.0,
    )
    sell_book = BookSnapshot(
        market_id="market-1",
        token_id="token-1",
        outcome_label="11°C",
        source="clob",
        bids=[BookLevel(price=0.55, size=10.0)],
        asks=[BookLevel(price=0.65, size=10.0)],
    )

    fills = mm.simulate_quotes([sell_quote], {"11°C": sell_book})

    assert len(fills) == 1
    assert "token-1" not in mm.inventory
    assert mm.realized_pnl == pytest.approx(1.5)
    assert mm.unrealized_pnl == pytest.approx(0.0)
    assert mm.summary()["net_pnl"] == pytest.approx(1.5)


def test_paper_market_maker_stops_after_max_loss_breach() -> None:
    mm = PaperMarketMaker(
        risk_limits=RiskLimits(
            max_position_per_outcome=100.0,
            max_total_exposure=500.0,
            max_loss=1.0,
        )
    )
    mm.pnl = -1.0

    fills = mm.simulate_quotes(
        [
            Quote(
                token_id="token-1",
                outcome_label="11°C",
                fair_value=0.5,
                bid_price=0.45,
                bid_size=10.0,
                ask_price=0.55,
                ask_size=10.0,
            )
        ],
        {
            "11°C": BookSnapshot(
                market_id="market-1",
                token_id="token-1",
                outcome_label="11°C",
                source="clob",
                bids=[BookLevel(price=0.50, size=10.0)],
                asks=[BookLevel(price=0.40, size=10.0)],
            )
        },
    )

    assert fills == []
    assert mm.summary()["total_fills"] == 0
