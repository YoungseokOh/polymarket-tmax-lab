from __future__ import annotations

from pmtmax.execution.live_market_maker import LiveMarketMaker
from pmtmax.execution.quoter import Quote


class _FailingBroker:
    def cancel_orders(self, order_ids: list[str]) -> dict[str, object]:
        raise RuntimeError(f"failed to cancel {order_ids}")

    def post_limit_order(self, signal, *, size: float) -> dict[str, object]:
        raise AssertionError("post_limit_order should not run after cancel failure")


def test_live_market_maker_fails_closed_on_cancel_error() -> None:
    engine = LiveMarketMaker(broker=_FailingBroker())  # type: ignore[arg-type]
    engine.active_order_ids = ["order-1"]

    results = engine.update_quotes(
        [
            Quote(
                token_id="token-1",
                outcome_label="11°C",
                fair_value=0.5,
                bid_price=0.48,
                bid_size=5.0,
                ask_price=0.52,
                ask_size=5.0,
            )
        ],
        market_id="market-1",
        dry_run=False,
    )

    assert results[0]["reason"] == "cancel_failed"
    assert engine.active_order_ids == ["order-1"]
