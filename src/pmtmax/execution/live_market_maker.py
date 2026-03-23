"""Live market-making order management via LiveBroker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pmtmax.execution.live_broker import LiveBroker
from pmtmax.execution.quoter import Quote
from pmtmax.logging_utils import get_logger
from pmtmax.storage.schemas import RiskLimits, TradeSignal

LOGGER = get_logger(__name__)


@dataclass
class LiveMarketMaker:
    """Manage two-sided live orders via LiveBroker.

    Requires the same safety gates as LiveBroker:
    - PMTMAX_LIVE_TRADING=true
    - PMTMAX_CONFIRM_LIVE_TRADING=YES_I_UNDERSTAND
    """

    broker: LiveBroker
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    active_order_ids: list[str] = field(default_factory=list)
    inventory: dict[str, float] = field(default_factory=dict)

    def cancel_all(self) -> Any:
        """Emergency cancel all active orders."""

        if not self.active_order_ids:
            LOGGER.info("No active orders to cancel")
            return None
        LOGGER.info("Cancelling %d active orders", len(self.active_order_ids))
        result = self.broker.cancel_orders(self.active_order_ids)
        self.active_order_ids.clear()
        return result

    def update_quotes(
        self,
        quotes: list[Quote],
        *,
        market_id: str,
        dry_run: bool = True,
    ) -> list[dict[str, Any]]:
        """Cancel existing orders and post new two-sided quotes.

        In dry_run mode, only previews are generated — no orders are posted.
        """

        results: list[dict[str, Any]] = []

        # Cancel existing orders first
        if self.active_order_ids and not dry_run:
            try:
                self.cancel_all()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to cancel existing orders, skipping quote refresh: %s", exc)
                return [{
                    "market_id": market_id,
                    "reason": "cancel_failed",
                    "error": str(exc),
                    "active_order_ids": list(self.active_order_ids),
                }]

        for quote in quotes:
            # Post bid (BUY) side
            bid_signal = TradeSignal(
                market_id=market_id,
                token_id=quote.token_id,
                outcome_label=quote.outcome_label,
                side="buy",
                fair_probability=quote.fair_value,
                executable_price=quote.bid_price,
                fee_estimate=0.0,
                slippage_estimate=0.0,
                edge=0.0,
                confidence=quote.fair_value,
                rationale=f"MM bid for {quote.outcome_label}",
                mode="live",
            )

            # Post ask (SELL) side
            ask_signal = TradeSignal(
                market_id=market_id,
                token_id=quote.token_id,
                outcome_label=quote.outcome_label,
                side="sell",
                fair_probability=quote.fair_value,
                executable_price=quote.ask_price,
                fee_estimate=0.0,
                slippage_estimate=0.0,
                edge=0.0,
                confidence=quote.fair_value,
                rationale=f"MM ask for {quote.outcome_label}",
                mode="live",
            )

            if dry_run:
                results.append({
                    "outcome": quote.outcome_label,
                    "bid": quote.bid_price,
                    "ask": quote.ask_price,
                    "size": quote.bid_size,
                    "dry_run": True,
                })
            else:
                try:
                    bid_result = self.broker.post_limit_order(bid_signal, size=quote.bid_size)
                    order_id = bid_result.get("orderID") or bid_result.get("id")
                    if order_id:
                        self.active_order_ids.append(order_id)
                    results.append({"side": "buy", "result": bid_result})
                except Exception as exc:  # noqa: BLE001
                    results.append({"side": "buy", "error": str(exc)})

                try:
                    ask_result = self.broker.post_limit_order(ask_signal, size=quote.ask_size)
                    order_id = ask_result.get("orderID") or ask_result.get("id")
                    if order_id:
                        self.active_order_ids.append(order_id)
                    results.append({"side": "sell", "result": ask_result})
                except Exception as exc:  # noqa: BLE001
                    results.append({"side": "sell", "error": str(exc)})

        return results
