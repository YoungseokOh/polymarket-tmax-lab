"""Paper trading broker."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from pmtmax.execution.edge import compute_edge
from pmtmax.execution.fees import estimate_fee
from pmtmax.execution.forecast_exit import should_forecast_exit
from pmtmax.execution.slippage import estimate_book_slippage
from pmtmax.execution.stops import evaluate_stops
from pmtmax.storage.schemas import BookSnapshot, ExecutionFill, PaperPosition, ProbForecast, TradeSignal


@dataclass
class PaperBroker:
    """Conservative paper broker with inventory and PnL tracking."""

    bankroll: float = 10_000.0
    inventory: list[ExecutionFill] = field(default_factory=list)
    positions: dict[str, PaperPosition] = field(default_factory=dict)
    stop_loss_pct: float = 0.20
    trailing_stop_rise_pct: float = 0.20
    forecast_exit_buffer: float = 0.05

    def simulate_fill(
        self,
        signal: TradeSignal,
        *,
        book: BookSnapshot,
        size: float,
    ) -> ExecutionFill | None:
        """Simulate a conservative fill if edge stays positive after costs."""

        ask_liq = sum(level.size for level in book.asks)
        size = min(size, ask_liq)
        if size <= 0:
            return None
        fee_bps = (signal.fee_estimate / signal.executable_price) * 10_000.0 if signal.executable_price > 0 else 0.0
        notional = signal.executable_price * size
        total_fee = estimate_fee(notional, taker_bps=fee_bps)
        fee_per_share = estimate_fee(signal.executable_price, taker_bps=fee_bps)
        slippage = estimate_book_slippage(signal.side, book.asks if signal.side == "buy" else book.bids, size)
        if slippage is None:
            return None
        edge = compute_edge(signal.fair_probability, signal.executable_price, fee_per_share, slippage)
        if edge <= 0 or self.bankroll < notional + total_fee:
            return None
        fill = ExecutionFill(
            market_id=signal.market_id,
            token_id=signal.token_id,
            outcome_label=signal.outcome_label,
            side=signal.side,
            price=signal.executable_price + slippage if signal.side == "buy" else signal.executable_price - slippage,
            size=size,
            mode="paper",
            timestamp=datetime.now(tz=UTC),
        )
        self.inventory.append(fill)
        self.positions[signal.token_id] = PaperPosition(
            market_id=signal.market_id,
            token_id=signal.token_id,
            outcome_label=signal.outcome_label,
            side=signal.side,
            avg_price=fill.price,
            size=size,
            high_water_mark=fill.price,
            opened_at=fill.timestamp,
        )
        self.bankroll -= notional + total_fee
        return fill

    def close_position(self, token_id: str, current_price: float, reason: str) -> ExecutionFill | None:
        """Close an open position and return the exit fill."""

        position = self.positions.get(token_id)
        if position is None:
            return None

        fill = ExecutionFill(
            market_id=position.market_id,
            token_id=token_id,
            outcome_label=position.outcome_label,
            side="sell" if position.side == "buy" else "buy",
            price=current_price,
            size=position.size,
            mode="paper",
            timestamp=datetime.now(tz=UTC),
        )
        self.inventory.append(fill)
        self.bankroll += current_price * position.size
        closed = position.model_copy(
            update={
                "closed_at": fill.timestamp,
                "close_reason": reason,
                "realized_pnl": (current_price - position.avg_price) * position.size
                if position.side == "buy"
                else (position.avg_price - current_price) * position.size,
            },
        )
        self.positions[token_id] = closed
        del self.positions[token_id]
        return fill

    def check_stops(self, current_prices: dict[str, float]) -> list[ExecutionFill]:
        """Evaluate stop-loss and trailing-stop for all open positions."""

        fills: list[ExecutionFill] = []
        for token_id in list(self.positions):
            price = current_prices.get(token_id)
            if price is None:
                continue
            position, reason = evaluate_stops(self.positions[token_id], price, self.stop_loss_pct)
            self.positions[token_id] = position
            if reason is not None:
                fill = self.close_position(token_id, price, reason)
                if fill is not None:
                    fills.append(fill)
        return fills

    def check_forecast_exits(self, forecasts: dict[str, ProbForecast]) -> list[ExecutionFill]:
        """Close positions whose forecast probability has dropped below the exit buffer."""

        fills: list[ExecutionFill] = []
        for token_id in list(self.positions):
            position = self.positions.get(token_id)
            if position is None:
                continue
            forecast = forecasts.get(position.market_id)
            if forecast is None:
                continue
            if should_forecast_exit(position.outcome_label, forecast, self.forecast_exit_buffer):
                fill = self.close_position(token_id, position.avg_price, "forecast_exit")
                if fill is not None:
                    fills.append(fill)
        return fills

    def inventory_frame(self) -> pd.DataFrame:
        """Return current inventory as a DataFrame."""

        return pd.DataFrame([item.model_dump() for item in self.inventory])
