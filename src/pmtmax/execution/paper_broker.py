"""Paper trading broker."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from pmtmax.execution.edge import compute_edge
from pmtmax.execution.fees import estimate_fee
from pmtmax.execution.slippage import estimate_slippage
from pmtmax.storage.schemas import ExecutionFill, PaperPosition, TradeSignal


@dataclass
class PaperBroker:
    """Conservative paper broker with inventory and PnL tracking."""

    bankroll: float = 10_000.0
    inventory: list[ExecutionFill] = field(default_factory=list)
    positions: dict[str, PaperPosition] = field(default_factory=dict)

    def simulate_fill(
        self,
        signal: TradeSignal,
        *,
        spread: float,
        liquidity: float,
        size: float,
    ) -> ExecutionFill | None:
        """Simulate a conservative fill if edge stays positive after costs."""

        notional = signal.executable_price * size
        fee = estimate_fee(notional)
        slippage = estimate_slippage(signal.executable_price, spread, liquidity, size)
        edge = compute_edge(signal.fair_probability, signal.executable_price, fee, slippage)
        if edge <= 0 or self.bankroll < notional + fee:
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
        )
        self.bankroll -= notional + fee
        return fill

    def inventory_frame(self) -> pd.DataFrame:
        """Return current inventory as a DataFrame."""

        return pd.DataFrame([item.model_dump() for item in self.inventory])
