"""Paper market-making simulator with inventory tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from pmtmax.execution.quoter import Quote
from pmtmax.logging_utils import get_logger
from pmtmax.storage.schemas import BookSnapshot, RiskLimits

LOGGER = get_logger(__name__)


@dataclass
class MMFill:
    """A simulated market-making fill."""

    token_id: str
    outcome_label: str
    side: str  # "buy" or "sell"
    price: float
    size: float
    timestamp: datetime


@dataclass
class PaperMarketMaker:
    """Simulate two-sided market making against the CLOB book."""

    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    inventory: dict[str, float] = field(default_factory=dict)  # token_id → net position
    pnl: float = 0.0
    total_fills: list[MMFill] = field(default_factory=list)
    cash: float = 0.0

    def simulate_quotes(
        self,
        quotes: list[Quote],
        books: dict[str, BookSnapshot],
    ) -> list[MMFill]:
        """Check quotes against current CLOB books and simulate fills.

        A fill occurs when:
        - Our bid >= counterparty's best ask → we buy
        - Our ask <= counterparty's best bid → we sell
        """

        fills: list[MMFill] = []
        now = datetime.now(tz=UTC)

        for quote in quotes:
            book = books.get(quote.outcome_label)
            if book is None:
                continue

            current_position = self.inventory.get(quote.token_id, 0.0)

            # Check bid fill: our bid crosses their ask
            if book.asks and quote.bid_price >= book.best_ask():
                fill_price = book.best_ask()
                fill_size = min(quote.bid_size, book.asks[0].size)

                if current_position + fill_size <= self.risk_limits.max_position_per_outcome:
                    fill = MMFill(
                        token_id=quote.token_id,
                        outcome_label=quote.outcome_label,
                        side="buy",
                        price=fill_price,
                        size=fill_size,
                        timestamp=now,
                    )
                    fills.append(fill)
                    self.inventory[quote.token_id] = current_position + fill_size
                    self.cash -= fill_price * fill_size
                    current_position += fill_size
                    LOGGER.info(
                        "MM BUY fill: %s @ %.4f x %.2f",
                        quote.outcome_label,
                        fill_price,
                        fill_size,
                    )

            # Check ask fill: our ask crosses their bid
            if book.bids and quote.ask_price <= book.best_bid():
                fill_price = book.best_bid()
                fill_size = min(quote.ask_size, book.bids[0].size)

                if current_position - fill_size >= -self.risk_limits.max_position_per_outcome:
                    fill = MMFill(
                        token_id=quote.token_id,
                        outcome_label=quote.outcome_label,
                        side="sell",
                        price=fill_price,
                        size=fill_size,
                        timestamp=now,
                    )
                    fills.append(fill)
                    self.inventory[quote.token_id] = current_position - fill_size
                    self.cash += fill_price * fill_size
                    LOGGER.info(
                        "MM SELL fill: %s @ %.4f x %.2f",
                        quote.outcome_label,
                        fill_price,
                        fill_size,
                    )

        self.total_fills.extend(fills)
        self._update_pnl()

        # Check risk limits
        total_exposure = sum(abs(pos) for pos in self.inventory.values())
        if total_exposure > self.risk_limits.max_total_exposure:
            LOGGER.warning("Total exposure %.2f exceeds limit %.2f", total_exposure, self.risk_limits.max_total_exposure)

        return fills

    def _update_pnl(self) -> None:
        """Update realized PnL from cash flow."""

        self.pnl = self.cash

    def summary(self) -> dict:
        """Return a summary of current MM state."""

        return {
            "total_fills": len(self.total_fills),
            "buy_fills": sum(1 for f in self.total_fills if f.side == "buy"),
            "sell_fills": sum(1 for f in self.total_fills if f.side == "sell"),
            "net_pnl": round(self.pnl, 4),
            "positions": {k: round(v, 4) for k, v in self.inventory.items() if abs(v) > 1e-8},
            "total_exposure": round(sum(abs(v) for v in self.inventory.values()), 4),
        }
