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
    average_entry_price: dict[str, float] = field(default_factory=dict)
    pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_fills: list[MMFill] = field(default_factory=list)
    cash: float = 0.0
    last_marks: dict[str, float] = field(default_factory=dict)

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
        if self.pnl <= -self.risk_limits.max_loss:
            LOGGER.warning("Skipping quotes because PnL %.2f breached max_loss %.2f", self.pnl, self.risk_limits.max_loss)
            return fills

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
                    self._apply_fill(fill)
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
                    self._apply_fill(fill)
                    LOGGER.info(
                        "MM SELL fill: %s @ %.4f x %.2f",
                        quote.outcome_label,
                        fill_price,
                        fill_size,
                    )

            mark = self._mark_price(book)
            if mark is not None:
                self.last_marks[quote.token_id] = mark

        self.total_fills.extend(fills)
        self._update_pnl()

        # Check risk limits
        total_exposure = sum(abs(pos) for pos in self.inventory.values())
        if total_exposure > self.risk_limits.max_total_exposure:
            LOGGER.warning("Total exposure %.2f exceeds limit %.2f", total_exposure, self.risk_limits.max_total_exposure)

        return fills

    def _update_pnl(self) -> None:
        """Update net PnL using realized cashflows plus open-position marks."""

        unrealized = 0.0
        for token_id, position in self.inventory.items():
            mark = self.last_marks.get(token_id)
            avg_entry = self.average_entry_price.get(token_id)
            if mark is None or avg_entry is None or abs(position) <= 1e-9:
                continue
            if position > 0:
                unrealized += (mark - avg_entry) * position
            else:
                unrealized += (avg_entry - mark) * abs(position)
        self.unrealized_pnl = unrealized
        self.pnl = self.realized_pnl + self.unrealized_pnl

    def _apply_fill(self, fill: MMFill) -> None:
        """Update position state and realized cashflows after one fill."""

        token_id = fill.token_id
        current_position = self.inventory.get(token_id, 0.0)
        avg_entry = self.average_entry_price.get(token_id, fill.price)
        signed_size = fill.size if fill.side == "buy" else -fill.size

        if fill.side == "buy":
            self.cash -= fill.price * fill.size
            if current_position < 0:
                cover_size = min(fill.size, abs(current_position))
                self.realized_pnl += (avg_entry - fill.price) * cover_size
                new_position = current_position + fill.size
                if new_position > 0:
                    self.inventory[token_id] = new_position
                    self.average_entry_price[token_id] = fill.price
                elif abs(new_position) <= 1e-9:
                    self.inventory.pop(token_id, None)
                    self.average_entry_price.pop(token_id, None)
                else:
                    self.inventory[token_id] = new_position
            else:
                new_position = current_position + fill.size
                total_size = current_position + fill.size
                weighted_cost = (current_position * avg_entry) + (fill.size * fill.price)
                self.inventory[token_id] = new_position
                self.average_entry_price[token_id] = weighted_cost / max(total_size, 1e-9)
        else:
            self.cash += fill.price * fill.size
            if current_position > 0:
                close_size = min(fill.size, current_position)
                self.realized_pnl += (fill.price - avg_entry) * close_size
                new_position = current_position - fill.size
                if new_position < 0:
                    self.inventory[token_id] = new_position
                    self.average_entry_price[token_id] = fill.price
                elif abs(new_position) <= 1e-9:
                    self.inventory.pop(token_id, None)
                    self.average_entry_price.pop(token_id, None)
                else:
                    self.inventory[token_id] = new_position
            else:
                new_position = current_position + signed_size
                old_abs = abs(current_position)
                new_abs = abs(new_position)
                weighted_entry = (old_abs * avg_entry) + (fill.size * fill.price)
                self.inventory[token_id] = new_position
                self.average_entry_price[token_id] = weighted_entry / max(new_abs, 1e-9)

    @staticmethod
    def _mark_price(book: BookSnapshot) -> float | None:
        """Return a conservative mark price from the current book."""

        if book.bids and book.asks:
            return (book.best_bid() + book.best_ask()) / 2.0
        if book.bids:
            return book.best_bid()
        if book.asks:
            return book.best_ask()
        return None

    def summary(self) -> dict:
        """Return a summary of current MM state."""

        return {
            "total_fills": len(self.total_fills),
            "buy_fills": sum(1 for f in self.total_fills if f.side == "buy"),
            "sell_fills": sum(1 for f in self.total_fills if f.side == "sell"),
            "realized_pnl": round(self.realized_pnl, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "net_pnl": round(self.pnl, 4),
            "positions": {k: round(v, 4) for k, v in self.inventory.items() if abs(v) > 1e-8},
            "total_exposure": round(sum(abs(v) for v in self.inventory.values()), 4),
        }
