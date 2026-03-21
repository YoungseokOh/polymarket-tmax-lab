"""Two-sided quote computation for model-based market making."""

from __future__ import annotations

from dataclasses import dataclass

from pmtmax.storage.schemas import RiskLimits


@dataclass
class Quote:
    """A single two-sided quote for one outcome."""

    token_id: str
    outcome_label: str
    fair_value: float
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float


@dataclass
class Quoter:
    """Compute two-sided quotes from model probabilities and current inventory."""

    base_half_spread: float = 0.02
    skew_factor: float = 0.5
    base_size: float = 10.0

    def compute_quotes(
        self,
        outcome_probs: dict[str, float],
        token_ids: dict[str, str],
        inventory: dict[str, float],
        risk_limits: RiskLimits,
    ) -> list[Quote]:
        """Generate bid/ask quotes for each outcome.

        Parameters
        ----------
        outcome_probs:
            Mapping of outcome_label → model probability.
        token_ids:
            Mapping of outcome_label → token_id.
        inventory:
            Mapping of token_id → net position (positive = long).
        risk_limits:
            Position and exposure limits.
        """

        quotes: list[Quote] = []
        orders_remaining = risk_limits.max_orders_per_cycle

        for outcome_label, fair_prob in outcome_probs.items():
            if orders_remaining <= 0:
                break

            token_id = token_ids.get(outcome_label)
            if token_id is None:
                continue

            net_position = inventory.get(token_id, 0.0)

            # Inventory skew: shift quotes to rebalance position
            skew = self.skew_factor * net_position / max(risk_limits.max_position_per_outcome, 1e-6)

            bid_price = fair_prob - self.base_half_spread - skew
            ask_price = fair_prob + self.base_half_spread - skew

            # Clamp to valid price range
            bid_price = max(0.01, min(bid_price, 0.99))
            ask_price = max(0.01, min(ask_price, 0.99))

            # Ensure bid < ask
            if bid_price >= ask_price:
                mid = (bid_price + ask_price) / 2.0
                bid_price = max(0.01, mid - 0.005)
                ask_price = min(0.99, mid + 0.005)

            # Size scaling: reduce size as position approaches limit
            position_ratio = abs(net_position) / max(risk_limits.max_position_per_outcome, 1e-6)
            size = self.base_size * max(1.0 - position_ratio, 0.0)

            if size <= 0:
                continue

            quotes.append(
                Quote(
                    token_id=token_id,
                    outcome_label=outcome_label,
                    fair_value=fair_prob,
                    bid_price=round(bid_price, 4),
                    bid_size=round(size, 2),
                    ask_price=round(ask_price, 4),
                    ask_size=round(size, 2),
                )
            )
            orders_remaining -= 2  # bid + ask

        return quotes
