"""PnL accounting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    outcome_label: str
    price: float
    size: float
    side: str


def settle_position(position: Position, winning_label: str, fee_paid: float = 0.0) -> float:
    """Settle a binary-style outcome token position."""

    payout = 1.0 if position.outcome_label == winning_label else 0.0
    if position.side == "buy":
        return (payout - position.price) * position.size - fee_paid
    return (position.price - payout) * position.size - fee_paid

