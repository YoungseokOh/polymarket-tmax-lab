"""Position sizing."""

from __future__ import annotations


def capped_kelly(
    edge: float,
    fair_prob: float,
    bankroll: float,
    price: float,
    max_fraction: float = 0.05,
) -> float:
    """Kelly criterion for binary outcome tokens.

    Uses the standard Kelly formula for binary payoffs:
        f* = (b*p - q) / b
    where b = (1/price - 1) is net odds, p = fair_prob, q = 1 - p.
    """
    if price <= 0 or price >= 1 or edge <= 0:
        return 0.0
    b = (1.0 / price) - 1.0
    q = 1.0 - fair_prob
    raw_fraction = max((b * fair_prob - q) / b, 0.0)
    return bankroll * min(raw_fraction, max_fraction)
