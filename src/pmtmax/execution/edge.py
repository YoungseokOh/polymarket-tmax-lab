"""Trade edge calculations."""

from __future__ import annotations


def compute_edge(
    fair_probability: float,
    executable_price: float,
    fee_estimate: float = 0.0,
    slippage_estimate: float = 0.0,
) -> float:
    """Compute after-cost probability edge."""

    return fair_probability - executable_price - fee_estimate - slippage_estimate

