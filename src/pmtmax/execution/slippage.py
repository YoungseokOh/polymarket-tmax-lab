"""Slippage estimation."""

from __future__ import annotations


def estimate_slippage(price: float, spread: float, liquidity: float, order_size: float) -> float:
    """Estimate slippage as a function of spread and relative order size."""

    liquidity_penalty = 0.0 if liquidity <= 0 else min(order_size / liquidity, 1.0) * spread
    return max(spread / 2.0, 0.0) + liquidity_penalty * price

