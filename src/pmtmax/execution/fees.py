"""Fee estimation."""

from __future__ import annotations


def estimate_fee(notional: float, taker_bps: float = 200.0) -> float:
    """Estimate execution fees in quote currency."""

    return notional * taker_bps / 10_000.0

