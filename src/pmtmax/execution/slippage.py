"""Slippage estimation."""

from __future__ import annotations

from typing import Literal

from pmtmax.storage.schemas import BookLevel


def estimate_book_slippage(
    side: Literal["buy", "sell"],
    levels: list[BookLevel],
    order_size: float,
) -> float | None:
    """Estimate slippage from top-of-book using a depth walk.

    Returns the VWAP distance from the best visible level, or ``None`` when the
    requested order size cannot be filled from the visible book.
    """

    if order_size <= 0:
        return 0.0
    if not levels:
        return None

    remaining = order_size
    notional = 0.0
    for level in levels:
        if remaining <= 0:
            break
        take = min(level.size, remaining)
        if take <= 0:
            continue
        notional += take * level.price
        remaining -= take

    if remaining > 1e-9:
        return None

    top_price = levels[0].price
    avg_price = notional / order_size
    if side == "buy":
        return max(avg_price - top_price, 0.0)
    return max(top_price - avg_price, 0.0)
