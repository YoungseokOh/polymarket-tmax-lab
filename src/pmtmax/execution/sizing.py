"""Position sizing."""

from __future__ import annotations


def capped_kelly(edge: float, confidence: float, bankroll: float, max_fraction: float = 0.05) -> float:
    """Return a capped Kelly-style stake."""

    raw_fraction = max(edge * confidence, 0.0)
    return bankroll * min(raw_fraction, max_fraction)

