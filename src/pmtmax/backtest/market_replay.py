"""Historical market replay utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MarketReplay:
    """Replay public market price history for research backtests."""

    history: pd.DataFrame

    def latest_before(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        """Return the last snapshot before a decision timestamp."""

        subset = self.history.loc[self.history["timestamp"] <= timestamp]
        if subset.empty:
            return subset
        latest_ts = subset["timestamp"].max()
        return subset.loc[subset["timestamp"] == latest_ts].copy()

    def market_implied_probs(self) -> dict[str, float]:
        """Compute the latest market-implied distribution from a replay frame."""

        latest = self.history.sort_values("timestamp").groupby("outcome_label").tail(1)
        probs = dict(zip(latest["outcome_label"], latest["price"], strict=True))
        total = sum(probs.values())
        return {label: value / total for label, value in probs.items()} if total > 0 else probs

