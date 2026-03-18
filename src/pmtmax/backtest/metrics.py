"""Backtest summary metrics."""

from __future__ import annotations

import pandas as pd


def summarize_trade_log(trades: pd.DataFrame) -> dict[str, float]:
    """Summarize a paper or replay trade log."""

    if trades.empty:
        return {"num_trades": 0.0, "pnl": 0.0, "hit_rate": 0.0, "avg_edge": 0.0}
    return {
        "num_trades": float(len(trades)),
        "pnl": float(trades["realized_pnl"].sum()),
        "hit_rate": float((trades["realized_pnl"] > 0).mean()),
        "avg_edge": float(trades["edge"].mean()),
    }

