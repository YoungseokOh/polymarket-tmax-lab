"""Champion selection from benchmark leaderboards."""

from __future__ import annotations

import pandas as pd

_LOWER_IS_BETTER_WEIGHTS = {
    "avg_crps_mean": 0.25,
    "avg_brier_mean": 0.2,
    "calibration_gap_mean": 0.15,
    "mae_mean": 0.15,
    "rmse_mean": 0.05,
    "nll_mean": 0.05,
    "mae_std": 0.05,
}
_HIGHER_IS_BETTER_WEIGHTS = {
    "real_history_pnl_mean": 0.05,
    "quote_proxy_pnl_mean": 0.05,
}


def score_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    """Attach benchmark-based champion scores to a leaderboard."""

    scored = results.copy()
    total_score = pd.Series(0.0, index=scored.index, dtype=float)

    for column, weight in _LOWER_IS_BETTER_WEIGHTS.items():
        if column not in scored.columns:
            continue
        total_score += scored[column].rank(method="dense", ascending=True) * weight
    for column, weight in _HIGHER_IS_BETTER_WEIGHTS.items():
        if column not in scored.columns:
            continue
        total_score += scored[column].rank(method="dense", ascending=False) * weight

    scored["champion_score"] = total_score
    return scored.sort_values(["champion_score", "model_name"], ignore_index=True)


def select_champion(results: pd.DataFrame) -> str:
    """Select the champion model from an aggregated benchmark leaderboard."""

    if results.empty:
        raise ValueError("Cannot select a champion from an empty leaderboard.")
    scored = score_leaderboard(results)
    return str(scored.iloc[0]["model_name"])
