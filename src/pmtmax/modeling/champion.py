"""Champion selection from benchmark leaderboards."""

from __future__ import annotations

import pandas as pd

_RESEARCH_LOWER_IS_BETTER_WEIGHTS = {
    "avg_crps_mean": 0.25,
    "avg_brier_mean": 0.2,
    "calibration_gap_mean": 0.15,
    "mae_mean": 0.15,
    "rmse_mean": 0.05,
    "nll_mean": 0.05,
    "mae_std": 0.05,
}
_RESEARCH_HIGHER_IS_BETTER_WEIGHTS = {
    "real_history_pnl_mean": 0.05,
    "quote_proxy_pnl_mean": 0.05,
}
_EXECUTION_CANDIDATE_LOWER_IS_BETTER_WEIGHTS = {
    "calibration_gap_mean": 0.2,
    "avg_crps_mean": 0.1,
    "avg_brier_mean": 0.1,
    "mae_mean": 0.05,
    "rmse_mean": 0.05,
    "nll_mean": 0.05,
}
_EXECUTION_CANDIDATE_HIGHER_IS_BETTER_WEIGHTS = {
    "quote_proxy_pnl_mean": 0.2,
    "real_history_pnl_mean": 0.15,
    "quote_proxy_avg_edge_mean": 0.04,
    "real_history_avg_edge_mean": 0.03,
    "quote_proxy_hit_rate_mean": 0.02,
    "real_history_hit_rate_mean": 0.01,
}


def _score_ranked_leaderboard(
    results: pd.DataFrame,
    *,
    lower_is_better_weights: dict[str, float],
    higher_is_better_weights: dict[str, float],
    score_column: str,
) -> pd.DataFrame:
    """Attach one weighted rank-based score to a leaderboard."""

    scored = results.copy()
    total_score = pd.Series(0.0, index=scored.index, dtype=float)

    for column, weight in lower_is_better_weights.items():
        if column not in scored.columns:
            continue
        total_score += scored[column].rank(method="dense", ascending=True) * weight
    for column, weight in higher_is_better_weights.items():
        if column not in scored.columns:
            continue
        total_score += scored[column].rank(method="dense", ascending=False) * weight

    scored[score_column] = total_score
    return scored


def score_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    """Attach benchmark-based research champion scores to a leaderboard."""

    scored = _score_ranked_leaderboard(
        results,
        lower_is_better_weights=_RESEARCH_LOWER_IS_BETTER_WEIGHTS,
        higher_is_better_weights=_RESEARCH_HIGHER_IS_BETTER_WEIGHTS,
        score_column="champion_score",
    )
    return scored.sort_values(["champion_score", "model_name"], ignore_index=True)


def score_execution_candidate_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    """Attach benchmark-based execution-candidate scores to a leaderboard."""

    scored = results.copy()
    if "sample_adequacy_passed" in scored.columns and scored["sample_adequacy_passed"].astype(bool).any():
        scored = scored.loc[scored["sample_adequacy_passed"].astype(bool)].copy()
    scored = _score_ranked_leaderboard(
        scored,
        lower_is_better_weights=_EXECUTION_CANDIDATE_LOWER_IS_BETTER_WEIGHTS,
        higher_is_better_weights=_EXECUTION_CANDIDATE_HIGHER_IS_BETTER_WEIGHTS,
        score_column="execution_candidate_score",
    )
    return scored.sort_values(["execution_candidate_score", "model_name"], ignore_index=True)


def select_champion(results: pd.DataFrame) -> str:
    """Select the champion model from an aggregated benchmark leaderboard."""

    if results.empty:
        raise ValueError("Cannot select a champion from an empty leaderboard.")
    scored = score_leaderboard(results)
    return str(scored.iloc[0]["model_name"])


def select_execution_candidate(results: pd.DataFrame) -> str:
    """Select the execution-focused model from an aggregated leaderboard."""

    if results.empty:
        raise ValueError("Cannot select an execution candidate from an empty leaderboard.")
    scored = score_execution_candidate_leaderboard(results)
    return str(scored.iloc[0]["model_name"])
