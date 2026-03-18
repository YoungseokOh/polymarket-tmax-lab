"""Champion model selection."""

from __future__ import annotations

import pandas as pd


def select_champion(results: pd.DataFrame) -> str:
    """Select the champion model by rolling-origin metrics."""

    score = (
        results["crps_rank"] * 0.35
        + results["calibration_rank"] * 0.2
        + results["brier_rank"] * 0.2
        + results["ev_rank"] * 0.25
    )
    return str(results.assign(score=score).sort_values("score").iloc[0]["model_name"])

