"""Pseudo-ensemble construction from lagged and multi-model runs."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def build_lagged_ensemble(trajectories: Iterable[np.ndarray]) -> np.ndarray:
    """Stack available trajectories into a pseudo-ensemble tensor."""

    members = [trajectory for trajectory in trajectories if trajectory.size > 0]
    if not members:
        return np.empty((0, 0))
    min_length = min(member.shape[0] for member in members)
    trimmed = [member[:min_length] for member in members]
    return np.vstack(trimmed)


def ensemble_statistics(ensemble: np.ndarray) -> dict[str, float]:
    """Summarize pseudo-ensemble uncertainty."""

    if ensemble.size == 0:
        return {"ensemble_mean": 0.0, "ensemble_std": 0.0, "ensemble_spread": 0.0}
    daily_max = ensemble.max(axis=1)
    return {
        "ensemble_mean": float(np.mean(daily_max)),
        "ensemble_std": float(np.std(daily_max)),
        "ensemble_spread": float(np.max(daily_max) - np.min(daily_max)),
    }

