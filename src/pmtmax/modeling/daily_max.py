"""Daily-max transformations."""

from __future__ import annotations

import numpy as np


def daily_max_from_hourly(hourly: np.ndarray) -> float:
    """Return the daily maximum from an hourly trajectory."""

    if hourly.size == 0:
        return float("nan")
    return float(np.max(hourly))


def daily_max_from_samples(hourly_samples: np.ndarray) -> np.ndarray:
    """Map trajectory samples of shape [samples, hours] to daily maxima."""

    if hourly_samples.ndim != 2:
        msg = "hourly_samples must be 2D"
        raise ValueError(msg)
    return np.asarray(hourly_samples.max(axis=1), dtype=float)


def sample_correlated_hourly_normals(
    means: np.ndarray,
    stds: np.ndarray,
    rho: float = 0.85,
    num_samples: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Generate approximate joint hourly trajectories using a Toeplitz correlation."""

    size = len(means)
    if size == 0:
        return np.empty((0, 0))
    distances = np.abs(np.subtract.outer(np.arange(size), np.arange(size)))
    corr = rho ** distances
    covariance = corr * np.outer(stds, stds)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(means, covariance, size=num_samples)
