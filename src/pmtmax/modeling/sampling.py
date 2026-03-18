"""Sampling utilities for probabilistic forecasts."""

from __future__ import annotations

import math

import numpy as np


def sample_normal(mean: float, std: float, num_samples: int, seed: int = 42) -> np.ndarray:
    """Sample from a Gaussian forecast."""

    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=max(std, 1e-6), size=num_samples)


def sample_gaussian_mixture(
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    num_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Sample from a finite Gaussian mixture."""

    rng = np.random.default_rng(seed)
    weights = weights / weights.sum()
    component_ids = rng.choice(len(weights), size=num_samples, p=weights)
    samples = np.empty(num_samples, dtype=float)
    for idx, component in enumerate(component_ids):
        samples[idx] = rng.normal(means[component], max(stds[component], 1e-6))
    return samples


def normal_cdf(x: float, mean: float, std: float) -> float:
    """Gaussian CDF without SciPy runtime dependency inside hot loops."""

    scaled = (x - mean) / max(std, 1e-6) / math.sqrt(2.0)
    return 0.5 * (1.0 + math.erf(scaled))

