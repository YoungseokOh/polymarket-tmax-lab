"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""

    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def gaussian_nll(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Average Gaussian negative log-likelihood."""

    sigma = np.clip(std, 1e-6, None)
    losses = 0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y_true - mean) / sigma) ** 2
    return float(np.mean(losses))


def crps_from_samples(samples: np.ndarray, y_true: float) -> float:
    """Approximate CRPS from Monte Carlo samples."""

    term_1 = np.mean(np.abs(samples - y_true))
    term_2 = 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
    return float(term_1 - term_2)


def brier_score(probabilities: dict[str, float], winning_label: str) -> float:
    """Multi-class Brier score."""

    return float(
        np.mean(
            [
                (prob - (1.0 if label == winning_label else 0.0)) ** 2
                for label, prob in probabilities.items()
            ]
        )
    )


def calibration_gap(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error for binary forecasts."""

    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities >= start) & (probabilities < end)
        if not np.any(mask):
            continue
        confidence = probabilities[mask].mean()
        accuracy = outcomes[mask].mean()
        error += abs(confidence - accuracy) * mask.mean()
    return float(error)

