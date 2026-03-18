"""Map predictive distributions to Polymarket outcomes."""

from __future__ import annotations

import numpy as np

from pmtmax.markets.market_spec import MarketSpec, OutcomeBin, PrecisionRule
from pmtmax.modeling.sampling import normal_cdf


def _resolution_bounds(bin_spec: OutcomeBin, precision: PrecisionRule) -> tuple[float, float]:
    half_step = precision.step / 2.0
    if bin_spec.lower is not None and bin_spec.upper is not None and bin_spec.lower == bin_spec.upper:
        center = bin_spec.lower
        return center - half_step, center + half_step

    lower = -np.inf if bin_spec.lower is None else bin_spec.lower - half_step
    upper = np.inf if bin_spec.upper is None else bin_spec.upper + half_step
    return lower, upper


def map_normal_to_outcomes(spec: MarketSpec, mean: float, std: float) -> dict[str, float]:
    """Map a Gaussian daily-max forecast to outcome probabilities."""

    result: dict[str, float] = {}
    for outcome in spec.outcome_schema:
        lower, upper = _resolution_bounds(outcome, spec.precision_rule)
        probability = normal_cdf(upper, mean, std) - normal_cdf(lower, mean, std)
        result[outcome.label] = max(probability, 0.0)
    return normalize_probabilities(result)


def map_samples_to_outcomes(spec: MarketSpec, samples: np.ndarray) -> dict[str, float]:
    """Map Monte Carlo samples to outcome probabilities."""

    result: dict[str, float] = {}
    for outcome in spec.outcome_schema:
        lower, upper = _resolution_bounds(outcome, spec.precision_rule)
        mask = (samples >= lower) & (samples < upper)
        if np.isinf(upper):
            mask = samples >= lower
        if np.isinf(lower):
            mask = samples < upper
        result[outcome.label] = float(mask.mean()) if samples.size else 0.0
    return normalize_probabilities(result)


def normalize_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    """Normalize a dict of probabilities to sum to one."""

    total = sum(probabilities.values())
    if total <= 0:
        uniform = 1.0 / max(len(probabilities), 1)
        return dict.fromkeys(probabilities, uniform)
    normalized = {label: value / total for label, value in probabilities.items()}
    correction = 1.0 - sum(normalized.values())
    if normalized:
        first_key = next(iter(normalized))
        normalized[first_key] += correction
    return normalized


def infer_winning_label(spec: MarketSpec, realized_value: float) -> str:
    """Return the resolving outcome label for a realized settlement value."""

    for outcome in spec.outcome_schema:
        lower, upper = _resolution_bounds(outcome, spec.precision_rule)
        if realized_value >= lower and (realized_value < upper or np.isinf(upper)):
            return outcome.label
    msg = f"No outcome matched {realized_value}"
    raise ValueError(msg)

