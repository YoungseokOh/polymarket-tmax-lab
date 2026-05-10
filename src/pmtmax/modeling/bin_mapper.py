"""Map predictive distributions to Polymarket outcomes."""

from __future__ import annotations

import numpy as np

from pmtmax.markets.market_spec import MarketSpec, OutcomeBin, PrecisionRule
from pmtmax.modeling.sampling import normal_cdf


def _resolution_bounds(bin_spec: OutcomeBin, precision: PrecisionRule) -> tuple[float, float]:
    # Market labels resolve against whole-degree official readings.  A 2-degree
    # label such as ``35-36°F`` therefore covers readings that round/settle to
    # either 35 or 36, not the full ``step`` width on both sides.  Expanding by
    # ``precision.step / 2`` would make adjacent labels overlap.
    settlement_half_step = 0.5 if precision.rounding in {"whole_degree", "range_bin"} else precision.step / 2.0
    if bin_spec.lower is not None and bin_spec.upper is not None and bin_spec.lower == bin_spec.upper:
        center = bin_spec.lower
        return center - settlement_half_step, center + settlement_half_step

    lower = -np.inf if bin_spec.lower is None else bin_spec.lower - settlement_half_step
    upper = np.inf if bin_spec.upper is None else bin_spec.upper + settlement_half_step
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

