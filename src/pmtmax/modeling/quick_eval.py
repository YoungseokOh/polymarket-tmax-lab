"""Reusable quick-eval helpers for saved model artifacts."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd
from scipy.stats import norm as _scipy_norm

from pmtmax.markets.market_spec import MarketSpec
from pmtmax.modeling.bin_mapper import map_normal_to_outcomes, map_samples_to_outcomes
from pmtmax.modeling.evaluation import brier_score, crps_from_samples
from pmtmax.modeling.sampling import sample_gaussian_mixture, sample_normal
from pmtmax.modeling.train import load_model, sanitize_model_frame

DEFAULT_HOLDOUT_FRAC = 0.20
DEFAULT_NUM_SAMPLES = 500
LOGGER = logging.getLogger(__name__)


class PredictiveModel(Protocol):
    def predict(self, frame: pd.DataFrame) -> object: ...


def _prediction_parts(prediction: object) -> list[object] | None:
    if isinstance(prediction, (list, tuple)):
        return list(prediction)
    return None


def quick_eval_holdout(
    frame: pd.DataFrame,
    *,
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one frame into fit/holdout partitions using the tail of target dates."""

    clean = sanitize_model_frame(frame)
    if "target_date" in clean.columns:
        dates = clean["target_date"].sort_values().unique()
        cutoff = dates[int(len(dates) * (1.0 - holdout_frac))]
        holdout = clean[clean["target_date"] >= cutoff].reset_index(drop=True)
        fit = clean[clean["target_date"] < cutoff].reset_index(drop=True)
        return fit, holdout
    cutoff_idx = int(len(clean) * (1.0 - holdout_frac))
    return clean.iloc[:cutoff_idx].reset_index(drop=True), clean.iloc[cutoff_idx:].reset_index(drop=True)


def _predict_row(model: PredictiveModel, row: pd.Series, *, num_samples: int) -> dict[str, object] | None:
    spec = MarketSpec.model_validate_json(str(row["market_spec_json"]))
    frame = row.to_frame().T.reset_index(drop=True)
    try:
        prediction = model.predict(frame)
    except Exception:
        return None
    parts = _prediction_parts(prediction)
    if parts is None:
        return None

    if len(parts) == 2:
        mean_arr, std_arr = parts
        mean = float(np.asarray(mean_arr).reshape(-1)[0])
        std = float(max(float(np.asarray(std_arr).reshape(-1)[0]), 0.1))
        samples = sample_normal(mean, std, num_samples=num_samples)
        probs = map_normal_to_outcomes(spec, mean, std)
    elif len(parts) == 3:
        weights_arr, means_arr, scales_arr = parts
        samples = sample_gaussian_mixture(
            np.asarray(weights_arr)[0],
            np.asarray(means_arr)[0],
            np.asarray(scales_arr)[0],
            num_samples=num_samples,
        )
        probs = map_samples_to_outcomes(spec, samples)
    else:
        return None
    return {
        "samples": samples,
        "probs": probs,
    }


def _gaussian_crps_vectorized(means: np.ndarray, stds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Analytical CRPS for Gaussian predictive distributions — exact, no sampling.

    CRPS(N(μ,σ), y) = σ * (z*(2Φ(z)-1) + 2φ(z) - 1/√π)  where z = (y-μ)/σ
    """
    z = (y_true - means) / stds
    values = stds * (z * (2.0 * _scipy_norm.cdf(z) - 1.0) + 2.0 * _scipy_norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return np.asarray(values, dtype=float)


def _market_units(frame: pd.DataFrame) -> np.ndarray:
    """Return market temperature units parsed from each row's MarketSpec JSON."""

    units: list[str] = []
    for payload in frame["market_spec_json"].tolist():
        try:
            spec = MarketSpec.model_validate_json(str(payload))
            units.append(spec.unit)
        except Exception:
            units.append("C")
    return np.asarray(units, dtype=object)


def _values_to_celsius(values: np.ndarray, units: np.ndarray, *, scale: bool = False) -> np.ndarray:
    """Convert per-row market-unit temperatures or scales to Celsius."""

    converted = values.astype(float, copy=True)
    fahrenheit = units == "F"
    if scale:
        converted[fahrenheit] = converted[fahrenheit] * (5.0 / 9.0)
    else:
        converted[fahrenheit] = (converted[fahrenheit] - 32.0) * (5.0 / 9.0)
    return converted


def evaluate_saved_model(
    model_path: Path,
    holdout: pd.DataFrame,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> dict[str, float] | None:
    """Evaluate one saved model artifact on a fixed holdout split.

    Uses batch prediction + analytical Gaussian CRPS instead of row-by-row
    Monte Carlo sampling — reduces evaluation from ~25 min to ~10 s on 55k rows.
    """
    try:
        model = cast(PredictiveModel, load_model(model_path))
    except Exception:
        return None

    if len(holdout) == 0:
        return None

    # Single batch prediction call replaces 55k individual model.predict(1_row) calls.
    try:
        prediction = model.predict(holdout)
    except Exception:
        # Fallback to legacy row-by-row path for models that can't handle batches.
        return _evaluate_saved_model_legacy(model, holdout, num_samples=num_samples)
    parts = _prediction_parts(prediction)
    if parts is None:
        return _evaluate_saved_model_legacy(model, holdout, num_samples=num_samples)

    units = _market_units(holdout)
    y_true = holdout["realized_daily_max"].to_numpy(dtype=float)
    winners = holdout["winning_outcome"].astype(str).tolist()
    valid = np.isfinite(y_true)

    if len(parts) == 2:
        means = np.asarray(parts[0]).reshape(-1).astype(float)
        stds = np.maximum(np.asarray(parts[1]).reshape(-1).astype(float), 0.1)
        if len(means) != len(holdout) or len(stds) != len(holdout):
            return _evaluate_saved_model_legacy(model, holdout, num_samples=num_samples)
        crps_market_vals = np.full(len(holdout), np.nan)
        crps_market_vals[valid] = _gaussian_crps_vectorized(means[valid], stds[valid], y_true[valid])
        mae_market_vals = np.full(len(holdout), np.nan)
        mae_market_vals[valid] = np.abs(y_true[valid] - means[valid])
        means_c = _values_to_celsius(means, units)
        stds_c = np.maximum(_values_to_celsius(stds, units, scale=True), 0.1)
        y_true_c = _values_to_celsius(y_true, units)
        crps_celsius_vals = np.full(len(holdout), np.nan)
        crps_celsius_vals[valid] = _gaussian_crps_vectorized(means_c[valid], stds_c[valid], y_true_c[valid])
        mae_celsius_vals = np.full(len(holdout), np.nan)
        mae_celsius_vals[valid] = np.abs(y_true_c[valid] - means_c[valid])
    else:
        # Mixture model: fall back to sampling path (uncommon case).
        return _evaluate_saved_model_legacy(model, holdout, num_samples=num_samples)

    # Brier score, Direction Accuracy, ECE: parse market specs per-row (JSON parse only, no model call).
    specs_json = holdout["market_spec_json"].tolist()
    valid_indices = np.where(valid)[0]
    briers: list[float] = []
    dir_acc_vals: list[float] = []
    ece_pairs: list[tuple[float, float]] = []  # (predicted_prob, actual_outcome) across all bins
    metric_indices: list[int] = []
    for idx in valid_indices:
        try:
            spec = MarketSpec.model_validate_json(str(specs_json[idx]))
            probs = map_normal_to_outcomes(spec, float(means[idx]), float(stds[idx]))
            winner = winners[idx]
            briers.append(brier_score(probs, winner))
            metric_indices.append(int(idx))
            # Direction Accuracy: is the top-1 predicted bin the actual winner?
            predicted_winner = max(probs, key=lambda k: probs[k])
            dir_acc_vals.append(1.0 if predicted_winner == winner else 0.0)
            # ECE: collect (predicted_prob, is_winner) for every bin in this row
            for bin_label, p in probs.items():
                ece_pairs.append((float(p), 1.0 if bin_label == winner else 0.0))
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("quick_eval probability row skipped", exc_info=exc)

    n = len(briers)
    if n == 0:
        return None
    metric_idx = np.asarray(metric_indices, dtype=int)

    # Expected Calibration Error — 10 equal-width probability buckets
    ece = 0.0
    if ece_pairs:
        p_all = np.array([x[0] for x in ece_pairs])
        y_all = np.array([x[1] for x in ece_pairs])
        bin_edges = np.linspace(0.0, 1.0, 11)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
            mask = (p_all >= lo) & (p_all < hi)
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / len(p_all)) * abs(p_all[mask].mean() - y_all[mask].mean())

    return {
        "n": float(n),
        "mae": float(np.mean(mae_celsius_vals[metric_idx])),
        "mae_market_unit": float(np.mean(mae_market_vals[metric_idx])),
        "mae_celsius_normalized": float(np.mean(mae_celsius_vals[metric_idx])),
        "crps": float(np.mean(crps_celsius_vals[metric_idx])),
        "crps_market_unit": float(np.mean(crps_market_vals[metric_idx])),
        "crps_celsius_normalized": float(np.mean(crps_celsius_vals[metric_idx])),
        "brier": float(np.mean(briers)),
        "dir_acc": float(np.mean(dir_acc_vals)),
        "ece": float(ece),
    }


def _evaluate_saved_model_legacy(
    model: PredictiveModel,
    holdout: pd.DataFrame,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> dict[str, float] | None:
    """Legacy row-by-row evaluation path (kept as fallback for non-Gaussian models)."""
    maes: list[float] = []
    maes_market: list[float] = []
    crps_scores: list[float] = []
    crps_scores_market: list[float] = []
    briers: list[float] = []
    dir_acc_vals: list[float] = []
    ece_pairs: list[tuple[float, float]] = []
    for _, row in holdout.iterrows():
        result = _predict_row(model, row, num_samples=num_samples)
        if result is None:
            continue
        y_true = float(row["realized_daily_max"])
        winner = str(row["winning_outcome"])
        probs_obj = result["probs"]
        if not isinstance(probs_obj, Mapping):
            continue
        probs = {str(label): float(prob) for label, prob in probs_obj.items()}
        samples = result["samples"]
        sample_array = np.asarray(samples, dtype=float)
        maes_market.append(abs(y_true - float(np.mean(sample_array))))
        crps_scores_market.append(crps_from_samples(sample_array, y_true))
        try:
            unit = MarketSpec.model_validate_json(str(row["market_spec_json"])).unit
        except Exception:
            unit = "C"
        if unit == "F":
            y_true_c = (y_true - 32.0) * (5.0 / 9.0)
            samples_c = (sample_array - 32.0) * (5.0 / 9.0)
        else:
            y_true_c = y_true
            samples_c = sample_array
        maes.append(abs(y_true_c - float(np.mean(samples_c))))
        crps_scores.append(crps_from_samples(samples_c, y_true_c))
        briers.append(brier_score(probs, winner))
        predicted_winner = max(probs, key=lambda k: probs[k])
        dir_acc_vals.append(1.0 if predicted_winner == winner else 0.0)
        for bin_label, p in probs.items():
            ece_pairs.append((float(p), 1.0 if bin_label == winner else 0.0))
    if not maes:
        return None

    ece = 0.0
    if ece_pairs:
        p_all = np.array([x[0] for x in ece_pairs])
        y_all = np.array([x[1] for x in ece_pairs])
        bin_edges = np.linspace(0.0, 1.0, 11)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
            mask = (p_all >= lo) & (p_all < hi)
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / len(p_all)) * abs(p_all[mask].mean() - y_all[mask].mean())

    return {
        "n": float(len(maes)),
        "mae": float(np.mean(maes)),
        "mae_market_unit": float(np.mean(maes_market)),
        "mae_celsius_normalized": float(np.mean(maes)),
        "crps": float(np.mean(crps_scores)),
        "crps_market_unit": float(np.mean(crps_scores_market)),
        "crps_celsius_normalized": float(np.mean(crps_scores)),
        "brier": float(np.mean(briers)),
        "dir_acc": float(np.mean(dir_acc_vals)),
        "ece": float(ece),
    }
