"""Reusable quick-eval helpers for saved model artifacts."""

from __future__ import annotations

from pathlib import Path

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


def _predict_row(model: object, row: pd.Series, *, num_samples: int) -> dict[str, object] | None:
    spec = MarketSpec.model_validate_json(str(row["market_spec_json"]))
    frame = row.to_frame().T.reset_index(drop=True)
    try:
        prediction = model.predict(frame)
    except Exception:
        return None

    if len(prediction) == 2:
        mean_arr, std_arr = prediction
        mean = float(np.asarray(mean_arr).reshape(-1)[0])
        std = float(max(float(np.asarray(std_arr).reshape(-1)[0]), 0.1))
        samples = sample_normal(mean, std, num_samples=num_samples)
        probs = map_normal_to_outcomes(spec, mean, std)
    else:
        weights_arr, means_arr, scales_arr = prediction
        samples = sample_gaussian_mixture(
            np.asarray(weights_arr)[0],
            np.asarray(means_arr)[0],
            np.asarray(scales_arr)[0],
            num_samples=num_samples,
        )
        probs = map_samples_to_outcomes(spec, samples)
    return {
        "samples": samples,
        "probs": probs,
    }


def _gaussian_crps_vectorized(means: np.ndarray, stds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Analytical CRPS for Gaussian predictive distributions — exact, no sampling.

    CRPS(N(μ,σ), y) = σ * (z*(2Φ(z)-1) + 2φ(z) - 1/√π)  where z = (y-μ)/σ
    """
    z = (y_true - means) / stds
    return stds * (z * (2.0 * _scipy_norm.cdf(z) - 1.0) + 2.0 * _scipy_norm.pdf(z) - 1.0 / np.sqrt(np.pi))


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
        model = load_model(model_path)
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

    y_true = holdout["realized_daily_max"].to_numpy(dtype=float)
    winners = holdout["winning_outcome"].astype(str).tolist()
    valid = np.isfinite(y_true)

    if len(prediction) == 2:
        means = np.asarray(prediction[0]).reshape(-1).astype(float)
        stds = np.maximum(np.asarray(prediction[1]).reshape(-1).astype(float), 0.1)
        crps_vals = _gaussian_crps_vectorized(means[valid], stds[valid], y_true[valid])
        mae_vals = np.abs(y_true[valid] - means[valid])
    else:
        # Mixture model: fall back to sampling path (uncommon case).
        return _evaluate_saved_model_legacy(model, holdout, num_samples=num_samples)

    # Brier score: parse market specs per-row (JSON parse only, no model call).
    specs_json = holdout["market_spec_json"].tolist()
    valid_indices = np.where(valid)[0]
    briers: list[float] = []
    for idx in valid_indices:
        try:
            spec = MarketSpec.model_validate_json(str(specs_json[idx]))
            probs = map_normal_to_outcomes(spec, float(means[idx]), float(stds[idx]))
            briers.append(brier_score(probs, winners[idx]))
        except Exception:
            continue

    n = len(briers)
    if n == 0:
        return None
    return {
        "n": float(n),
        "mae": float(np.mean(mae_vals[:n])),
        "crps": float(np.mean(crps_vals[:n])),
        "brier": float(np.mean(briers)),
    }


def _evaluate_saved_model_legacy(
    model: object,
    holdout: pd.DataFrame,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> dict[str, float] | None:
    """Legacy row-by-row evaluation path (kept as fallback for non-Gaussian models)."""
    maes: list[float] = []
    crps_scores: list[float] = []
    briers: list[float] = []
    for _, row in holdout.iterrows():
        result = _predict_row(model, row, num_samples=num_samples)
        if result is None:
            continue
        y_true = float(row["realized_daily_max"])
        winner = str(row["winning_outcome"])
        probs = result["probs"]
        samples = result["samples"]
        maes.append(abs(y_true - float(np.mean(samples))))
        crps_scores.append(crps_from_samples(np.asarray(samples), y_true))
        briers.append(brier_score(probs, winner))
    if not maes:
        return None
    return {
        "n": float(len(maes)),
        "mae": float(np.mean(maes)),
        "crps": float(np.mean(crps_scores)),
        "brier": float(np.mean(briers)),
    }
