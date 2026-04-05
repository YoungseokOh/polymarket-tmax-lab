"""Reusable quick-eval helpers for saved model artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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


def evaluate_saved_model(
    model_path: Path,
    holdout: pd.DataFrame,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> dict[str, float] | None:
    """Evaluate one saved model artifact on a fixed holdout split."""

    try:
        model = load_model(model_path)
    except Exception:
        return None

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
