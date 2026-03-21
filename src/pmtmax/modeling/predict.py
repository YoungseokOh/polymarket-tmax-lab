"""Prediction orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pmtmax.markets.market_spec import MarketSpec
from pmtmax.modeling.bin_mapper import map_normal_to_outcomes, map_samples_to_outcomes
from pmtmax.modeling.sampling import sample_normal
from pmtmax.modeling.train import load_model, sanitize_model_frame
from pmtmax.storage.schemas import CalibrationMetadata, ProbForecast


def predict_market(model_path: Path, model_name: str, spec: MarketSpec, frame: pd.DataFrame) -> ProbForecast:
    """Load a model, generate a probabilistic forecast, and map to market outcomes."""

    from pmtmax.modeling.advanced.pinn_postproc import PermutationInvariantNNModel
    from pmtmax.modeling.advanced.transformer_postproc import TransformerPostprocModel

    model = load_model(model_path)
    clean = sanitize_model_frame(frame)
    if hasattr(model, "feature_names"):
        for col in model.feature_names:
            if col not in clean.columns:
                clean[col] = 0.0

    # Sequence-based models need numpy array input, not DataFrame
    if isinstance(model, TransformerPostprocModel):
        arr = clean[model.feature_names].to_numpy(dtype=np.float32)
        prediction = model.predict(arr)
    elif isinstance(model, PermutationInvariantNNModel):
        arr = clean[model.feature_names].to_numpy(dtype=np.float32)
        arr_3d = arr.reshape(arr.shape[0], arr.shape[1], 1)
        prediction = model.predict(arr_3d)
    else:
        prediction = model.predict(clean)
    if len(prediction) == 2:
        mean_arr, std_arr = prediction
        mean = float(np.asarray(mean_arr).reshape(-1)[0])
        std = float(np.asarray(std_arr).reshape(-1)[0])
        samples = sample_normal(mean, std, num_samples=2000)
        probabilities = map_normal_to_outcomes(spec, mean, std)
    else:
        weights, means, scales = prediction
        weights_arr = np.asarray(weights)[0]
        means_arr = np.asarray(means)[0]
        scales_arr = np.asarray(scales)[0]
        mean = float(np.sum(weights_arr * means_arr))
        variance = float(np.sum(weights_arr * (scales_arr**2 + means_arr**2)) - mean**2)
        std = float(np.sqrt(max(variance, 1e-6)))
        samples = sample_normal(mean, std, num_samples=2000)
        probabilities = map_samples_to_outcomes(spec, samples)
    return ProbForecast(
        target_market=spec.market_id,
        generated_at=datetime.now(tz=UTC),
        samples=samples[:500].tolist(),
        mean=mean,
        std=std,
        daily_max_distribution={"family": "gaussian_approximation"},
        outcome_probabilities=probabilities,
        calibration_metadata=CalibrationMetadata(
            model_name=model_name,
            calibration_method="none",
            fitted_at=datetime.now(tz=UTC),
            notes="Calibration optional in v1 CLI workflow",
        ),
    )
