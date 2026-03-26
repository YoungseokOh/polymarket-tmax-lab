"""Prediction orchestration."""

from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from pmtmax.markets.market_spec import MarketSpec
from pmtmax.modeling.bin_mapper import map_normal_to_outcomes, map_samples_to_outcomes
from pmtmax.modeling.sampling import sample_gaussian_mixture, sample_normal
from pmtmax.modeling.train import _artifact_calibration_path, load_model, sanitize_model_frame
from pmtmax.storage.schemas import CalibrationMetadata, ProbForecast


def _feature_availability(frame: pd.DataFrame, feature_names: list[str]) -> dict[str, bool]:
    """Return a best-effort feature availability map for one prediction frame."""

    if "feature_availability_json" in frame.columns and not frame.empty:
        raw = frame.iloc[0].get("feature_availability_json")
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    return {str(key): bool(value) for key, value in payload.items()}
            except json.JSONDecodeError:
                pass
    return {
        name: bool(name in frame.columns and not frame[name].isna().all())
        for name in feature_names
    }


def _load_calibrator(model_path: Path) -> object | None:
    """Load a sibling calibrator artifact when present."""

    path = _artifact_calibration_path(model_path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301


def predict_market(
    model_path: Path,
    model_name: str,
    spec: MarketSpec,
    frame: pd.DataFrame,
    *,
    calibrate: bool = True,
) -> ProbForecast:
    """Load a model, generate a probabilistic forecast, and map to market outcomes."""

    model = load_model(model_path)
    clean = sanitize_model_frame(frame)
    feature_names = list(getattr(model, "feature_names", []))
    availability = _feature_availability(frame, feature_names)
    if hasattr(model, "feature_names"):
        for col in model.feature_names:
            if col not in clean.columns:
                clean[col] = np.nan

    prediction = model.predict(clean)
    if len(prediction) == 2:
        mean_arr, std_arr = prediction
        mean = float(np.asarray(mean_arr).reshape(-1)[0])
        std = float(np.asarray(std_arr).reshape(-1)[0])
        samples = sample_normal(mean, std, num_samples=2000)
        raw_probabilities = map_normal_to_outcomes(spec, mean, std)
        distribution_family = "gaussian"
        distribution_payload: dict[str, object] = {"mean": mean, "std": std}
    else:
        weights, means, scales = prediction
        weights_arr = np.asarray(weights)[0]
        means_arr = np.asarray(means)[0]
        scales_arr = np.asarray(scales)[0]
        mean = float(np.sum(weights_arr * means_arr))
        variance = float(np.sum(weights_arr * (scales_arr**2 + means_arr**2)) - mean**2)
        std = float(np.sqrt(max(variance, 1e-6)))
        samples = sample_gaussian_mixture(weights_arr, means_arr, scales_arr, num_samples=2000)
        raw_probabilities = map_samples_to_outcomes(spec, samples)
        distribution_family = "gaussian_mixture"
        distribution_payload = {
            "weights": weights_arr.tolist(),
            "means": means_arr.tolist(),
            "scales": scales_arr.tolist(),
        }

    calibrated_probabilities: dict[str, float] = {}
    active_probabilities = raw_probabilities
    probability_source: Literal["raw", "calibrated"] = "raw"
    calibration_metadata = CalibrationMetadata(
        model_name=model_name,
        calibration_method="none",
        fitted_at=datetime.now(tz=UTC),
        notes="Calibration unavailable.",
    )
    if calibrate:
        calibrator = _load_calibrator(model_path)
        if calibrator is not None and hasattr(calibrator, "calibrate"):
            calibrated_probabilities = calibrator.calibrate(raw_probabilities)
            active_probabilities = calibrated_probabilities
            probability_source = "calibrated"
            calibration_metadata = CalibrationMetadata(
                model_name=model_name,
                calibration_method="isotonic",
                fitted_at=datetime.now(tz=UTC),
                notes="Loaded sibling calibrator artifact.",
                path=str(_artifact_calibration_path(model_path)),
            )
    return ProbForecast(
        target_market=spec.market_id,
        generated_at=datetime.now(tz=UTC),
        contract_version="v2",
        samples=samples[:500].tolist(),
        mean=mean,
        std=std,
        distribution_family=distribution_family,
        distribution_payload=distribution_payload,
        daily_max_distribution={"family": distribution_family, **distribution_payload},
        outcome_probabilities=active_probabilities,
        outcome_probabilities_raw=raw_probabilities,
        outcome_probabilities_calibrated=calibrated_probabilities,
        probability_source=probability_source,
        feature_availability=availability,
        calibration_metadata=calibration_metadata,
    )
