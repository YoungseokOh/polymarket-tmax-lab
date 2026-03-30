"""Training orchestration."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from pmtmax.markets.market_spec import MarketSpec
from pmtmax.modeling.advanced.det2prob_nn import (
    Det2ProbNNModel,
    resolve_det2prob_variant,
    supported_det2prob_variants,
)
from pmtmax.modeling.advanced.tuned_ensemble import (
    TunedEnsembleModel,
    resolve_tuned_ensemble_variant,
    supported_tuned_ensemble_variants,
)
from pmtmax.modeling.baselines.gaussian_emos import (
    GaussianEMOSModel,
    resolve_gaussian_emos_variant,
    supported_gaussian_emos_variants,
)
from pmtmax.modeling.calibration import OutcomeCalibrator
from pmtmax.storage.schemas import ModelArtifact
from pmtmax.utils import set_global_seed, stable_hash

SUPPORTED_MODEL_NAMES = ("gaussian_emos", "tuned_ensemble", "det2prob_nn")


def supported_model_names() -> tuple[str, ...]:
    """Return the canonical v2 model registry."""

    return SUPPORTED_MODEL_NAMES


def require_supported_model_name(model_name: str) -> str:
    """Validate a public model name against the v2 registry."""

    if model_name not in SUPPORTED_MODEL_NAMES:
        supported = ", ".join(SUPPORTED_MODEL_NAMES)
        msg = f"Unsupported model: {model_name}. Supported models: {supported}."
        raise ValueError(msg)
    return model_name


def supported_ablation_variants(model_name: str) -> tuple[str, ...]:
    """Return supported ablation variants for a model family."""

    require_supported_model_name(model_name)
    if model_name == "tuned_ensemble":
        return supported_tuned_ensemble_variants()
    if model_name == "det2prob_nn":
        return supported_det2prob_variants()
    if model_name == "gaussian_emos":
        return supported_gaussian_emos_variants()
    return ()


def require_supported_variant(model_name: str, variant: str | None) -> str | None:
    """Validate an optional internal ablation variant for a model family."""

    if variant is None:
        return None
    supported = supported_ablation_variants(model_name)
    if not supported:
        msg = f"Model {model_name} does not expose ablation variants."
        raise ValueError(msg)
    if variant not in supported:
        msg = f"Unsupported {model_name} variant: {variant}. Supported variants: {', '.join(supported)}."
        raise ValueError(msg)
    return variant


def sanitize_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a shallow copy without global imputation.

    Individual models now own their missing-value strategy so availability and
    missingness indicators remain recoverable at fit/predict time.
    """

    return frame.copy()


def default_feature_names(frame: pd.DataFrame) -> list[str]:
    """Infer the default modeling features from a dataset."""

    excluded = {
        "market_id",
        "station_id",
        "target_date",
        "realized_daily_max",
        "winning_outcome",
        "source_priority",
        "settlement_eligible",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].nunique(dropna=False) > 1
        and not column.startswith("kma_ldps_")
        and not column.endswith("_num_hours")
    ]


def _artifact_calibration_path(model_path: Path) -> Path:
    """Return the sibling calibrator path for a stored model artifact."""

    return model_path.with_name(f"{model_path.stem}.calibrator.pkl")


def _dataset_signature(frame: pd.DataFrame) -> str:
    """Return a stable dataset fingerprint for a training frame."""

    payload = {
        "rows": len(frame),
        "columns": list(frame.columns),
        "target_date_min": str(frame["target_date"].min()) if "target_date" in frame.columns and not frame.empty else None,
        "target_date_max": str(frame["target_date"].max()) if "target_date" in frame.columns and not frame.empty else None,
    }
    return stable_hash(json.dumps(payload, sort_keys=True, default=str))


def _calibration_split(
    frame: pd.DataFrame,
    *,
    split_policy: Literal["market_day", "target_day"],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Split a frame into fit/calibration partitions."""

    if frame.empty or "winning_outcome" not in frame.columns or "market_spec_json" not in frame.columns:
        return frame, None
    if len(frame) < 40:
        return frame, None

    sort_columns = [column for column in ["target_date", "decision_time_utc", "market_id", "decision_horizon"] if column in frame.columns]
    ordered = frame.sort_values(sort_columns).reset_index(drop=True)
    if split_policy == "market_day" and {"market_id", "target_date"}.issubset(ordered.columns):
        group_ids = ordered[["market_id", "target_date"]].astype(str).agg("|".join, axis=1)
    elif "target_date" in ordered.columns:
        group_ids = ordered["target_date"].astype(str)
    else:
        return ordered, None

    ordered = ordered.assign(_cal_group_id=group_ids)
    unique_groups = ordered["_cal_group_id"].drop_duplicates().tolist()
    calibration_groups = max(int(len(unique_groups) * 0.2), 1)
    if len(unique_groups) - calibration_groups < 2:
        return ordered.drop(columns="_cal_group_id"), None
    fit_groups = set(unique_groups[:-calibration_groups])
    holdout_groups = set(unique_groups[-calibration_groups:])
    fit_frame = ordered.loc[ordered["_cal_group_id"].isin(fit_groups)].drop(columns="_cal_group_id").copy()
    calibration_frame = ordered.loc[ordered["_cal_group_id"].isin(holdout_groups)].drop(columns="_cal_group_id").copy()
    if fit_frame.empty or calibration_frame.empty:
        return ordered.drop(columns="_cal_group_id"), None
    return fit_frame, calibration_frame


def _fit_and_persist_calibrator(
    *,
    model_path: Path,
    model_name: str,
    calibration_frame: pd.DataFrame,
) -> Path | None:
    """Fit an outcome calibrator on held-out rows and persist it next to the model."""

    if calibration_frame.empty or "winning_outcome" not in calibration_frame.columns or "market_spec_json" not in calibration_frame.columns:
        return None

    from pmtmax.modeling.predict import predict_market

    probability_rows: list[dict[str, float]] = []
    winners: list[str] = []
    for _, row in calibration_frame.iterrows():
        spec = MarketSpec.model_validate_json(str(row["market_spec_json"]))
        forecast = predict_market(
            model_path,
            model_name,
            spec,
            row.to_frame().T,
            calibrate=False,
        )
        raw_probs = forecast.outcome_probabilities_raw or forecast.outcome_probabilities
        probability_rows.append({label: float(value) for label, value in raw_probs.items()})
        winners.append(str(row["winning_outcome"]))

    if len(probability_rows) < 10 or len(set(winners)) < 2:
        return None

    labels = sorted({label for row in probability_rows for label in row})
    probabilities = {
        label: np.asarray([row.get(label, 0.0) for row in probability_rows], dtype=float)
        for label in labels
    }
    calibrator = OutcomeCalibrator()
    calibrator.fit(probabilities, np.asarray(winners, dtype=object))
    path = _artifact_calibration_path(model_path)
    with path.open("wb") as handle:
        pickle.dump(calibrator, handle)
    return path


def train_model(
    model_name: str,
    frame: pd.DataFrame,
    artifacts_dir: Path,
    *,
    split_policy: Literal["market_day", "target_day"] = "market_day",
    seed: int | None = None,
    variant: str | None = None,
) -> ModelArtifact:
    """Train a named model and persist it to disk."""

    require_supported_model_name(model_name)
    require_supported_variant(model_name, variant)
    if seed is not None:
        set_global_seed(seed)

    clean_frame = sanitize_model_frame(frame)
    fit_frame, calibration_frame = _calibration_split(clean_frame, split_policy=split_policy)
    features = default_feature_names(clean_frame)
    model: Any
    if model_name == "gaussian_emos":
        resolved_variant = resolve_gaussian_emos_variant(variant)
        model = GaussianEMOSModel(features, variant=resolved_variant.name)
        model.fit(fit_frame)
    elif model_name == "det2prob_nn":
        resolved_variant = resolve_det2prob_variant(variant)
        model = Det2ProbNNModel(features, split_policy=split_policy, variant=resolved_variant.name)
        model.fit(fit_frame)
    elif model_name == "tuned_ensemble":
        resolved_variant = resolve_tuned_ensemble_variant(variant)
        model = TunedEnsembleModel(features, split_policy=split_policy, variant=resolved_variant.name)
        model.fit(fit_frame)
    else:
        msg = f"Unsupported trainable model: {model_name}"
        raise ValueError(msg)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = model_name if variant is None else f"{model_name}__{variant}"
    path = artifacts_dir / f"{artifact_stem}.pkl"
    with path.open("wb") as handle:
        pickle.dump(model, handle)

    calibration_path = None
    metrics: dict[str, float] = {
        "fit_rows": float(len(fit_frame)),
        "dataset_rows": float(len(clean_frame)),
        "variant_is_default": float(variant is None),
    }
    if calibration_frame is not None:
        metrics["calibration_rows"] = float(len(calibration_frame))
    if calibration_frame is not None:
        calibration_path_obj = _fit_and_persist_calibrator(
            model_path=path,
            model_name=model_name,
            calibration_frame=calibration_frame,
        )
        if calibration_path_obj is not None:
            calibration_path = str(calibration_path_obj)
            metrics["calibration_fitted"] = 1.0
        else:
            metrics["calibration_fitted"] = 0.0

    return ModelArtifact(
        model_name=model_name,
        version="0.1.0",
        trained_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
        features=features,
        metrics=metrics,
        path=str(path),
        contract_version="v2",
        seed=seed,
        dataset_signature=_dataset_signature(clean_frame),
        split_policy=split_policy,
        variant=variant,
        calibration_path=calibration_path,
        diagnostics=dict(getattr(model, "diagnostics_", {})),
        status="stable" if variant is None else "experimental",
    )


def load_model(path: Path) -> Any:
    """Load a serialized model artifact."""

    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301
