"""Train-side source gating wrappers for Gaussian forecast blends."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from pmtmax.markets.market_spec import MarketSpec
from pmtmax.modeling.quick_eval import _gaussian_crps_vectorized
from pmtmax.modeling.tail_calibration import (
    _from_celsius,
    _market_units,
    _to_celsius,
    sentinel_daily_max_count,
)

DEFAULT_SOURCE_NAMES = (
    "ecmwf_ifs025",
    "ecmwf_aifs025_single",
    "kma_gdps",
    "gfs_seamless",
)

SampleWeightMode = Literal["uniform", "absolute_regret", "fallback_regret"]


@dataclass(frozen=True)
class SourceGatingConfig:
    """Configuration for learning a fallback blend from train-side rows only."""

    name: str = "source_gate"
    crps_margin_c: float = 0.0
    min_fallback_weight: float = 0.0
    max_fallback_weight: float = 1.0
    min_std_c: float = 0.1
    source_names: tuple[str, ...] = field(default_factory=lambda: DEFAULT_SOURCE_NAMES)
    random_state: int = 1729
    max_iter: int = 120
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    l2_regularization: float = 0.05
    sample_weight_mode: SampleWeightMode = "uniform"
    sample_weight_floor: float = 0.05
    sample_weight_cap: float = 20.0


@dataclass
class ConstantFallbackGate:
    """Small pickle-safe gate used when train labels contain only one class."""

    fallback_weight: float
    classes_: np.ndarray = field(default_factory=lambda: np.asarray([0, 1], dtype=int))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        weight = float(np.clip(self.fallback_weight, 0.0, 1.0))
        return np.column_stack(
            [
                np.full(len(frame), 1.0 - weight, dtype=float),
                np.full(len(frame), weight, dtype=float),
            ]
        )


def _safe_token(value: str) -> str:
    token = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    token = "_".join(part for part in token.split("_") if part)
    return token or "unknown"


def _extract_cities(frame: pd.DataFrame) -> pd.Series:
    if "city" in frame.columns:
        return frame["city"].fillna("unknown").astype(str)
    if "market_spec_json" not in frame.columns:
        return pd.Series(["unknown"] * len(frame), index=frame.index, dtype=object)

    values: list[str] = []
    cache: dict[str, str] = {}
    for raw in frame["market_spec_json"].tolist():
        key = str(raw)
        if key not in cache:
            city = "unknown"
            try:
                spec = MarketSpec.model_validate_json(key)
                city = spec.city
            except Exception:
                try:
                    payload = json.loads(key)
                    if isinstance(payload, dict):
                        city = str(payload.get("city") or "unknown")
                except json.JSONDecodeError:
                    city = "unknown"
            cache[key] = city
        values.append(cache[key])
    return pd.Series(values, index=frame.index, dtype=object)


def _parse_feature_availability(frame: pd.DataFrame) -> pd.DataFrame:
    if "feature_availability_json" not in frame.columns:
        return pd.DataFrame(index=frame.index)

    def _parse(raw: object) -> dict[str, Any]:
        if not isinstance(raw, str):
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    return pd.DataFrame(list(frame["feature_availability_json"].map(_parse)), index=frame.index)


def _fit_categories(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    cities = sorted({str(value) for value in _extract_cities(frame).tolist() if str(value).strip()})
    horizons = sorted(
        {
            str(value)
            for value in frame.get(
                "decision_horizon",
                pd.Series(["unknown"] * len(frame), index=frame.index, dtype=object),
            )
            .fillna("unknown")
            .astype(str)
            .tolist()
            if str(value).strip()
        }
    )
    return cities, horizons


@dataclass
class SourceGatedGaussianModel:
    """Wrap two Gaussian models with a learned unit-aware fallback blend."""

    primary_model: object
    fallback_model: object
    gate_model: object
    config: SourceGatingConfig = field(default_factory=SourceGatingConfig)
    feature_columns: list[str] = field(default_factory=list)
    city_categories: list[str] = field(default_factory=list)
    horizon_categories: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.feature_names:
            self.feature_names = list(getattr(self.primary_model, "feature_names", []))

    @staticmethod
    def _predict_gaussian(model: object, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        prediction = model.predict(frame)  # type: ignore[attr-defined]
        if not isinstance(prediction, (tuple, list)) or len(prediction) != 2:
            msg = "SourceGatedGaussianModel only supports Gaussian mean/std models."
            raise ValueError(msg)
        mean, std = prediction
        return np.asarray(mean).reshape(-1).astype(float), np.asarray(std).reshape(-1).astype(float)

    def _gate_features(
        self,
        frame: pd.DataFrame,
        *,
        units: np.ndarray,
        primary_mean_c: np.ndarray,
        primary_std_c: np.ndarray,
        fallback_mean_c: np.ndarray,
        fallback_std_c: np.ndarray,
    ) -> pd.DataFrame:
        source_frame = pd.DataFrame(index=frame.index)
        availability = _parse_feature_availability(frame)
        source_values: list[np.ndarray] = []
        for source in self.config.source_names:
            column = f"{source}_model_daily_max"
            numeric = pd.to_numeric(
                frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index),
                errors="coerce",
            )
            values_c = _to_celsius(numeric.to_numpy(dtype=float), units)
            source_values.append(values_c)
            source_frame[f"{source}_daily_max_c"] = values_c
            source_frame[f"{source}_delta_vs_fallback_c"] = values_c - fallback_mean_c
            if source in availability.columns:
                available = availability[source].fillna(False).astype(float)
            else:
                available = numeric.notna().astype(float)
            source_frame[f"avail__{source}"] = available.to_numpy(dtype=float)

        source_matrix = np.column_stack(source_values) if source_values else np.empty((len(frame), 0))
        if source_matrix.shape[1] > 0:
            finite = np.isfinite(source_matrix)
            finite_count = finite.sum(axis=1)
            source_sum = np.nansum(source_matrix, axis=1)
            source_mean = np.divide(
                source_sum,
                finite_count,
                out=np.full(len(frame), np.nan, dtype=float),
                where=finite_count > 0,
            )
            squared_delta = np.where(finite, np.square(source_matrix - source_mean[:, None]), 0.0)
            source_variance = np.divide(
                squared_delta.sum(axis=1),
                finite_count,
                out=np.full(len(frame), np.nan, dtype=float),
                where=finite_count > 0,
            )
            source_min = np.min(np.where(finite, source_matrix, np.inf), axis=1)
            source_max = np.max(np.where(finite, source_matrix, -np.inf), axis=1)
            source_frame["source_mean_c"] = source_mean
            source_frame["source_std_c"] = np.sqrt(source_variance)
            source_frame["source_range_c"] = np.where(finite_count > 0, source_max - source_min, np.nan)
        else:
            source_frame["source_mean_c"] = np.nan
            source_frame["source_std_c"] = np.nan
            source_frame["source_range_c"] = np.nan

        avail_columns = [f"avail__{source}" for source in self.config.source_names]
        source_frame["available_source_count"] = source_frame[avail_columns].sum(axis=1)
        source_frame["available_source_fraction"] = (
            source_frame["available_source_count"] / max(len(self.config.source_names), 1)
        )

        lead_hours = pd.to_numeric(
            frame["lead_hours"] if "lead_hours" in frame.columns else pd.Series(0.0, index=frame.index),
            errors="coerce",
        ).fillna(0.0)
        target_dates = pd.to_datetime(
            frame["target_date"] if "target_date" in frame.columns else pd.Series(pd.NaT, index=frame.index),
            errors="coerce",
        )
        decision_times = pd.to_datetime(
            frame["decision_time_utc"]
            if "decision_time_utc" in frame.columns
            else pd.Series(pd.NaT, index=frame.index),
            errors="coerce",
        )
        day_of_year = target_dates.dt.dayofyear.fillna(1.0).to_numpy(dtype=float)

        data: dict[str, np.ndarray] = {
            "primary_mean_c": primary_mean_c,
            "primary_std_c": primary_std_c,
            "fallback_mean_c": fallback_mean_c,
            "fallback_std_c": fallback_std_c,
            "mean_diff_c": primary_mean_c - fallback_mean_c,
            "abs_mean_diff_c": np.abs(primary_mean_c - fallback_mean_c),
            "std_diff_c": primary_std_c - fallback_std_c,
            "abs_std_diff_c": np.abs(primary_std_c - fallback_std_c),
            "lead_hours": lead_hours.to_numpy(dtype=float),
            "lead_hours_sq": np.square(lead_hours.to_numpy(dtype=float)),
            "day_of_year_sin": np.sin(2.0 * np.pi * day_of_year / 366.0),
            "day_of_year_cos": np.cos(2.0 * np.pi * day_of_year / 366.0),
            "target_month": target_dates.dt.month.fillna(0.0).to_numpy(dtype=float),
            "decision_hour": decision_times.dt.hour.fillna(0.0).to_numpy(dtype=float),
            "sentinel_daily_max_count": sentinel_daily_max_count(frame, units=units),
        }
        feature_frame = pd.concat([pd.DataFrame(data, index=frame.index), source_frame], axis=1)

        cities = _extract_cities(frame).fillna("unknown").astype(str)
        horizons = frame.get(
            "decision_horizon",
            pd.Series(["unknown"] * len(frame), index=frame.index, dtype=object),
        ).fillna("unknown").astype(str)
        for city in self.city_categories:
            feature_frame[f"city__{_safe_token(city)}"] = (cities == city).astype(float)
        for horizon in self.horizon_categories:
            feature_frame[f"horizon__{_safe_token(horizon)}"] = (horizons == horizon).astype(float)

        return feature_frame.replace([np.inf, -np.inf], np.nan)

    def _predict_fallback_weight(
        self,
        frame: pd.DataFrame,
        *,
        units: np.ndarray,
        primary_mean_c: np.ndarray,
        primary_std_c: np.ndarray,
        fallback_mean_c: np.ndarray,
        fallback_std_c: np.ndarray,
    ) -> np.ndarray:
        features = self._gate_features(
            frame,
            units=units,
            primary_mean_c=primary_mean_c,
            primary_std_c=primary_std_c,
            fallback_mean_c=fallback_mean_c,
            fallback_std_c=fallback_std_c,
        )
        if self.feature_columns:
            for column in self.feature_columns:
                if column not in features.columns:
                    features[column] = np.nan
            features = features[self.feature_columns]
        probabilities = self.gate_model.predict_proba(features)
        classes = np.asarray(getattr(self.gate_model, "classes_", np.asarray([0, 1])))
        if 1 not in classes:
            weights = np.zeros(len(frame), dtype=float)
        else:
            class_idx = int(np.flatnonzero(classes == 1)[0])
            weights = np.asarray(probabilities[:, class_idx], dtype=float)
        return np.clip(
            weights,
            float(self.config.min_fallback_weight),
            float(self.config.max_fallback_weight),
        )

    def predict_fallback_weight(self, frame: pd.DataFrame) -> np.ndarray:
        frame_reset = frame.reset_index(drop=True)
        units = _market_units(frame_reset)
        primary_mean, primary_std = self._predict_gaussian(self.primary_model, frame_reset)
        fallback_mean, fallback_std = self._predict_gaussian(self.fallback_model, frame_reset)
        return self._predict_fallback_weight(
            frame_reset,
            units=units,
            primary_mean_c=_to_celsius(primary_mean, units),
            primary_std_c=np.maximum(_to_celsius(primary_std, units, scale=True), self.config.min_std_c),
            fallback_mean_c=_to_celsius(fallback_mean, units),
            fallback_std_c=np.maximum(_to_celsius(fallback_std, units, scale=True), self.config.min_std_c),
        )

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        frame_reset = frame.reset_index(drop=True)
        units = _market_units(frame_reset)
        primary_mean, primary_std = self._predict_gaussian(self.primary_model, frame_reset)
        fallback_mean, fallback_std = self._predict_gaussian(self.fallback_model, frame_reset)
        primary_mean_c = _to_celsius(primary_mean, units)
        primary_std_c = np.maximum(_to_celsius(primary_std, units, scale=True), self.config.min_std_c)
        fallback_mean_c = _to_celsius(fallback_mean, units)
        fallback_std_c = np.maximum(_to_celsius(fallback_std, units, scale=True), self.config.min_std_c)
        fallback_weight = self._predict_fallback_weight(
            frame_reset,
            units=units,
            primary_mean_c=primary_mean_c,
            primary_std_c=primary_std_c,
            fallback_mean_c=fallback_mean_c,
            fallback_std_c=fallback_std_c,
        )

        mean_c = (1.0 - fallback_weight) * primary_mean_c + fallback_weight * fallback_mean_c
        variance_c = (
            (1.0 - fallback_weight) * (np.square(primary_std_c) + np.square(primary_mean_c - mean_c))
            + fallback_weight * (np.square(fallback_std_c) + np.square(fallback_mean_c - mean_c))
        )
        std_c = np.maximum(np.sqrt(np.maximum(variance_c, 0.0)), self.config.min_std_c)
        return _from_celsius(mean_c, units), _from_celsius(std_c, units, scale=True)


def fit_source_gated_model(
    frame: pd.DataFrame,
    *,
    primary_model: object,
    fallback_model: object,
    config: SourceGatingConfig | None = None,
) -> tuple[SourceGatedGaussianModel, dict[str, float | str]]:
    """Fit a fallback classifier using only the provided training frame."""

    cfg = config or SourceGatingConfig()
    clean = frame.reset_index(drop=True).copy()
    units = _market_units(clean)
    truth_c = _to_celsius(clean["realized_daily_max"].to_numpy(dtype=float), units)
    primary_mean, primary_std = SourceGatedGaussianModel._predict_gaussian(primary_model, clean)
    fallback_mean, fallback_std = SourceGatedGaussianModel._predict_gaussian(fallback_model, clean)
    primary_mean_c = _to_celsius(primary_mean, units)
    primary_std_c = np.maximum(_to_celsius(primary_std, units, scale=True), cfg.min_std_c)
    fallback_mean_c = _to_celsius(fallback_mean, units)
    fallback_std_c = np.maximum(_to_celsius(fallback_std, units, scale=True), cfg.min_std_c)
    primary_crps = _gaussian_crps_vectorized(primary_mean_c, primary_std_c, truth_c)
    fallback_crps = _gaussian_crps_vectorized(fallback_mean_c, fallback_std_c, truth_c)
    valid = (
        np.isfinite(truth_c)
        & np.isfinite(primary_mean_c)
        & np.isfinite(primary_std_c)
        & np.isfinite(fallback_mean_c)
        & np.isfinite(fallback_std_c)
    )
    labels = (fallback_crps + cfg.crps_margin_c < primary_crps).astype(int)
    cities, horizons = _fit_categories(clean.loc[valid].reset_index(drop=True))
    model = SourceGatedGaussianModel(
        primary_model=primary_model,
        fallback_model=fallback_model,
        gate_model=ConstantFallbackGate(float(labels[valid].mean()) if valid.any() else 0.0),
        config=cfg,
        city_categories=cities,
        horizon_categories=horizons,
        feature_names=list(getattr(primary_model, "feature_names", [])),
    )
    features = model._gate_features(
        clean,
        units=units,
        primary_mean_c=primary_mean_c,
        primary_std_c=primary_std_c,
        fallback_mean_c=fallback_mean_c,
        fallback_std_c=fallback_std_c,
    )
    model.feature_columns = list(features.columns)
    if valid.sum() >= 20 and len(np.unique(labels[valid])) == 2:
        sample_weight: np.ndarray | None = None
        if cfg.sample_weight_mode != "uniform":
            if cfg.sample_weight_mode == "absolute_regret":
                raw_weight = np.abs(primary_crps - fallback_crps)
            else:
                raw_weight = np.maximum(primary_crps - fallback_crps, 0.0)
            sample_weight = np.clip(
                raw_weight + cfg.sample_weight_floor,
                cfg.sample_weight_floor,
                cfg.sample_weight_cap,
            )
            sample_weight = sample_weight / max(float(np.mean(sample_weight[valid])), 1e-9)
        classifier = HistGradientBoostingClassifier(
            max_iter=cfg.max_iter,
            learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.max_leaf_nodes,
            l2_regularization=cfg.l2_regularization,
            random_state=cfg.random_state,
        )
        classifier.fit(
            features.loc[valid],
            labels[valid],
            sample_weight=sample_weight[valid] if sample_weight is not None else None,
        )
        model.gate_model = classifier

    weights = model._predict_fallback_weight(
        clean,
        units=units,
        primary_mean_c=primary_mean_c,
        primary_std_c=primary_std_c,
        fallback_mean_c=fallback_mean_c,
        fallback_std_c=fallback_std_c,
    )
    diagnostics = {
        "n_fit_rows": float(valid.sum()),
        "fallback_better_rate": float(labels[valid].mean()) if valid.any() else 0.0,
        "primary_fit_crps_c": float(np.mean(primary_crps[valid])) if valid.any() else float("nan"),
        "fallback_fit_crps_c": float(np.mean(fallback_crps[valid])) if valid.any() else float("nan"),
        "mean_fallback_weight": float(np.mean(weights[valid])) if valid.any() else 0.0,
        "median_fallback_weight": float(np.median(weights[valid])) if valid.any() else 0.0,
        "sample_weight_mode": cfg.sample_weight_mode,
    }
    return model, diagnostics
