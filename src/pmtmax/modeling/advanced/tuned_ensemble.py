"""Variant-aware tuned ensemble implementations for ablation and production."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from pmtmax.modeling.design_matrix import (
    ContextualFeatureBuilder,
    recency_weights,
    temporal_validation_splits,
)

_EXPERT_ORDER = ("blend", "leadtime", "tree")


@dataclass(frozen=True)
class TunedEnsembleVariantConfig:
    name: str
    feature_mode: Literal["legacy_raw", "contextual"]
    experts: tuple[str, ...]
    combiner: Literal["fixed_average", "linear_stacker", "classifier_gate"]
    use_scale_model: bool
    output_distribution: Literal["gaussian", "gaussian_mixture"]


TUNED_ENSEMBLE_VARIANTS: dict[str, TunedEnsembleVariantConfig] = {
    "legacy_fixed2": TunedEnsembleVariantConfig(
        name="legacy_fixed2",
        feature_mode="legacy_raw",
        experts=("blend", "leadtime"),
        combiner="fixed_average",
        use_scale_model=True,
        output_distribution="gaussian",
    ),
    "context_fixed2": TunedEnsembleVariantConfig(
        name="context_fixed2",
        feature_mode="contextual",
        experts=("blend", "leadtime"),
        combiner="fixed_average",
        use_scale_model=True,
        output_distribution="gaussian",
    ),
    "context_fixed3": TunedEnsembleVariantConfig(
        name="context_fixed3",
        feature_mode="contextual",
        experts=("blend", "leadtime", "tree"),
        combiner="fixed_average",
        use_scale_model=True,
        output_distribution="gaussian",
    ),
    "context_linear2": TunedEnsembleVariantConfig(
        name="context_linear2",
        feature_mode="contextual",
        experts=("blend", "leadtime"),
        combiner="linear_stacker",
        use_scale_model=False,
        output_distribution="gaussian",
    ),
    "context_linear3": TunedEnsembleVariantConfig(
        name="context_linear3",
        feature_mode="contextual",
        experts=("blend", "leadtime", "tree"),
        combiner="linear_stacker",
        use_scale_model=False,
        output_distribution="gaussian",
    ),
    "context_linear3_scale": TunedEnsembleVariantConfig(
        name="context_linear3_scale",
        feature_mode="contextual",
        experts=("blend", "leadtime", "tree"),
        combiner="linear_stacker",
        use_scale_model=True,
        output_distribution="gaussian",
    ),
    "current_gate3_scale": TunedEnsembleVariantConfig(
        name="current_gate3_scale",
        feature_mode="contextual",
        experts=("blend", "leadtime", "tree"),
        combiner="classifier_gate",
        use_scale_model=True,
        output_distribution="gaussian_mixture",
    ),
}


def supported_tuned_ensemble_variants() -> tuple[str, ...]:
    """Return all internal tuned-ensemble ablation variants."""

    return tuple(TUNED_ENSEMBLE_VARIANTS)


def resolve_tuned_ensemble_variant(variant: str | None = None) -> TunedEnsembleVariantConfig:
    """Return the requested tuned-ensemble variant or the production default."""

    name = variant or "current_gate3_scale"
    if name not in TUNED_ENSEMBLE_VARIANTS:
        supported = ", ".join(supported_tuned_ensemble_variants())
        msg = f"Unsupported tuned_ensemble variant: {name}. Supported variants: {supported}."
        raise ValueError(msg)
    return TUNED_ENSEMBLE_VARIANTS[name]


@dataclass
class _ConstantScaleModel:
    value: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value, dtype=float)


@dataclass
class _ConstantGate:
    weights: np.ndarray

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.empty((0, len(self.weights)), dtype=float)
        return np.repeat(self.weights.reshape(1, -1), len(frame), axis=0)


@dataclass
class TunedEnsembleModel:
    """Variant-aware tuned ensemble used for benchmark ablations and production."""

    feature_names: list[str]
    split_policy: Literal["market_day", "target_day"] = "market_day"
    alpha: float = 1.0
    min_train_rows: int = 40
    variant: str | None = None

    def __post_init__(self) -> None:
        self.variant_config = resolve_tuned_ensemble_variant(self.variant)
        self.constant_mean_: float | None = None
        self.constant_std_: float = 1.0
        self.builder = ContextualFeatureBuilder(self.feature_names)
        self._legacy_medians: dict[str, float] = {}
        self._feature_columns: list[str] = []
        self._expert_columns: dict[str, list[str]] = {}
        self._expert_mean_models: dict[str, Any] = {}
        self._expert_scale_models: dict[str, Any] = {}
        self._stacker_model: Ridge | None = None
        self._global_scale_model: Ridge | _ConstantScaleModel | None = None
        self._gate_model: Pipeline | _ConstantGate | None = None
        self._gate_columns: list[str] = []
        self.diagnostics_: dict[str, float] = {}

    def _new_blend_model(self) -> Ridge:
        return Ridge(alpha=self.alpha)

    def _new_lead_model(self) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )

    def _new_tree_model(self) -> LGBMRegressor:
        return LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        )

    def _new_scale_model(self) -> Ridge:
        return Ridge(alpha=self.alpha)

    def _fit_feature_builder(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.variant_config.feature_mode == "contextual":
            self.builder.fit(frame)
            transformed = self.builder.transform(frame)
            self._feature_columns = list(transformed.columns)
            return transformed

        self._legacy_medians = {}
        for feature in self.feature_names:
            numeric = pd.to_numeric(frame.get(feature), errors="coerce") if feature in frame.columns else pd.Series(np.nan, index=frame.index)
            median = numeric.median()
            self._legacy_medians[feature] = float(median) if pd.notna(median) else 0.0
        self._feature_columns = list(self.feature_names)
        return self._legacy_transform(frame)

    def _legacy_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        data: dict[str, np.ndarray] = {}
        for feature in self.feature_names:
            if feature in frame.columns:
                numeric = pd.to_numeric(frame[feature], errors="coerce")
            else:
                numeric = pd.Series(np.nan, index=frame.index, dtype=float)
            filled = numeric.fillna(self._legacy_medians.get(feature, 0.0)).to_numpy(dtype=float)
            data[feature] = filled
        return pd.DataFrame(data, index=frame.index, dtype=float)

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.variant_config.feature_mode == "contextual":
            return self.builder.transform(frame)
        return self._legacy_transform(frame)

    def _select_expert_columns(self, transformed: pd.DataFrame) -> None:
        columns = list(transformed.columns)
        lead_columns = [
            column
            for column in columns
            if column.endswith("_model_daily_max")
            or column in {"lead_hours", "lead_hours_sq", "day_of_year_sin", "day_of_year_cos"}
            or column.startswith("horizon__")
        ]
        if not lead_columns:
            lead_columns = columns[: min(8, len(columns))]

        self._expert_columns = {}
        if "blend" in self.variant_config.experts:
            self._expert_columns["blend"] = columns
        if "leadtime" in self.variant_config.experts:
            self._expert_columns["leadtime"] = lead_columns
        if "tree" in self.variant_config.experts:
            self._expert_columns["tree"] = columns

        if self.variant_config.feature_mode == "contextual":
            self._gate_columns = [
                column
                for column in columns
                if column in {
                    "lead_hours",
                    "lead_hours_sq",
                    "day_of_year_sin",
                    "day_of_year_cos",
                    "available_feature_fraction",
                    "available_model_feature_count",
                }
                or column.startswith("city__")
                or column.startswith("horizon__")
            ]
        else:
            self._gate_columns = [
                column for column in columns if column == "lead_hours" or column.endswith("_model_daily_max")
            ]
        if not self._gate_columns:
            self._gate_columns = columns[: min(6, len(columns))]

    def _fit_expert_mean_model(self, expert: str, x: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray) -> Any:
        if expert == "blend":
            model = self._new_blend_model()
            model.fit(x[self._expert_columns[expert]], y, sample_weight=sample_weight)
            return model
        if expert == "leadtime":
            model = self._new_lead_model()
            model.fit(x[self._expert_columns[expert]], y, ridge__sample_weight=sample_weight)
            return model
        model = self._new_tree_model()
        model.fit(x[self._expert_columns[expert]], y, sample_weight=sample_weight)
        return model

    def _fit_expert_scale_model(
        self,
        expert: str,
        x: pd.DataFrame,
        residuals: np.ndarray,
        sample_weight: np.ndarray,
    ) -> Any:
        target = np.clip(np.asarray(residuals, dtype=float), 0.25, 12.0)
        if len(target) < 12:
            return _ConstantScaleModel(float(np.clip(np.mean(target), 0.25, 12.0)))
        if expert == "tree":
            model = self._new_tree_model()
            model.fit(x[self._expert_columns[expert]], target, sample_weight=sample_weight)
            return model
        model = self._new_scale_model()
        model.fit(x[self._expert_columns[expert]], target, sample_weight=sample_weight)
        return model

    def _predict_experts(self, transformed: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        means = []
        scales = []
        for expert in self.variant_config.experts:
            model = self._expert_mean_models[expert]
            expert_frame = transformed[self._expert_columns[expert]]
            means.append(np.asarray(model.predict(expert_frame), dtype=float))
            if self.variant_config.use_scale_model and expert in self._expert_scale_models:
                scale_model = self._expert_scale_models[expert]
                scales.append(np.asarray(scale_model.predict(expert_frame), dtype=float))
            else:
                scales.append(np.full(len(transformed), self.constant_std_, dtype=float))
        mean_arr = np.column_stack(means) if means else np.zeros((len(transformed), 0), dtype=float)
        scale_arr = np.column_stack(scales) if scales else np.zeros((len(transformed), 0), dtype=float)
        return mean_arr, np.clip(scale_arr, 0.25, 12.0)

    def _gate_frame(self, transformed: pd.DataFrame, means: np.ndarray, scales: np.ndarray) -> pd.DataFrame:
        gate = transformed[self._gate_columns].copy()
        for idx, expert in enumerate(self.variant_config.experts):
            gate[f"mean__{expert}"] = means[:, idx]
            gate[f"scale__{expert}"] = scales[:, idx]
        if means.size:
            gate["expert_mean_spread"] = means.std(axis=1)
            gate["expert_mean_range"] = means.max(axis=1) - means.min(axis=1)
        else:
            gate["expert_mean_spread"] = 0.0
            gate["expert_mean_range"] = 0.0
        gate["expert_scale_mean"] = scales.mean(axis=1) if scales.size else 0.0
        return gate

    def _combine_gaussian_means(self, means: np.ndarray) -> np.ndarray:
        if means.shape[1] == 0:
            return np.full(means.shape[0], self.constant_mean_ or 0.0, dtype=float)
        if self.variant_config.combiner == "fixed_average":
            return means.mean(axis=1)
        assert self._stacker_model is not None
        return np.asarray(self._stacker_model.predict(means), dtype=float)

    def _fit_global_scale_model(
        self,
        transformed: pd.DataFrame,
        combined_means: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> Ridge | _ConstantScaleModel:
        residual_target = np.clip(np.abs(y - combined_means), 0.25, 12.0)
        if len(residual_target) < 12:
            return _ConstantScaleModel(float(np.clip(np.mean(residual_target), 0.25, 12.0)))
        model = Ridge(alpha=self.alpha)
        model.fit(transformed[self._gate_columns], residual_target, sample_weight=sample_weight)
        return model

    def _fit_gate(self, gate_frame: pd.DataFrame, labels: np.ndarray) -> Pipeline | _ConstantGate:
        unique = np.unique(labels)
        if len(unique) < 2 or len(gate_frame) < 20:
            best = int(unique[0]) if len(unique) else 0
            weights = np.zeros(len(self.variant_config.experts), dtype=float)
            weights[best] = 1.0
            return _ConstantGate(weights=weights)
        gate = Pipeline(
            [
                ("scale", StandardScaler()),
                ("logit", LogisticRegression(max_iter=500)),
            ]
        )
        gate.fit(gate_frame, labels)
        return gate

    def _predict_gate_weights(self, gate_frame: pd.DataFrame) -> np.ndarray:
        if isinstance(self._gate_model, _ConstantGate):
            return self._gate_model.predict_proba(gate_frame)
        if self._gate_model is None:
            return np.repeat(np.full((1, len(self.variant_config.experts)), 1.0 / max(len(self.variant_config.experts), 1)), len(gate_frame), axis=0)

        probabilities = self._gate_model.predict_proba(gate_frame)
        logit = self._gate_model.named_steps["logit"]
        weights = np.zeros((len(gate_frame), len(self.variant_config.experts)), dtype=float)
        for column_idx, cls in enumerate(getattr(logit, "classes_", np.arange(probabilities.shape[1]))):
            weights[:, int(cls)] = probabilities[:, column_idx]
        denom = np.clip(weights.sum(axis=1, keepdims=True), 1e-6, None)
        return weights / denom

    def _record_diagnostics(
        self,
        *,
        y: np.ndarray,
        oof_means: np.ndarray,
        oof_scales: np.ndarray,
        gate_weights: np.ndarray | None = None,
        best_expert_labels: np.ndarray | None = None,
    ) -> None:
        diagnostics: dict[str, float] = {
            "variant_is_contextual": float(self.variant_config.feature_mode == "contextual"),
            "expert_count": float(len(self.variant_config.experts)),
        }
        if oof_means.size:
            for idx, expert in enumerate(self.variant_config.experts):
                diagnostics[f"expert_mae_{expert}"] = float(np.mean(np.abs(oof_means[:, idx] - y)))
            diagnostics["oracle_best_expert_mae"] = float(np.mean(np.min(np.abs(oof_means - y[:, None]), axis=1)))
        if oof_scales.size:
            diagnostics["pred_std_p50"] = float(np.percentile(oof_scales.mean(axis=1), 50))
            diagnostics["pred_std_p90"] = float(np.percentile(oof_scales.mean(axis=1), 90))
            diagnostics["pred_std_p99"] = float(np.percentile(oof_scales.mean(axis=1), 99))
        if gate_weights is not None and gate_weights.size:
            diagnostics["gate_weight_entropy_mean"] = float(
                np.mean(-(gate_weights * np.log(np.clip(gate_weights, 1e-6, None))).sum(axis=1))
            )
            if "tree" in self.variant_config.experts:
                tree_idx = self.variant_config.experts.index("tree")
                diagnostics["tree_weight_mean"] = float(np.mean(gate_weights[:, tree_idx]))
        if best_expert_labels is not None and gate_weights is not None and gate_weights.size:
            diagnostics["gate_best_expert_match_rate"] = float(np.mean(np.argmax(gate_weights, axis=1) == best_expert_labels))
        self.diagnostics_ = diagnostics

    def _single_component_fallback(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        size = len(frame)
        mean = float(self.constant_mean_ or 0.0)
        std = float(max(self.constant_std_, 0.25))
        return np.full(size, mean, dtype=float), np.full(size, std, dtype=float)

    def fit(self, frame: pd.DataFrame) -> None:
        ordered = frame.sort_values(
            [column for column in ["target_date", "decision_time_utc", "market_id", "decision_horizon"] if column in frame.columns]
        ).reset_index(drop=True)
        y = ordered["realized_daily_max"].to_numpy(dtype=float)
        self.constant_mean_ = float(np.mean(y)) if len(y) else 0.0
        self.constant_std_ = float(max(np.std(y), 0.5)) if len(y) else 1.0

        if not self.feature_names or len(ordered) < self.min_train_rows:
            return

        transformed = self._fit_feature_builder(ordered)
        self._select_expert_columns(transformed)
        weights = recency_weights(ordered)
        splits = temporal_validation_splits(ordered, split_policy=self.split_policy)
        if not splits:
            return

        oof_means = np.full((len(ordered), len(self.variant_config.experts)), np.nan, dtype=float)
        oof_scales = np.full((len(ordered), len(self.variant_config.experts)), np.nan, dtype=float)

        for train_idx, valid_idx in splits:
            x_train = transformed.iloc[train_idx].reset_index(drop=True)
            x_valid = transformed.iloc[valid_idx].reset_index(drop=True)
            y_train = y[train_idx]
            train_weights = weights[train_idx]

            fold_mean_models: dict[str, Any] = {}
            fold_scale_models: dict[str, Any] = {}
            for expert in self.variant_config.experts:
                fold_mean_models[expert] = self._fit_expert_mean_model(expert, x_train, y_train, train_weights)
                if self.variant_config.use_scale_model:
                    train_pred = np.asarray(
                        fold_mean_models[expert].predict(x_train[self._expert_columns[expert]]),
                        dtype=float,
                    )
                    fold_scale_models[expert] = self._fit_expert_scale_model(
                        expert,
                        x_train,
                        np.abs(y_train - train_pred),
                        train_weights,
                    )

            valid_means = []
            valid_scales = []
            for expert in self.variant_config.experts:
                valid_means.append(
                    np.asarray(
                        fold_mean_models[expert].predict(x_valid[self._expert_columns[expert]]),
                        dtype=float,
                    )
                )
                if self.variant_config.use_scale_model:
                    valid_scales.append(
                        np.asarray(
                            fold_scale_models[expert].predict(x_valid[self._expert_columns[expert]]),
                            dtype=float,
                        )
                    )
                else:
                    train_pred = np.asarray(
                        fold_mean_models[expert].predict(x_train[self._expert_columns[expert]]),
                        dtype=float,
                    )
                    valid_scales.append(
                        np.full(
                            len(x_valid),
                            float(np.clip(np.mean(np.abs(y_train - train_pred)), 0.25, 12.0)),
                            dtype=float,
                        )
                    )

            oof_means[valid_idx] = np.column_stack(valid_means)
            oof_scales[valid_idx] = np.clip(np.column_stack(valid_scales), 0.25, 12.0)

        valid_mask = np.isfinite(oof_means).all(axis=1) & np.isfinite(oof_scales).all(axis=1)
        if not valid_mask.any():
            return

        oof_means_valid = oof_means[valid_mask]
        oof_scales_valid = oof_scales[valid_mask]
        y_valid = y[valid_mask]
        transformed_valid = transformed.iloc[valid_mask].reset_index(drop=True)
        valid_weights = weights[valid_mask]

        gate_weights: np.ndarray | None = None
        best_expert = np.abs(oof_means_valid - y_valid[:, None]).argmin(axis=1)
        if self.variant_config.combiner == "linear_stacker":
            self._stacker_model = Ridge(alpha=self.alpha)
            self._stacker_model.fit(oof_means_valid, y_valid, sample_weight=valid_weights)
            combined_means = self._combine_gaussian_means(oof_means_valid)
        elif self.variant_config.combiner == "classifier_gate":
            gate_frame = self._gate_frame(transformed_valid, oof_means_valid, oof_scales_valid)
            self._gate_model = self._fit_gate(gate_frame, best_expert.astype(int))
            gate_weights = self._predict_gate_weights(gate_frame)
            combined_means = np.sum(gate_weights * oof_means_valid, axis=1)
        else:
            combined_means = self._combine_gaussian_means(oof_means_valid)

        if self.variant_config.combiner == "linear_stacker" and self.variant_config.use_scale_model:
            self._global_scale_model = self._fit_global_scale_model(
                transformed_valid,
                combined_means,
                y_valid,
                valid_weights,
            )
        else:
            self.constant_std_ = float(
                np.clip(
                    np.mean(np.abs(y_valid - combined_means)),
                    0.25,
                    12.0,
                )
            )

        self._record_diagnostics(
            y=y_valid,
            oof_means=oof_means_valid,
            oof_scales=oof_scales_valid,
            gate_weights=gate_weights,
            best_expert_labels=best_expert.astype(int),
        )

        self._expert_mean_models = {}
        self._expert_scale_models = {}
        for expert in self.variant_config.experts:
            self._expert_mean_models[expert] = self._fit_expert_mean_model(expert, transformed, y, weights)
            if self.variant_config.use_scale_model:
                full_pred = np.asarray(
                    self._expert_mean_models[expert].predict(transformed[self._expert_columns[expert]]),
                    dtype=float,
                )
                self._expert_scale_models[expert] = self._fit_expert_scale_model(
                    expert,
                    transformed,
                    np.abs(y - full_pred),
                    weights,
                )

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._expert_mean_models:
            return self._single_component_fallback(frame)

        transformed = self._transform(frame.reset_index(drop=True).copy())
        means, scales = self._predict_experts(transformed)

        if self.variant_config.output_distribution == "gaussian_mixture":
            gate_frame = self._gate_frame(transformed, means, scales)
            weights = self._predict_gate_weights(gate_frame)
            return weights.astype(float), means.astype(float), np.clip(scales, 0.25, 12.0).astype(float)

        combined_mean = self._combine_gaussian_means(means)
        if self.variant_config.combiner == "linear_stacker" and self.variant_config.use_scale_model and self._global_scale_model is not None:
            std = np.asarray(self._global_scale_model.predict(transformed[self._gate_columns]), dtype=float)
        elif self.variant_config.use_scale_model:
            std = scales.mean(axis=1)
        else:
            std = np.full(len(frame), self.constant_std_, dtype=float)
        return combined_mean.astype(float), np.clip(std, 0.25, 12.0).astype(float)
