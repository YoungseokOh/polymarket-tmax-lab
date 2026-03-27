"""Variant-aware deterministic-to-probabilistic neural models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pmtmax.modeling.advanced.torch_device import get_torch_device
from pmtmax.modeling.advanced.tuned_ensemble import TunedEnsembleModel
from pmtmax.modeling.baselines.gaussian_emos import GaussianEMOSModel
from pmtmax.modeling.design_matrix import (
    ContextualFeatureBuilder,
    group_id_series,
    recency_weights,
)
from pmtmax.modeling.sampling import sample_gaussian_mixture

_MIN_TRAIN_ROWS = 40


@dataclass(frozen=True)
class Det2ProbVariantConfig:
    name: str
    feature_mode: Literal["legacy_raw", "contextual", "hybrid_residual"]
    head: Literal["mean_only", "gaussian", "mdn"]
    num_components: int
    hidden_dims: tuple[int, ...]
    activation: Literal["relu", "silu"]
    use_layernorm: bool
    dropout: float
    use_recency_weights: bool
    loss_name: Literal[
        "mean_only",
        "gaussian_nll",
        "gaussian_nll_mean",
        "mdn_nll",
        "mdn_nll_mean",
        "mdn_nll_mean_entropy",
    ]
    early_stop_metric: Literal["loss", "nll", "crps"]
    # When False, skip validation split and train on all data (better for small rolling-origin sets)
    use_val_split: bool = True
    # Override the global _MIN_TRAIN_ROWS threshold for this variant (0 = use global default)
    min_train_rows_override: int = 0


DET2PROB_VARIANTS: dict[str, Det2ProbVariantConfig] = {
    "legacy_gaussian": Det2ProbVariantConfig(
        name="legacy_gaussian",
        feature_mode="legacy_raw",
        head="gaussian",
        num_components=1,
        hidden_dims=(64, 64),
        activation="relu",
        use_layernorm=False,
        dropout=0.0,
        use_recency_weights=False,
        loss_name="gaussian_nll",
        early_stop_metric="loss",
    ),
    "context_mean_only": Det2ProbVariantConfig(
        name="context_mean_only",
        feature_mode="contextual",
        head="mean_only",
        num_components=1,
        hidden_dims=(64, 64),
        activation="relu",
        use_layernorm=False,
        dropout=0.0,
        use_recency_weights=False,
        loss_name="mean_only",
        early_stop_metric="loss",
    ),
    "context_hetero_gaussian": Det2ProbVariantConfig(
        name="context_hetero_gaussian",
        feature_mode="contextual",
        head="gaussian",
        num_components=1,
        hidden_dims=(64, 64),
        activation="relu",
        use_layernorm=False,
        dropout=0.0,
        use_recency_weights=False,
        loss_name="gaussian_nll_mean",
        early_stop_metric="nll",
    ),
    "context_hetero_gaussian_recency": Det2ProbVariantConfig(
        name="context_hetero_gaussian_recency",
        feature_mode="contextual",
        head="gaussian",
        num_components=1,
        hidden_dims=(64, 64),
        activation="relu",
        use_layernorm=False,
        dropout=0.0,
        use_recency_weights=True,
        loss_name="gaussian_nll_mean",
        early_stop_metric="nll",
    ),
    "context_hetero_gaussian_deep": Det2ProbVariantConfig(
        name="context_hetero_gaussian_deep",
        feature_mode="contextual",
        head="gaussian",
        num_components=1,
        hidden_dims=(128, 128, 64),
        activation="silu",
        use_layernorm=True,
        dropout=0.1,
        use_recency_weights=False,
        loss_name="gaussian_nll_mean",
        early_stop_metric="nll",
    ),
    "context_mdn2_nll": Det2ProbVariantConfig(
        name="context_mdn2_nll",
        feature_mode="contextual",
        head="mdn",
        num_components=2,
        hidden_dims=(64, 64),
        activation="relu",
        use_layernorm=False,
        dropout=0.0,
        use_recency_weights=False,
        loss_name="mdn_nll",
        early_stop_metric="nll",
    ),
    "context_mdn3_nll_mean": Det2ProbVariantConfig(
        name="context_mdn3_nll_mean",
        feature_mode="contextual",
        head="mdn",
        num_components=3,
        hidden_dims=(64, 64),
        activation="relu",
        use_layernorm=False,
        dropout=0.0,
        use_recency_weights=False,
        loss_name="mdn_nll_mean",
        early_stop_metric="nll",
    ),
    "current_full_mdn": Det2ProbVariantConfig(
        name="current_full_mdn",
        feature_mode="contextual",
        head="mdn",
        num_components=3,
        hidden_dims=(128, 128, 64),
        activation="silu",
        use_layernorm=True,
        dropout=0.1,
        use_recency_weights=True,
        loss_name="mdn_nll_mean_entropy",
        early_stop_metric="crps",
    ),
    "hybrid_residual_gaussian": Det2ProbVariantConfig(
        name="hybrid_residual_gaussian",
        feature_mode="hybrid_residual",
        head="gaussian",
        num_components=1,
        hidden_dims=(64, 64),
        activation="relu",
        use_layernorm=False,
        dropout=0.0,
        use_recency_weights=True,
        loss_name="gaussian_nll_mean",
        early_stop_metric="nll",
    ),
    "hybrid_residual_mdn2": Det2ProbVariantConfig(
        name="hybrid_residual_mdn2",
        feature_mode="hybrid_residual",
        head="mdn",
        num_components=2,
        hidden_dims=(64, 64),
        activation="relu",
        use_layernorm=False,
        dropout=0.0,
        use_recency_weights=True,
        loss_name="mdn_nll_mean",
        early_stop_metric="nll",
    ),
    # Contextual features + Gaussian head + no validation split
    # Designed for rolling-origin backtests where training sets are small.
    # Trains on all data from row 30 onward (no 20% val holdout waste).
    "robust_gaussian": Det2ProbVariantConfig(
        name="robust_gaussian",
        feature_mode="contextual",
        head="gaussian",
        num_components=1,
        hidden_dims=(64, 64),
        activation="relu",
        use_layernorm=False,
        dropout=0.0,
        use_recency_weights=False,
        loss_name="gaussian_nll_mean",
        early_stop_metric="loss",
        use_val_split=False,
        min_train_rows_override=30,
    ),
}


def supported_det2prob_variants() -> tuple[str, ...]:
    """Return all internal det2prob ablation variants."""

    return tuple(DET2PROB_VARIANTS)


def resolve_det2prob_variant(variant: str | None = None) -> Det2ProbVariantConfig:
    """Return the requested det2prob ablation variant or the production default."""

    name = variant or "current_full_mdn"
    if name not in DET2PROB_VARIANTS:
        supported = ", ".join(supported_det2prob_variants())
        msg = f"Unsupported det2prob_nn variant: {name}. Supported variants: {supported}."
        raise ValueError(msg)
    return DET2PROB_VARIANTS[name]


class FlexibleProbNet(nn.Module):
    """Configurable probabilistic head over a compact MLP trunk."""

    def __init__(self, input_dim: int, config: Det2ProbVariantConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        prev_dim = input_dim
        activation_cls = nn.SiLU if config.activation == "silu" else nn.ReLU
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_cls())
            if config.use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            if config.dropout > 0.0:
                layers.append(nn.Dropout(p=config.dropout))
            prev_dim = hidden_dim
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        if config.head == "mean_only":
            self.mean_head = nn.Linear(prev_dim, 1)
        elif config.head == "gaussian":
            self.mean_head = nn.Linear(prev_dim, 1)
            self.scale_head = nn.Linear(prev_dim, 1)
        else:
            self.weight_head = nn.Linear(prev_dim, config.num_components)
            self.mean_head = nn.Linear(prev_dim, config.num_components)
            self.scale_head = nn.Linear(prev_dim, config.num_components)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.trunk(x)
        if self.config.head == "mean_only":
            mean = self.mean_head(hidden).squeeze(-1)
            return (mean,)
        if self.config.head == "gaussian":
            mean = self.mean_head(hidden).squeeze(-1)
            scale = torch.clamp(torch.nn.functional.softplus(self.scale_head(hidden)).squeeze(-1) + 1e-3, min=0.05, max=4.0)
            return mean, scale
        logits = self.weight_head(hidden)
        means = self.mean_head(hidden)
        scales = torch.clamp(torch.nn.functional.softplus(self.scale_head(hidden)) + 1e-3, min=0.05, max=4.0)
        return logits, means, scales


def _gaussian_nll(mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    centered = (target - mean) / std
    return 0.5 * centered.pow(2) + torch.log(std) + 0.5 * np.log(2.0 * np.pi)


def _mixture_nll(
    logits: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    log_weights = torch.log_softmax(logits, dim=-1)
    centered = (targets.unsqueeze(-1) - means) / scales
    log_component = -0.5 * centered.pow(2) - torch.log(scales) - 0.5 * np.log(2.0 * np.pi)
    return -torch.logsumexp(log_weights + log_component, dim=-1)


def _mixture_crps(
    weights: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    targets: np.ndarray,
) -> float:
    if len(targets) == 0:
        return float("inf")
    scores: list[float] = []
    for row_idx, target in enumerate(targets):
        samples = sample_gaussian_mixture(
            weights[row_idx],
            means[row_idx],
            scales[row_idx],
            num_samples=256,
            seed=42 + row_idx,
        )
        scores.append(
            float(
                np.mean(np.abs(samples - target))
                - 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
            )
        )
    return float(np.mean(scores))


@dataclass
class Det2ProbNNModel:
    """Variant-aware det2prob family covering Gaussian, MDN, and hybrid forms."""

    feature_names: list[str]
    split_policy: Literal["market_day", "target_day"] = "market_day"
    epochs: int = 60
    batch_size: int = 128
    learning_rate: float = 8e-4
    validation_fraction: float = 0.2
    early_stopping_patience: int = 8
    variant: str | None = None

    def __post_init__(self) -> None:
        self.variant_config = resolve_det2prob_variant(self.variant)
        self.constant_mean_: float | None = None
        self.constant_std_: float = 1.0
        self.builder = ContextualFeatureBuilder(self.feature_names)
        self._legacy_medians: dict[str, float] = {}
        self._input_columns: list[str] = []
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self.device = get_torch_device()
        self.network: FlexibleProbNet | None = None
        self._hybrid_gaussian: GaussianEMOSModel | None = None
        self._hybrid_tuned: TunedEnsembleModel | None = None
        self.diagnostics_: dict[str, float] = {}

    def _effective_min_train_rows(self) -> int:
        override = self.variant_config.min_train_rows_override
        return override if override > 0 else _MIN_TRAIN_ROWS

    def _validation_split(self, ordered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.variant_config.use_val_split:
            return ordered, ordered.iloc[0:0].copy()
        if len(ordered) < _MIN_TRAIN_ROWS:
            return ordered, ordered.iloc[0:0].copy()
        group_ids = group_id_series(ordered, split_policy=self.split_policy)
        unique_groups = group_ids.drop_duplicates().tolist()
        val_groups = max(int(len(unique_groups) * self.validation_fraction), 1)
        if len(unique_groups) - val_groups < 2:
            return ordered, ordered.iloc[0:0].copy()
        train_groups = set(unique_groups[:-val_groups])
        valid_groups = set(unique_groups[-val_groups:])
        train = ordered.loc[group_ids.isin(train_groups)].reset_index(drop=True).copy()
        valid = ordered.loc[group_ids.isin(valid_groups)].reset_index(drop=True).copy()
        return train, valid

    def _constant_fallback(self, y_raw: np.ndarray) -> None:
        self.constant_mean_ = float(y_raw.mean())
        self.constant_std_ = float(max(y_raw.std(), 0.5))

    def _fit_feature_builder(self, frame: pd.DataFrame) -> None:
        if self.variant_config.feature_mode != "legacy_raw":
            self.builder.fit(frame)
            return
        self._legacy_medians = {}
        for feature in self.feature_names:
            numeric = pd.to_numeric(frame.get(feature), errors="coerce") if feature in frame.columns else pd.Series(np.nan, index=frame.index)
            median = numeric.median()
            self._legacy_medians[feature] = float(median) if pd.notna(median) else 0.0

    def _legacy_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        data: dict[str, np.ndarray] = {}
        for feature in self.feature_names:
            if feature in frame.columns:
                numeric = pd.to_numeric(frame[feature], errors="coerce")
            else:
                numeric = pd.Series(np.nan, index=frame.index, dtype=float)
            data[feature] = numeric.fillna(self._legacy_medians.get(feature, 0.0)).to_numpy(dtype=float)
        return pd.DataFrame(data, index=frame.index, dtype=float)

    def _ensure_hybrid_models(self, train_frame: pd.DataFrame) -> None:
        if self._hybrid_gaussian is not None and self._hybrid_tuned is not None:
            return
        gaussian = GaussianEMOSModel(self.feature_names)
        gaussian.fit(train_frame)
        tuned = TunedEnsembleModel(self.feature_names, split_policy=self.split_policy, variant="legacy_fixed2")
        tuned.fit(train_frame)
        self._hybrid_gaussian = gaussian
        self._hybrid_tuned = tuned

    def _hybrid_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert self._hybrid_gaussian is not None
        assert self._hybrid_tuned is not None
        contextual = self.builder.transform(frame).copy()
        gauss_mean, gauss_std = self._hybrid_gaussian.predict(frame)
        tuned_prediction = self._hybrid_tuned.predict(frame)
        if len(tuned_prediction) == 2:
            tuned_mean_arr, tuned_std_arr = tuned_prediction
            tuned_mean = np.asarray(tuned_mean_arr, dtype=float)
            tuned_std = np.asarray(tuned_std_arr, dtype=float)
        else:
            weights, means, scales = tuned_prediction
            weights_arr = np.asarray(weights, dtype=float)
            means_arr = np.asarray(means, dtype=float)
            scales_arr = np.asarray(scales, dtype=float)
            tuned_mean = np.sum(weights_arr * means_arr, axis=1)
            tuned_var = np.sum(weights_arr * (scales_arr**2 + means_arr**2), axis=1) - tuned_mean**2
            tuned_std = np.sqrt(np.clip(tuned_var, 1e-6, None))
        contextual["base_gaussian_mean"] = np.asarray(gauss_mean, dtype=float)
        contextual["base_gaussian_std"] = np.asarray(gauss_std, dtype=float)
        contextual["base_tuned_mean"] = tuned_mean.astype(float)
        contextual["base_tuned_std"] = tuned_std.astype(float)
        contextual["base_mean_gap"] = contextual["base_tuned_mean"] - contextual["base_gaussian_mean"]
        return contextual

    def _feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.variant_config.feature_mode == "legacy_raw":
            transformed = self._legacy_transform(frame.reset_index(drop=True).copy())
        elif self.variant_config.feature_mode == "hybrid_residual":
            transformed = self._hybrid_transform(frame.reset_index(drop=True).copy())
        else:
            transformed = self.builder.transform(frame.reset_index(drop=True).copy())
        if not self._input_columns:
            self._input_columns = list(transformed.columns)
        for column in self._input_columns:
            if column not in transformed.columns:
                transformed[column] = 0.0
        return transformed[self._input_columns].copy()

    def _normalise_fit(self, x_raw: np.ndarray, y_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._x_mean = x_raw.mean(axis=0)
        x_std = x_raw.std(axis=0)
        self._x_std = np.where(x_std < 1e-8, 1.0, x_std)
        self._y_mean = float(y_raw.mean())
        self._y_std = float(max(y_raw.std(), 0.5))
        x = ((x_raw - self._x_mean) / self._x_std).astype(np.float32)
        y = ((y_raw - self._y_mean) / self._y_std).astype(np.float32)
        return x, y

    def _normalise_x(self, x_raw: np.ndarray) -> np.ndarray:
        assert self._x_mean is not None
        assert self._x_std is not None
        return ((x_raw - self._x_mean) / self._x_std).astype(np.float32)

    def _train_weights(self, frame: pd.DataFrame) -> np.ndarray:
        if self.variant_config.use_recency_weights:
            return recency_weights(frame)
        return np.ones(len(frame), dtype=np.float32)

    def _build_network(self, input_dim: int) -> FlexibleProbNet:
        return FlexibleProbNet(input_dim=input_dim, config=self.variant_config).to(self.device)

    def _loss_from_outputs(
        self,
        outputs: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.variant_config.head == "mean_only":
            mean = outputs[0]
            loss = torch.nn.functional.smooth_l1_loss(mean, targets, reduction="none")
            return loss, mean, None
        if self.variant_config.head == "gaussian":
            mean, std = outputs
            nll = _gaussian_nll(mean, std, targets)
            if self.variant_config.loss_name == "gaussian_nll":
                return nll, mean, std
            mean_loss = torch.nn.functional.smooth_l1_loss(mean, targets, reduction="none")
            return nll + 0.25 * mean_loss, mean, std

        logits, means, scales = outputs
        nll = _mixture_nll(logits, means, scales, targets)
        weights = torch.softmax(logits, dim=-1)
        expected_mean = torch.sum(weights * means, dim=-1)
        loss = nll
        if self.variant_config.loss_name in {"mdn_nll_mean", "mdn_nll_mean_entropy"}:
            mean_loss = torch.nn.functional.smooth_l1_loss(expected_mean, targets, reduction="none")
            loss = loss + 0.25 * mean_loss
        if self.variant_config.loss_name == "mdn_nll_mean_entropy":
            entropy = -(weights * torch.log(torch.clamp(weights, min=1e-6))).sum(dim=-1)
            loss = loss - 0.01 * entropy
        return loss, expected_mean, scales.mean(dim=-1)

    def _evaluate_metric(
        self,
        outputs: tuple[torch.Tensor, ...],
        valid_frame: pd.DataFrame,
        valid_weights: torch.Tensor,
        valid_targets: torch.Tensor,
        weighted_valid_loss: torch.Tensor,
    ) -> float:
        if self.variant_config.early_stop_metric == "loss":
            return float(weighted_valid_loss.detach().cpu().item())
        if self.variant_config.early_stop_metric == "nll":
            return float(weighted_valid_loss.detach().cpu().item())

        logits, means, scales = outputs
        weights = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        means_np = means.detach().cpu().numpy() * self._y_std + self._y_mean
        scales_np = scales.detach().cpu().numpy() * self._y_std
        targets_np = valid_frame["realized_daily_max"].to_numpy(dtype=float)
        valid_crps = _mixture_crps(weights, means_np, scales_np, targets_np)
        return valid_crps + 0.05 * float(weighted_valid_loss.detach().cpu().item())

    def fit(self, frame: pd.DataFrame) -> None:
        ordered = frame.sort_values(
            [column for column in ["target_date", "decision_time_utc", "market_id", "decision_horizon"] if column in frame.columns]
        ).reset_index(drop=True)
        y_raw = ordered["realized_daily_max"].to_numpy(dtype=np.float32)
        self.constant_mean_ = float(y_raw.mean()) if len(y_raw) else 0.0
        self.constant_std_ = float(max(y_raw.std(), 0.5)) if len(y_raw) else 1.0

        min_train = self._effective_min_train_rows()
        if not self.feature_names or len(ordered) < min_train:
            return

        self._fit_feature_builder(ordered)
        train_frame, valid_frame = self._validation_split(ordered)
        if len(train_frame) < min_train:
            self._constant_fallback(y_raw)
            return

        if self.variant_config.feature_mode == "hybrid_residual":
            self.builder.fit(train_frame)
            self._ensure_hybrid_models(train_frame)
        x_train_raw = self._feature_frame(train_frame).to_numpy(dtype=np.float32)
        self._input_columns = list(self._feature_frame(train_frame).columns)
        y_train_raw = train_frame["realized_daily_max"].to_numpy(dtype=np.float32)
        x_train, y_train = self._normalise_fit(x_train_raw, y_train_raw)

        if self.variant_config.feature_mode == "hybrid_residual":
            x_valid_raw = self._feature_frame(valid_frame).to_numpy(dtype=np.float32)
        else:
            x_valid_raw = self._feature_frame(valid_frame).to_numpy(dtype=np.float32)
        x_valid = self._normalise_x(x_valid_raw) if len(valid_frame) else np.empty((0, x_train.shape[1]), dtype=np.float32)
        y_valid = ((valid_frame["realized_daily_max"].to_numpy(dtype=np.float32) - self._y_mean) / self._y_std).astype(np.float32)

        train_weights_np = self._train_weights(train_frame)
        valid_weights_np = self._train_weights(valid_frame) if not valid_frame.empty else np.empty((0,), dtype=np.float32)

        self.network = self._build_network(x_train.shape[1])
        train_x = torch.tensor(x_train, device=self.device)
        train_y = torch.tensor(y_train, device=self.device)
        train_w = torch.tensor(train_weights_np, device=self.device)
        dataset = TensorDataset(train_x, train_y, train_w)
        generator = torch.Generator()
        generator.manual_seed(42)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )

        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        best_state = {key: value.detach().cpu().clone() for key, value in self.network.state_dict().items()}
        best_metric = float("inf")
        patience = 0
        stable = True

        valid_x_tensor = torch.tensor(x_valid, device=self.device) if len(x_valid) else None
        valid_y_tensor = torch.tensor(y_valid, device=self.device) if len(y_valid) else None
        valid_w_tensor = torch.tensor(valid_weights_np, device=self.device) if len(valid_weights_np) else None

        for _ in range(self.epochs):
            if not stable:
                break
            self.network.train()
            last_weighted_loss = torch.tensor(0.0, device=self.device)
            for batch_x, batch_y, batch_w in loader:
                outputs = self.network(batch_x)
                loss, _, _ = self._loss_from_outputs(outputs, batch_y)
                weighted_loss = torch.sum(loss * batch_w) / torch.clamp(batch_w.sum(), min=1e-6)
                if not torch.isfinite(weighted_loss):
                    stable = False
                    break
                optimizer.zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                last_weighted_loss = weighted_loss

            if not stable:
                break

            eval_metric = float(last_weighted_loss.detach().cpu().item())
            if valid_x_tensor is not None and valid_y_tensor is not None and valid_w_tensor is not None and len(valid_frame) > 0:
                self.network.eval()
                with torch.no_grad():
                    valid_outputs = self.network(valid_x_tensor)
                    valid_loss, _, _ = self._loss_from_outputs(valid_outputs, valid_y_tensor)
                    weighted_valid_loss = torch.sum(valid_loss * valid_w_tensor) / torch.clamp(valid_w_tensor.sum(), min=1e-6)
                    eval_metric = self._evaluate_metric(
                        valid_outputs,
                        valid_frame,
                        valid_w_tensor,
                        valid_y_tensor,
                        weighted_valid_loss,
                    )
                self.network.train()

            if eval_metric + 1e-6 < best_metric:
                best_metric = eval_metric
                best_state = {key: value.detach().cpu().clone() for key, value in self.network.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    break

        if not stable or self.network is None:
            self._constant_fallback(y_raw)
            self.network = None
            self.diagnostics_ = {"fallback_used": 1.0}
            return

        self.network.load_state_dict(best_state)

        diagnostics: dict[str, float] = {"fallback_used": 0.0}
        eval_frame = valid_frame if not valid_frame.empty else train_frame
        eval_raw = self._feature_frame(eval_frame).to_numpy(dtype=np.float32)
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(torch.tensor(self._normalise_x(eval_raw), device=self.device))
        if self.variant_config.head == "mean_only":
            mean = outputs[0].detach().cpu().numpy() * self._y_std + self._y_mean
            residual = np.abs(eval_frame["realized_daily_max"].to_numpy(dtype=float) - mean)
            self.constant_std_ = float(np.clip(np.mean(residual), 0.25, 12.0))
            diagnostics["pred_std_p50"] = float(self.constant_std_)
            diagnostics["pred_std_p90"] = float(self.constant_std_)
            diagnostics["pred_std_p99"] = float(self.constant_std_)
        elif self.variant_config.head == "gaussian":
            _, std = outputs
            std_np = np.clip(std.detach().cpu().numpy() * self._y_std, 0.25, 12.0)
            diagnostics["pred_std_p50"] = float(np.percentile(std_np, 50))
            diagnostics["pred_std_p90"] = float(np.percentile(std_np, 90))
            diagnostics["pred_std_p99"] = float(np.percentile(std_np, 99))
        else:
            logits, _, scales = outputs
            weights = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            scales_np = np.clip(scales.detach().cpu().numpy() * self._y_std, 0.25, 12.0)
            component_entropy = -(weights * np.log(np.clip(weights, 1e-6, None))).sum(axis=1)
            diagnostics["mixture_entropy_mean"] = float(np.mean(component_entropy))
            diagnostics["effective_components_mean"] = float(np.mean(np.exp(component_entropy)))
            diagnostics["component_weight_max_mean"] = float(np.mean(weights.max(axis=1)))
            diagnostics["pred_std_p50"] = float(np.percentile(scales_np.mean(axis=1), 50))
            diagnostics["pred_std_p90"] = float(np.percentile(scales_np.mean(axis=1), 90))
            diagnostics["pred_std_p99"] = float(np.percentile(scales_np.mean(axis=1), 99))
        self.diagnostics_ = diagnostics

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.network is None:
            size = len(frame)
            return (
                np.full(size, self.constant_mean_, dtype=float),
                np.full(size, self.constant_std_, dtype=float),
            )

        x_raw = self._feature_frame(frame).to_numpy(dtype=np.float32)
        x = torch.tensor(self._normalise_x(x_raw), device=self.device)
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(x)

        if self.variant_config.head == "mean_only":
            mean = outputs[0].detach().cpu().numpy() * self._y_std + self._y_mean
            std = np.full(len(frame), self.constant_std_, dtype=float)
            return mean.astype(float), std.astype(float)
        if self.variant_config.head == "gaussian":
            mean, std = outputs
            mean_np = mean.detach().cpu().numpy() * self._y_std + self._y_mean
            std_np = np.clip(std.detach().cpu().numpy() * self._y_std, 0.25, 12.0)
            return mean_np.astype(float), std_np.astype(float)

        logits, means, scales = outputs
        weights = torch.softmax(logits, dim=-1)
        weights_np = weights.detach().cpu().numpy().astype(float)
        means_np = (means.detach().cpu().numpy() * self._y_std + self._y_mean).astype(float)
        scales_np = np.clip(scales.detach().cpu().numpy() * self._y_std, 0.25, 12.0).astype(float)
        return weights_np, means_np, scales_np
