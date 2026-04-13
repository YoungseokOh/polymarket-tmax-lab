"""Standalone LightGBM EMOS model with contextual features and heteroscedastic output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from pmtmax.modeling.design_matrix import (
    ContextualFeatureBuilder,
    recency_weights,
    temporal_validation_splits,
)


@dataclass(frozen=True)
class LgbmEMOSVariantConfig:
    name: str
    n_estimators: int
    num_leaves: int
    max_depth: int
    learning_rate: float
    min_child_samples: int
    use_recency_weights: bool
    recency_half_life_days: float
    use_oof_scale: bool = True  # False → faster (in-sample scale), True → honest OOF scale
    subsample_freq: int = 0  # 0 = disabled (default); 1 = enable row subsampling every tree
    subsample: float = 0.9  # Row subsampling ratio per tree
    colsample_bytree: float = 0.9  # Column subsampling ratio per tree
    reg_alpha: float = 0.1  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    use_quantile_loss: bool = False  # True → fit q10/q50/q90 quantile models; scale = (q90-q10)/2.56
    use_neighbor_delta: bool = False  # True → add nwp_vs_neighbor_delta/spread_ratio to feature matrix
    fixed_std: float | None = None  # If set, skip scale model entirely and use this constant std
    drop_dead_features: bool = False  # True → exclude xmod_*, kma_gdps_*, gfs_seamless_* (near-zero importance)
    use_city_lat: bool = False  # True → add continuous city_latitude feature to design matrix


LGBM_EMOS_VARIANTS: dict[str, LgbmEMOSVariantConfig] = {
    "default": LgbmEMOSVariantConfig(
        name="default",
        n_estimators=300,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=8,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=True,
    ),
    "fast": LgbmEMOSVariantConfig(
        name="fast",
        n_estimators=300,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=8,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=False,
    ),
    "high_capacity": LgbmEMOSVariantConfig(
        name="high_capacity",
        n_estimators=500,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.03,
        min_child_samples=20,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=True,
    ),
    "recency": LgbmEMOSVariantConfig(
        name="recency",
        n_estimators=300,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=8,
        use_recency_weights=True,
        recency_half_life_days=90.0,
    ),
    "recency_high": LgbmEMOSVariantConfig(
        name="recency_high",
        n_estimators=500,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.03,
        min_child_samples=20,
        use_recency_weights=True,
        recency_half_life_days=90.0,
    ),
    "recency_fast": LgbmEMOSVariantConfig(
        name="recency_fast",
        n_estimators=300,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=8,
        use_recency_weights=True,
        recency_half_life_days=90.0,
        use_oof_scale=False,
    ),
    "high_capacity_fast": LgbmEMOSVariantConfig(
        name="high_capacity_fast",
        n_estimators=500,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.03,
        min_child_samples=5,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=False,
    ),
    "recency_short": LgbmEMOSVariantConfig(
        name="recency_short",
        n_estimators=300,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=8,
        use_recency_weights=True,
        recency_half_life_days=30.0,
        use_oof_scale=False,
    ),
    "recency_long": LgbmEMOSVariantConfig(
        name="recency_long",
        n_estimators=300,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=8,
        use_recency_weights=True,
        recency_half_life_days=180.0,
        use_oof_scale=False,
    ),
    "neighbor_capacity_fast": LgbmEMOSVariantConfig(
        name="neighbor_capacity_fast",
        n_estimators=500,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.03,
        min_child_samples=20,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=False,
    ),
    # --- autoresearch round 2: quantile + neighbor-delta features ---
    "quantile_fast": LgbmEMOSVariantConfig(
        name="quantile_fast",
        n_estimators=500,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.03,
        min_child_samples=5,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=False,
        use_quantile_loss=True,
    ),
    "neighbor_delta_fast": LgbmEMOSVariantConfig(
        name="neighbor_delta_fast",
        n_estimators=500,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.03,
        min_child_samples=5,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=False,
        use_neighbor_delta=True,
    ),
    "quantile_neighbor_fast": LgbmEMOSVariantConfig(
        name="quantile_neighbor_fast",
        n_estimators=500,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.03,
        min_child_samples=5,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=False,
        use_quantile_loss=True,
        use_neighbor_delta=True,
    ),
    # --- autoresearch round 3: neighbor_delta combos ---
    "recency_neighbor_fast": LgbmEMOSVariantConfig(
        name="recency_neighbor_fast",
        n_estimators=500,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.03,
        min_child_samples=5,
        use_recency_weights=True,
        recency_half_life_days=30.0,
        use_oof_scale=False,
        use_neighbor_delta=True,
    ),
    "ultra_neighbor_fast": LgbmEMOSVariantConfig(
        name="ultra_neighbor_fast",
        n_estimators=800,
        num_leaves=127,
        max_depth=10,
        learning_rate=0.02,
        min_child_samples=5,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=False,
        use_neighbor_delta=True,
    ),
    "slow_neighbor_fast": LgbmEMOSVariantConfig(
        name="slow_neighbor_fast",
        n_estimators=1000,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.01,
        min_child_samples=5,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=False,
        use_neighbor_delta=True,
    ),
    # --- autoresearch A/B/C candidates ---
    "ultra_capacity_fast": LgbmEMOSVariantConfig(
        name="ultra_capacity_fast",
        n_estimators=800,
        num_leaves=127,
        max_depth=10,
        learning_rate=0.02,
        min_child_samples=5,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=False,
    ),
    "sampled_fast": LgbmEMOSVariantConfig(
        name="sampled_fast",
        n_estimators=500,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.03,
        min_child_samples=5,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=False,
        subsample_freq=1,
    ),
    "slow_deep_fast": LgbmEMOSVariantConfig(
        name="slow_deep_fast",
        n_estimators=1000,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.01,
        min_child_samples=5,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=False,
    ),
    # Autoresearch round 5: recover CRPS to 0.47 on restored 5388-row dataset
    "ultra_high_neighbor_fast": LgbmEMOSVariantConfig(
        name="ultra_high_neighbor_fast",
        n_estimators=800,
        num_leaves=127,
        max_depth=10,
        learning_rate=0.02,
        min_child_samples=5,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=False,
        use_neighbor_delta=True,
    ),
    "recency_high_45d_fast": LgbmEMOSVariantConfig(
        name="recency_high_45d_fast",
        n_estimators=600,
        num_leaves=95,
        max_depth=9,
        learning_rate=0.025,
        min_child_samples=5,
        use_recency_weights=True,
        recency_half_life_days=45.0,
        use_oof_scale=False,
        use_neighbor_delta=True,
    ),
    "recency_high_90d_fast": LgbmEMOSVariantConfig(
        name="recency_high_90d_fast",
        n_estimators=600,
        num_leaves=95,
        max_depth=9,
        learning_rate=0.025,
        min_child_samples=5,
        use_recency_weights=True,
        recency_half_life_days=90.0,
        use_oof_scale=False,
        use_neighbor_delta=True,
    ),
    "mega_neighbor_fast": LgbmEMOSVariantConfig(
        name="mega_neighbor_fast",
        n_estimators=1000,
        num_leaves=150,
        max_depth=10,
        learning_rate=0.015,
        min_child_samples=5,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=False,
        use_neighbor_delta=True,
    ),
    # Autoresearch round 6: OOF scale — structural fix for sigma collapse
    "high_neighbor_oof": LgbmEMOSVariantConfig(
        name="high_neighbor_oof",
        n_estimators=600,
        num_leaves=95,
        max_depth=9,
        learning_rate=0.025,
        min_child_samples=5,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=True,
        use_neighbor_delta=True,
    ),
    "ultra_high_neighbor_oof": LgbmEMOSVariantConfig(
        name="ultra_high_neighbor_oof",
        n_estimators=800,
        num_leaves=127,
        max_depth=10,
        learning_rate=0.02,
        min_child_samples=5,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=True,
        use_neighbor_delta=True,
    ),
    "mega_neighbor_oof": LgbmEMOSVariantConfig(
        name="mega_neighbor_oof",
        n_estimators=1000,
        num_leaves=150,
        max_depth=10,
        learning_rate=0.015,
        min_child_samples=5,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=True,
        use_neighbor_delta=True,
    ),
    # Autoresearch round 4: push CRPS below 0.45
    "recency_tight_fast": LgbmEMOSVariantConfig(
        name="recency_tight_fast",
        n_estimators=500,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.03,
        min_child_samples=5,
        use_recency_weights=True,
        recency_half_life_days=15.0,
        use_oof_scale=False,
        use_neighbor_delta=True,
    ),
    "recency_mid_fast": LgbmEMOSVariantConfig(
        name="recency_mid_fast",
        n_estimators=500,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.03,
        min_child_samples=5,
        use_recency_weights=True,
        recency_half_life_days=20.0,
        use_oof_scale=False,
        use_neighbor_delta=True,
    ),
    "high_neighbor_fast": LgbmEMOSVariantConfig(
        name="high_neighbor_fast",
        n_estimators=600,
        num_leaves=95,
        max_depth=9,
        learning_rate=0.025,
        min_child_samples=5,
        use_recency_weights=False,
        recency_half_life_days=90.0,
        use_oof_scale=False,
        use_neighbor_delta=True,
    ),
    "recency_high_fast": LgbmEMOSVariantConfig(
        name="recency_high_fast",
        n_estimators=600,
        num_leaves=95,
        max_depth=9,
        learning_rate=0.025,
        min_child_samples=5,
        use_recency_weights=True,
        recency_half_life_days=30.0,
        use_oof_scale=False,
        use_neighbor_delta=True,
    ),
    # Scale fix variants: address the 0.5 constant-std collapse in recency_neighbor_fast
    "recency_neighbor_oof": LgbmEMOSVariantConfig(
        name="recency_neighbor_oof",
        n_estimators=500,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.03,
        min_child_samples=5,
        use_recency_weights=True,
        recency_half_life_days=30.0,
        use_oof_scale=True,
        use_neighbor_delta=True,
    ),
    "recency_neighbor_std3": LgbmEMOSVariantConfig(
        name="recency_neighbor_std3",
        n_estimators=500,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.03,
        min_child_samples=5,
        use_recency_weights=True,
        recency_half_life_days=30.0,
        use_oof_scale=False,
        use_neighbor_delta=True,
        fixed_std=3.0,
    ),
}


def supported_lgbm_emos_variants() -> tuple[str, ...]:
    """Return all supported lgbm_emos ablation variants."""

    return tuple(LGBM_EMOS_VARIANTS)


def resolve_lgbm_emos_variant(variant: str | None = None) -> LgbmEMOSVariantConfig:
    """Return the config for a named variant, defaulting to 'default'."""

    key = variant or "default"
    if key not in LGBM_EMOS_VARIANTS:
        supported = ", ".join(LGBM_EMOS_VARIANTS)
        msg = f"Unsupported lgbm_emos variant: {key}. Supported: {supported}"
        raise ValueError(msg)
    return LGBM_EMOS_VARIANTS[key]


_NWP_MAX_COLS = (
    "ecmwf_ifs025_model_daily_max",
    "ecmwf_aifs025_single_model_daily_max",
    "kma_gdps_model_daily_max",
    "gfs_seamless_model_daily_max",
)

# Variable suffixes for cross-model disagreement features (uncertainty signals)
_NWP_PREFIXES = (
    "ecmwf_ifs025",
    "ecmwf_aifs025_single",
    "kma_gdps",
    "gfs_seamless",
)
_CROSS_MODEL_VARS = (
    "wind_speed_mean",
    "dew_point_mean",
    "diurnal_amplitude",
    "midday_temp",
    "model_daily_min",
    "model_daily_mean",
    "cloud_cover_mean",
)


def _nwp_spread_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-model NWP spread features that capture forecast uncertainty."""

    result: dict[str, pd.Series] = {}

    # Max-temperature ensemble statistics
    available = [c for c in _NWP_MAX_COLS if c in frame.columns]
    if len(available) >= 2:
        vals = frame[available]
        result["nwp_spread"] = vals.max(axis=1) - vals.min(axis=1)
        result["nwp_std"] = vals.std(axis=1, ddof=0).fillna(0.0)
        result["nwp_ens_mean"] = vals.mean(axis=1)
    ifs = "ecmwf_ifs025_model_daily_max"
    gfs = "gfs_seamless_model_daily_max"
    aifs = "ecmwf_aifs025_single_model_daily_max"
    if ifs in frame.columns and gfs in frame.columns:
        result["ecmwf_gfs_diff"] = frame[ifs] - frame[gfs]
    if ifs in frame.columns and aifs in frame.columns:
        result["ecmwf_aifs_diff"] = frame[ifs] - frame[aifs]

    # Cross-model disagreement for other weather variables (uncertainty signals)
    for var in _CROSS_MODEL_VARS:
        cols = [f"{pfx}_{var}" for pfx in _NWP_PREFIXES if f"{pfx}_{var}" in frame.columns]
        if len(cols) >= 2:
            vals = frame[cols]
            result[f"xmod_spread__{var}"] = vals.max(axis=1) - vals.min(axis=1)
            result[f"xmod_std__{var}"] = vals.std(axis=1, ddof=0).fillna(0.0)

    out = pd.DataFrame(result, index=frame.index)
    # Ensure float dtype — columns may be object when inputs contain all-NaN
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out


def _new_lgbm(cfg: LgbmEMOSVariantConfig, alpha: float | None = None) -> LGBMRegressor:
    kwargs: dict[str, object] = {}
    if alpha is not None:
        kwargs["objective"] = "quantile"
        kwargs["alpha"] = alpha
    return LGBMRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=cfg.subsample_freq,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=42,
        verbose=-1,
        **kwargs,
    )


@dataclass
class LgbmEMOSModel:
    """Standalone LightGBM post-processor with contextual features and Gaussian output.

    Trains a LGBM regressor for the mean and a separate LGBM on out-of-fold
    absolute residuals for the scale, producing a heteroscedastic Gaussian.
    The contextual feature builder adds city/horizon dummies, seasonal encodings,
    availability flags, and lead-time features.
    """

    feature_names: list[str]
    split_policy: Literal["market_day", "target_day"] = "market_day"
    min_train_rows: int = 40
    variant: str | None = None
    variant_config: LgbmEMOSVariantConfig | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._variant_config = self.variant_config or resolve_lgbm_emos_variant(self.variant)
        self.variant = self._variant_config.name
        self.builder = ContextualFeatureBuilder(self.feature_names)
        self._mean_model: LGBMRegressor | None = None
        self._scale_model: LGBMRegressor | None = None
        self._q10_model: LGBMRegressor | None = None
        self._q90_model: LGBMRegressor | None = None
        self._constant_mean: float = 0.0
        self._constant_std: float = 1.0
        self.diagnostics_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> None:
        sort_cols = [c for c in ["target_date", "decision_time_utc", "market_id", "decision_horizon"] if c in frame.columns]
        ordered = frame.sort_values(sort_cols).reset_index(drop=True) if sort_cols else frame.reset_index(drop=True).copy()
        y = ordered["realized_daily_max"].to_numpy(dtype=float)
        self._constant_mean = float(np.mean(y)) if len(y) else 0.0
        self._constant_std = float(max(np.std(y), 0.5)) if len(y) else 1.0

        if not self.feature_names or len(ordered) < self.min_train_rows:
            return

        cfg = self._variant_config

        # Prune dead-weight features before fitting the builder
        active_feature_names = self.feature_names
        if cfg.drop_dead_features:
            _dead_prefixes = ("xmod_", "kma_gdps_", "gfs_seamless_", "ecmwf_aifs025_single_")
            active_feature_names = [
                f for f in self.feature_names
                if not any(f.startswith(p) for p in _dead_prefixes)
            ]
        self.builder = ContextualFeatureBuilder(active_feature_names, use_city_lat=cfg.use_city_lat)
        self.builder.fit(ordered)
        x_base = self.builder.transform(ordered)
        x_extra = _nwp_spread_features(ordered)
        x = pd.concat([x_base, x_extra], axis=1)

        if cfg.use_neighbor_delta:
            x = self._add_neighbor_delta(x, ordered)
        sw = recency_weights(ordered, half_life_days=cfg.recency_half_life_days) if cfg.use_recency_weights else None

        if cfg.use_quantile_loss:
            # Quantile regression mode: fit q10/q50/q90 models directly
            import sys
            print(f"[lgbm_emos] fitting q50 on {len(x)} rows ...", flush=True, file=sys.stderr)
            self._mean_model = _new_lgbm(cfg, alpha=0.5)
            self._mean_model.fit(x, y, sample_weight=sw)
            print(f"[lgbm_emos] fitting q10 ...", flush=True, file=sys.stderr)
            self._q10_model = _new_lgbm(cfg, alpha=0.1)
            self._q10_model.fit(x, y, sample_weight=sw)
            print(f"[lgbm_emos] fitting q90 ...", flush=True, file=sys.stderr)
            self._q90_model = _new_lgbm(cfg, alpha=0.9)
            self._q90_model.fit(x, y, sample_weight=sw)
            print(f"[lgbm_emos] quantile fit done", flush=True, file=sys.stderr)
            self._scale_model = None
        elif cfg.fixed_std is not None:
            # Fixed-std mode: mean model only, scale = constant (skips scale model entirely)
            self._mean_model = _new_lgbm(cfg)
            self._mean_model.fit(x, y, sample_weight=sw)
            self._constant_std = cfg.fixed_std
            self._scale_model = None
        else:
            # Standard mode: fit mean model + scale model on residuals
            self._mean_model = _new_lgbm(cfg)
            self._mean_model.fit(x, y, sample_weight=sw)

            # Compute scale targets: OOF residuals (honest but slow) or in-sample (fast)
            if cfg.use_oof_scale:
                oof_residuals = self._oof_residuals(ordered, x, y, sw)
            else:
                assert self._mean_model is not None
                preds = np.asarray(self._mean_model.predict(x), dtype=float)
                oof_residuals = np.clip(np.abs(y - preds), 0.25, 12.0)

            # Fit scale model on |OOF residuals|
            self._scale_model = _new_lgbm(cfg)
            self._scale_model.fit(x, oof_residuals, sample_weight=sw)

        self.diagnostics_ = {
            "train_rows": float(len(ordered)),
            "feature_count": float(len(x.columns)),
        }
        if not cfg.use_quantile_loss and cfg.fixed_std is None:
            self.diagnostics_["oof_mae"] = float(np.mean(oof_residuals))
            self.diagnostics_["oof_residual_p90"] = float(np.percentile(oof_residuals, 90))

    def _oof_residuals(
        self,
        ordered: pd.DataFrame,
        x: pd.DataFrame,
        y: np.ndarray,
        sw: np.ndarray | None,
    ) -> np.ndarray:
        """Return clipped |OOF residuals|, falling back to in-sample when splits are unavailable."""

        splits = temporal_validation_splits(ordered, split_policy=self.split_policy)
        if not splits:
            assert self._mean_model is not None
            preds = np.asarray(self._mean_model.predict(x), dtype=float)
            return np.clip(np.abs(y - preds), 0.25, 12.0)

        cfg = self._variant_config
        oof_preds = np.full(len(y), np.nan, dtype=float)
        for train_idx, valid_idx in splits:
            x_tr = x.iloc[train_idx].reset_index(drop=True)
            y_tr = y[train_idx]
            sw_tr = sw[train_idx] if sw is not None else None
            fold_model = _new_lgbm(cfg)
            fold_model.fit(x_tr, y_tr, sample_weight=sw_tr)
            oof_preds[valid_idx] = fold_model.predict(x.iloc[valid_idx].reset_index(drop=True))

        # Any rows not covered by OOF get in-sample predictions
        missing = np.isnan(oof_preds)
        if missing.any():
            assert self._mean_model is not None
            oof_preds[missing] = np.asarray(self._mean_model.predict(x.iloc[missing].reset_index(drop=True)), dtype=float)

        return np.clip(np.abs(y - oof_preds), 0.25, 12.0)

    @staticmethod
    def _add_neighbor_delta(x: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
        """Append NWP-vs-neighbor delta features to the feature matrix."""
        extra: dict[str, pd.Series] = {}
        if "nwp_ens_mean" in x.columns and "neighbor_mean_temp" in frame.columns:
            neighbor = pd.to_numeric(frame["neighbor_mean_temp"], errors="coerce").fillna(0.0)
            extra["nwp_vs_neighbor_delta"] = x["nwp_ens_mean"].values - neighbor.values
        if "nwp_std" in x.columns and "neighbor_spread" in frame.columns:
            nb_spread = pd.to_numeric(frame["neighbor_spread"], errors="coerce").fillna(0.0)
            extra["nwp_vs_neighbor_spread_ratio"] = x["nwp_std"].values - nb_spread.values
        if extra:
            return pd.concat([x, pd.DataFrame(extra, index=x.index)], axis=1)
        return x

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self._mean_model is None:
            size = len(frame)
            return np.full(size, self._constant_mean, dtype=float), np.full(size, self._constant_std, dtype=float)

        frame_reset = frame.reset_index(drop=True)
        x_base = self.builder.transform(frame_reset)
        x_extra = _nwp_spread_features(frame_reset)
        x = pd.concat([x_base, x_extra], axis=1)
        if self._variant_config.use_neighbor_delta:
            x = self._add_neighbor_delta(x, frame_reset)
        mean = np.asarray(self._mean_model.predict(x), dtype=float)

        if self._variant_config.use_quantile_loss and self._q10_model is not None and self._q90_model is not None:
            q10 = np.asarray(self._q10_model.predict(x), dtype=float)
            q90 = np.asarray(self._q90_model.predict(x), dtype=float)
            # σ = (q90 - q10) / 2.56  (for a standard normal, q90-q10 = 2.56σ)
            scale = np.clip((q90 - q10) / 2.56, 0.5, None)
        else:
            if self._scale_model is None:
                scale = np.full(len(frame), self._constant_std, dtype=float)
            else:
                scale = np.clip(np.asarray(self._scale_model.predict(x), dtype=float), 2.0, None)

        return mean, scale
