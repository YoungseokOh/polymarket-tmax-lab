"""Standalone LightGBM EMOS model with contextual features and heteroscedastic output."""

from __future__ import annotations

from dataclasses import dataclass
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
        min_child_samples=5,
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
        min_child_samples=5,
        use_recency_weights=True,
        recency_half_life_days=90.0,
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


def _nwp_spread_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-model NWP spread features that capture forecast uncertainty."""

    available = [c for c in _NWP_MAX_COLS if c in frame.columns]
    result: dict[str, pd.Series] = {}
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
    out = pd.DataFrame(result, index=frame.index)
    # Ensure float dtype — columns may be object when inputs contain all-NaN
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out


def _new_lgbm(cfg: LgbmEMOSVariantConfig) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_child_samples=cfg.min_child_samples,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
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

    def __post_init__(self) -> None:
        self._variant_config = resolve_lgbm_emos_variant(self.variant)
        self.variant = self._variant_config.name
        self.builder = ContextualFeatureBuilder(self.feature_names)
        self._mean_model: LGBMRegressor | None = None
        self._scale_model: LGBMRegressor | None = None
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

        self.builder.fit(ordered)
        x_base = self.builder.transform(ordered)
        x_extra = _nwp_spread_features(ordered)
        x = pd.concat([x_base, x_extra], axis=1)

        cfg = self._variant_config
        sw = recency_weights(ordered, half_life_days=cfg.recency_half_life_days) if cfg.use_recency_weights else None

        # Fit mean model on full training set
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
            "oof_mae": float(np.mean(oof_residuals)),
            "oof_residual_p90": float(np.percentile(oof_residuals, 90)),
        }

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

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self._mean_model is None or self._scale_model is None:
            size = len(frame)
            return np.full(size, self._constant_mean, dtype=float), np.full(size, self._constant_std, dtype=float)

        frame_reset = frame.reset_index(drop=True)
        x_base = self.builder.transform(frame_reset)
        x_extra = _nwp_spread_features(frame_reset)
        x = pd.concat([x_base, x_extra], axis=1)
        mean = np.asarray(self._mean_model.predict(x), dtype=float)
        scale = np.clip(np.asarray(self._scale_model.predict(x), dtype=float), 0.5, None)
        return mean, scale
