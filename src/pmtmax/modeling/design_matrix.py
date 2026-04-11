"""Shared contextual design-matrix helpers for tabular forecasting models."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


def _safe_token(value: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return token or "unknown"


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ["target_date", "decision_time_utc", "market_id", "decision_horizon"] if column in frame.columns]
    if not sort_columns:
        return frame.reset_index(drop=True).copy()
    return frame.sort_values(sort_columns).reset_index(drop=True).copy()


def group_id_series(
    frame: pd.DataFrame,
    split_policy: Literal["market_day", "target_day"],
) -> pd.Series:
    """Return grouped chronological IDs for leakage-safe temporal splits."""

    if split_policy == "market_day" and {"market_id", "target_date"}.issubset(frame.columns):
        return frame[["market_id", "target_date"]].astype(str).agg("|".join, axis=1)
    if "target_date" in frame.columns:
        return frame["target_date"].astype(str)
    return pd.Series([str(index) for index in range(len(frame))], index=frame.index, dtype=object)


def temporal_validation_splits(
    frame: pd.DataFrame,
    *,
    split_policy: Literal["market_day", "target_day"],
    n_splits: int = 4,
    min_train_groups: int = 16,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding-window train/validation index splits over chronological groups."""

    ordered = _ordered_frame(frame)
    group_ids = group_id_series(ordered, split_policy=split_policy)
    unique_groups = group_ids.drop_duplicates().tolist()
    if len(unique_groups) < 2:
        return []

    validation_groups = max(1, len(unique_groups) // max(n_splits + 1, 2))
    split_starts = list(
        range(
            max(min_train_groups, validation_groups),
            len(unique_groups) - validation_groups + 1,
            validation_groups,
        )
    )
    if not split_starts:
        validation_groups = max(1, int(math.ceil(len(unique_groups) * 0.2)))
        start = len(unique_groups) - validation_groups
        if start <= 0:
            return []
        split_starts = [start]

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    group_values = group_ids.to_numpy(dtype=object)
    for start in split_starts:
        train_groups = set(unique_groups[:start])
        valid_groups = set(unique_groups[start : start + validation_groups])
        if not train_groups or not valid_groups:
            continue
        train_idx = np.flatnonzero(np.isin(group_values, list(train_groups)))
        valid_idx = np.flatnonzero(np.isin(group_values, list(valid_groups)))
        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue
        splits.append((train_idx, valid_idx))

    return splits


def recency_weights(frame: pd.DataFrame, *, half_life_days: float = 90.0) -> np.ndarray:
    """Return exponentially decayed sample weights favoring newer observations."""

    if "target_date" not in frame.columns or frame.empty:
        return np.ones(len(frame), dtype=np.float32)
    target_dates = pd.to_datetime(frame["target_date"], errors="coerce")
    newest = target_dates.max()
    if pd.isna(newest):
        return np.ones(len(frame), dtype=np.float32)
    age_days = (newest - target_dates).dt.days.fillna(0).clip(lower=0).to_numpy(dtype=float)
    decay = np.exp(-np.log(2.0) * age_days / max(half_life_days, 1.0))
    return np.clip(decay, 0.2, 1.0).astype(np.float32)


def _parse_feature_availability(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    default_payloads: list[dict[str, bool]]
    if "feature_availability_json" not in frame.columns:
        default_payloads = [{} for _ in range(len(frame))]
    else:
        default_payloads = []
        for raw in frame["feature_availability_json"]:
            if not isinstance(raw, str):
                default_payloads.append({})
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
            default_payloads.append(payload if isinstance(payload, dict) else {})

    data: dict[str, list[float]] = {}
    for feature in feature_names:
        column_values: list[float] = []
        frame_has_feature = feature in frame.columns
        for row_idx, payload in enumerate(default_payloads):
            fallback = bool(frame_has_feature and pd.notna(frame.iloc[row_idx][feature]))
            column_values.append(float(bool(payload.get(feature, fallback))))
        data[feature] = column_values
    return pd.DataFrame(data, index=frame.index, dtype=float)


def _extract_cities(frame: pd.DataFrame) -> pd.Series:
    if "city" in frame.columns:
        return frame["city"].astype(str).fillna("unknown")
    if "market_spec_json" not in frame.columns:
        return pd.Series(["unknown"] * len(frame), index=frame.index, dtype=object)

    cache: dict[str, str] = {}
    values: list[str] = []
    for raw in frame["market_spec_json"]:
        key = str(raw)
        if key not in cache:
            city = "unknown"
            if isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                    if isinstance(payload, dict):
                        city = str(payload.get("city", "unknown"))
                except json.JSONDecodeError:
                    city = "unknown"
            cache[key] = city
        values.append(cache[key])
    return pd.Series(values, index=frame.index, dtype=object)


@dataclass
class ContextualFeatureBuilder:
    """Build a leakage-safe contextual design matrix from raw tabular rows."""

    base_feature_names: list[str]
    city_categories: list[str] = field(default_factory=list)
    horizon_categories: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)
    output_columns: list[str] = field(default_factory=list)
    model_daily_max_features: list[str] = field(default_factory=list)
    # Climate normals: (city, month, day_of_month) → (mean_°C, std_°C)
    # Computed from training-set realized_daily_max; safe to use at inference time.
    clim_normals: dict[tuple[str, int, int], tuple[float, float]] = field(default_factory=dict)

    def _compute_clim_normals(self, frame: pd.DataFrame) -> None:
        """Compute per-(city, month, day) climatological mean and std from truth labels."""
        if "realized_daily_max" not in frame.columns or "target_date" not in frame.columns:
            return
        cities = _extract_cities(frame)
        dates = pd.to_datetime(frame["target_date"], errors="coerce")
        truth = pd.to_numeric(frame["realized_daily_max"], errors="coerce")
        tmp = pd.DataFrame({
            "city": cities.values,
            "month": dates.dt.month.values,
            "day": dates.dt.day.values,
            "truth": truth.values,
        }).dropna(subset=["truth"])
        grouped = tmp.groupby(["city", "month", "day"])["truth"]
        means = grouped.mean()
        stds = grouped.std().fillna(1.0).clip(lower=0.5)
        self.clim_normals = {
            (str(idx[0]), int(idx[1]), int(idx[2])): (float(means[idx]), float(stds[idx]))
            for idx in means.index
        }

    def fit(self, frame: pd.DataFrame) -> ContextualFeatureBuilder:
        ordered = _ordered_frame(frame)
        self.base_feature_names = list(dict.fromkeys(self.base_feature_names))
        self.model_daily_max_features = [
            feature for feature in self.base_feature_names if feature.endswith("_model_daily_max")
        ]
        self.city_categories = sorted(
            {
                str(value)
                for value in _extract_cities(ordered).fillna("unknown").tolist()
                if str(value).strip()
            }
        )
        self.horizon_categories = sorted(
            {
                str(value)
                for value in ordered.get(
                    "decision_horizon",
                    pd.Series(["unknown"] * len(ordered), index=ordered.index, dtype=object),
                )
                .astype(str)
                .tolist()
                if str(value).strip()
            }
        )
        medians: dict[str, float] = {}
        for feature in self.base_feature_names:
            if feature in ordered.columns:
                numeric = pd.to_numeric(ordered[feature], errors="coerce")
                median = numeric.median()
                medians[feature] = float(median) if pd.notna(median) else 0.0
            else:
                medians[feature] = 0.0
        self.medians = medians
        self._compute_clim_normals(ordered)
        transformed = self._build_frame(ordered)
        self.output_columns = list(transformed.columns)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = self._build_frame(frame.reset_index(drop=True).copy())
        if not self.output_columns:
            self.output_columns = list(transformed.columns)
            return transformed
        for column in self.output_columns:
            if column not in transformed.columns:
                transformed[column] = 0.0
        return transformed[self.output_columns].copy()

    def _build_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        availability = _parse_feature_availability(frame, self.base_feature_names)
        cities = _extract_cities(frame).fillna("unknown").astype(str)
        horizons = frame.get(
            "decision_horizon",
            pd.Series(["unknown"] * len(frame), index=frame.index, dtype=object),
        ).fillna("unknown").astype(str)

        data: dict[str, pd.Series | np.ndarray] = {}
        for feature in self.base_feature_names:
            if feature in frame.columns:
                numeric = pd.to_numeric(frame[feature], errors="coerce")
            else:
                numeric = pd.Series(np.nan, index=frame.index, dtype=float)
            filled = numeric.fillna(self.medians.get(feature, 0.0)).astype(float)
            data[feature] = filled
            data[f"miss__{feature}"] = numeric.isna().astype(float)
            data[f"avail__{feature}"] = availability[feature].astype(float)

        target_date_values = frame.get(
            "target_date",
            pd.Series([pd.NaT] * len(frame), index=frame.index, dtype="datetime64[ns]"),
        )
        target_dates = pd.to_datetime(target_date_values, errors="coerce")
        day_of_year = target_dates.dt.dayofyear.fillna(1.0).to_numpy(dtype=float)
        data["day_of_year_sin"] = np.sin(2.0 * np.pi * day_of_year / 366.0)
        data["day_of_year_cos"] = np.cos(2.0 * np.pi * day_of_year / 366.0)

        if "lead_hours" in data:
            lead_hours = np.asarray(data["lead_hours"], dtype=float)
        elif "lead_hours" in frame.columns:
            lead_hours = pd.to_numeric(frame["lead_hours"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            lead_hours = np.zeros(len(frame), dtype=float)
            data["lead_hours"] = lead_hours
            data["miss__lead_hours"] = np.ones(len(frame), dtype=float)
            data["avail__lead_hours"] = np.zeros(len(frame), dtype=float)
        data["lead_hours_sq"] = np.square(lead_hours)

        availability_array = availability.to_numpy(dtype=float) if not availability.empty else np.zeros((len(frame), 0))
        if availability_array.shape[1] > 0:
            data["available_feature_fraction"] = availability_array.mean(axis=1)
        else:
            data["available_feature_fraction"] = np.zeros(len(frame), dtype=float)

        model_availability_columns = [feature for feature in self.model_daily_max_features if feature in availability.columns]
        if model_availability_columns:
            data["available_model_feature_count"] = availability[model_availability_columns].sum(axis=1).to_numpy(dtype=float)
        else:
            data["available_model_feature_count"] = np.zeros(len(frame), dtype=float)

        for category in self.city_categories:
            data[f"city__{_safe_token(category)}"] = (cities == category).astype(float)
        for category in self.horizon_categories:
            data[f"horizon__{_safe_token(category)}"] = (horizons == category).astype(float)

        # Weather interaction features derived from the primary NWP model.
        # These encode known meteorological relationships as explicit features so
        # the tree learner doesn't need deep splits to discover them.
        _p = "ecmwf_ifs025"
        _cloud = f"{_p}_cloud_cover_mean"
        _diurnal = f"{_p}_diurnal_amplitude"
        _dew = f"{_p}_dew_point_mean"
        _tmax = f"{_p}_model_daily_max"
        _wind = f"{_p}_wind_speed_mean"
        _humid = f"{_p}_humidity_mean"
        if _cloud in data and _diurnal in data:
            # Cloud suppresses diurnal swing; high cloud → small swing
            data["wx_cloud_diurnal"] = np.asarray(data[_cloud], dtype=float) * np.asarray(data[_diurnal], dtype=float)
        if _tmax in data and _dew in data:
            # Dew point depression: distance from saturation; higher → drier → more extreme heat possible
            data["wx_dew_depression"] = np.asarray(data[_tmax], dtype=float) - np.asarray(data[_dew], dtype=float)
        if _tmax in data and _humid in data:
            # Heat-index proxy: high temp × high humidity → feels hotter, harder to reach target max
            data["wx_heat_index_proxy"] = np.asarray(data[_tmax], dtype=float) * np.asarray(data[_humid], dtype=float) / 100.0
        if _wind in data and _diurnal in data:
            # Wind disperses heat; high wind suppresses diurnal max
            data["wx_wind_diurnal"] = np.asarray(data[_wind], dtype=float) * np.asarray(data[_diurnal], dtype=float)

        # Climate anomaly features: (forecast - clim_mean) / clim_std
        # Uses per-(city, month, day) normals fit on training truth labels.
        # At inference time, the stored normals are applied to the NWP forecast.
        # getattr guard: old pickles (pre-clim_normals) won't have this attribute.
        clim_normals = getattr(self, "clim_normals", {})
        if clim_normals:
            target_dates = pd.to_datetime(
                frame.get("target_date", pd.Series([pd.NaT] * len(frame), index=frame.index)),
                errors="coerce",
            )
            months = target_dates.dt.month.fillna(1).astype(int)
            days = target_dates.dt.day.fillna(1).astype(int)

            # Use ensemble mean of available model forecasts as the "forecast" for anomaly
            max_feat_cols = [f for f in self.model_daily_max_features if f in frame.columns]
            if max_feat_cols:
                forecast_vals = (
                    frame[max_feat_cols]
                    .apply(pd.to_numeric, errors="coerce")
                    .mean(axis=1)
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
            else:
                forecast_vals = np.zeros(len(frame), dtype=float)

            clim_mean_arr = np.zeros(len(frame), dtype=float)
            clim_std_arr = np.ones(len(frame), dtype=float)
            for i, (city, m, d) in enumerate(zip(cities, months, days, strict=False)):
                key = (str(city), int(m), int(d))
                if key in clim_normals:
                    clim_mean_arr[i], clim_std_arr[i] = clim_normals[key]

            anomaly = forecast_vals - clim_mean_arr
            # Only expose z-score: how many std-devs is this forecast from seasonal normal?
            # Raw anomaly and clim_mean cause regression-to-mean bias on anomalous weather days.
            data["clim_forecast_anomaly_zscore"] = anomaly / np.where(clim_std_arr > 0, clim_std_arr, 1.0)

        return pd.DataFrame(data, index=frame.index, dtype=float)
