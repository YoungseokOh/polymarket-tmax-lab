"""Shared contextual design-matrix helpers for tabular forecasting models."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _city_latitude_map() -> dict[str, float]:
    """Load city → latitude mapping from station catalog."""
    catalog_path = Path(__file__).resolve().parents[4] / "configs" / "market_inventory" / "station_catalog.json"
    try:
        catalog = json.loads(catalog_path.read_text())
        return {str(info.get("city") or key): float(info["lat"]) for key, info in catalog.items()}
    except Exception:
        return {}


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
    # Fast path: no JSON column — use column notna as availability proxy.
    if "feature_availability_json" not in frame.columns:
        data = {}
        for feature in feature_names:
            data[feature] = frame[feature].notna() if feature in frame.columns else pd.Series(False, index=frame.index)
        return pd.DataFrame(data, index=frame.index, dtype=float)

    # Parse JSON column in one vectorised pass (avoids per-row iloc which is O(n²)).
    def _safe_parse(x: object) -> dict:
        if not isinstance(x, str):
            return {}
        try:
            payload = json.loads(x)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}

    parsed_series = frame["feature_availability_json"].map(_safe_parse)
    # Build wide DataFrame from list of dicts — one allocation instead of N×F appends.
    json_df = pd.DataFrame(list(parsed_series), index=frame.index)

    data = {}
    for feature in feature_names:
        # Vectorised notna fallback — replaces the slow frame.iloc[row][feature] loop.
        fallback = frame[feature].notna() if feature in frame.columns else pd.Series(False, index=frame.index)
        if feature in json_df.columns:
            json_col = pd.to_numeric(json_df[feature], errors="coerce")
            # Use JSON value where present, else fall back to notna.
            merged = json_col.where(json_col.notna(), fallback.astype(float))
            data[feature] = merged.astype(float)
        else:
            data[feature] = fallback.astype(float)
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
    use_city_lat: bool = False  # True → add continuous city_latitude feature (lat/90, signed for hemisphere)
    use_city_month: bool = False  # True → add city×month interaction dummies (city×12 months)
    use_clim_anomaly: bool = False  # True → add clim_delta / clim_zscore (NWP forecast vs city×month normal)
    use_forecast_bias: bool = False  # True → add forecast_bias / bias_corrected_nwp (city×month systematic error)
    use_bin_position: bool = False  # True → add bin_rank / n_bins / tail / boundary distance features
    use_bin_boundary_dist: bool = False  # True → add only bin_boundary_dist (min dist to nearest bin edge)
    # Fitted climate normals: {(city, month) → (mean, std)} — populated by fit(), used in transform()
    clim_normals: dict = field(default_factory=dict)
    # Fitted forecast bias: {(city, month) → mean_bias} — populated by fit(), used in transform()
    forecast_bias_normals: dict = field(default_factory=dict)

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

        # Climate normals: city×month mean and std of realized_daily_max
        if self.use_clim_anomaly and "realized_daily_max" in ordered.columns and "target_date" in ordered.columns:
            cities_for_clim = _extract_cities(ordered).fillna("unknown").astype(str)
            months = pd.to_datetime(ordered["target_date"], errors="coerce").dt.month.fillna(1).astype(int)
            obs = pd.to_numeric(ordered["realized_daily_max"], errors="coerce")
            clim_df = pd.DataFrame({"city": cities_for_clim, "month": months, "obs": obs}).dropna()
            grouped = clim_df.groupby(["city", "month"])["obs"]
            self.clim_normals = {
                (city, month): (float(grp.mean()), max(float(grp.std(ddof=1)), 0.5))
                for (city, month), grp in grouped
                if len(grp) >= 5
            }

        # Forecast bias: city×month mean of (realized_daily_max - ecmwf_ifs025_model_daily_max)
        # Captures systematic model cold/warm bias to improve bin prediction accuracy.
        if (
            self.use_forecast_bias
            and "realized_daily_max" in ordered.columns
            and "target_date" in ordered.columns
            and "ecmwf_ifs025_model_daily_max" in ordered.columns
        ):
            cities_for_bias = _extract_cities(ordered).fillna("unknown").astype(str)
            months_for_bias = pd.to_datetime(ordered["target_date"], errors="coerce").dt.month.fillna(1).astype(int)
            realized = pd.to_numeric(ordered["realized_daily_max"], errors="coerce")
            forecast = pd.to_numeric(ordered["ecmwf_ifs025_model_daily_max"], errors="coerce")
            error = realized - forecast
            bias_df = pd.DataFrame({"city": cities_for_bias, "month": months_for_bias, "error": error}).dropna()
            grouped_bias = bias_df.groupby(["city", "month"])["error"]
            self.forecast_bias_normals = {
                (city, month): float(grp.mean())
                for (city, month), grp in grouped_bias
                if len(grp) >= 5
            }

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

        if self.use_city_lat:
            lat_map = _city_latitude_map()
            # Normalise to [-1, 1]; sign encodes hemisphere (+ = N, - = S)
            data["city_latitude"] = cities.map(lambda c: lat_map.get(c, 0.0) / 90.0).astype(float)

        if self.use_city_month:
            months = target_dates.dt.month.fillna(1).astype(int)
            for city in self.city_categories:
                city_tok = _safe_token(city)
                city_mask = (cities == city).to_numpy(dtype=bool)
                for month in range(1, 13):
                    data[f"cm__{city_tok}__m{month:02d}"] = (city_mask & (months == month).to_numpy()).astype(float)

        # Climate anomaly features: how unusual is the NWP forecast relative to historical normal?
        # clim_delta   = NWP forecast daily max - climatological mean for this city/month
        # clim_zscore  = clim_delta / climatological std (normalized anomaly)
        # Large |clim_zscore| → model sees something unusual → high uncertainty → be cautious
        if self.use_clim_anomaly and self.clim_normals:
            nwp_max = np.asarray(data.get("ecmwf_ifs025_model_daily_max", np.zeros(len(frame))), dtype=float)
            target_dates_for_clim = pd.to_datetime(
                frame.get("target_date", pd.Series([pd.NaT] * len(frame), index=frame.index)),
                errors="coerce",
            )
            months_for_clim = target_dates_for_clim.dt.month.fillna(1).astype(int).to_numpy()
            cities_for_clim = _extract_cities(frame).fillna("unknown").astype(str).to_numpy()
            clim_delta = np.zeros(len(frame), dtype=float)
            clim_zscore = np.zeros(len(frame), dtype=float)
            for i in range(len(frame)):
                key = (cities_for_clim[i], int(months_for_clim[i]))
                if key in self.clim_normals:
                    mean, std = self.clim_normals[key]
                    clim_delta[i] = nwp_max[i] - mean
                    clim_zscore[i] = clim_delta[i] / std
            data["clim_delta"] = clim_delta
            data["clim_zscore"] = clim_zscore

        # Forecast bias features: systematic city×month model error correction.
        # forecast_bias      = historical mean(realized - ecmwf) for this city/month
        # bias_corrected_nwp = ecmwf forecast + forecast_bias (bias-corrected point estimate)
        if self.use_forecast_bias and self.forecast_bias_normals:
            nwp_for_bias = np.asarray(data.get("ecmwf_ifs025_model_daily_max", np.zeros(len(frame))), dtype=float)
            target_dates_for_bias = pd.to_datetime(
                frame.get("target_date", pd.Series([pd.NaT] * len(frame), index=frame.index)),
                errors="coerce",
            )
            months_for_bias = target_dates_for_bias.dt.month.fillna(1).astype(int).to_numpy()
            cities_for_bias = _extract_cities(frame).fillna("unknown").astype(str).to_numpy()
            forecast_bias = np.zeros(len(frame), dtype=float)
            for i in range(len(frame)):
                key = (cities_for_bias[i], int(months_for_bias[i]))
                if key in self.forecast_bias_normals:
                    forecast_bias[i] = self.forecast_bias_normals[key]
            data["forecast_bias"] = forecast_bias
            data["bias_corrected_nwp"] = nwp_for_bias + forecast_bias

        # Bin position features: encode where the NWP forecast sits within the outcome schema.
        # bin_rank       = 0-indexed bin the forecast falls in (0 = lowest bin)
        # bin_rank_norm  = bin_rank / (n_bins - 1), normalised to [0, 1]
        # n_bins         = total number of outcome bins for this market
        # is_lower_tail_bin = 1 if forecast is in the open-lower boundary bin
        # is_upper_tail_bin = 1 if forecast is in the open-upper boundary bin
        # bin_boundary_dist = min distance to either bin boundary (in degrees); 0 = right on the edge
        if self.use_bin_position and "market_spec_json" in frame.columns:
            nwp_vals = np.asarray(data.get("ecmwf_ifs025_model_daily_max", np.zeros(len(frame))), dtype=float)
            bin_rank_arr = np.zeros(len(frame), dtype=float)
            bin_rank_norm_arr = np.zeros(len(frame), dtype=float)
            n_bins_arr = np.zeros(len(frame), dtype=float)
            is_lower_tail_arr = np.zeros(len(frame), dtype=float)
            is_upper_tail_arr = np.zeros(len(frame), dtype=float)
            bin_boundary_dist_arr = np.zeros(len(frame), dtype=float)
            _schema_cache: dict[str, list] = {}
            spec_jsons = frame["market_spec_json"].tolist()
            for i, (spec_raw, x) in enumerate(zip(spec_jsons, nwp_vals, strict=True)):
                if spec_raw not in _schema_cache:
                    try:
                        spec_dict = json.loads(spec_raw) if isinstance(spec_raw, str) else spec_raw
                        _schema_cache[spec_raw] = spec_dict.get("outcome_schema", [])
                    except Exception:
                        _schema_cache[spec_raw] = []
                schema = _schema_cache[spec_raw]
                n = len(schema)
                if n == 0:
                    continue
                n_bins_arr[i] = float(n)
                matched = -1
                for j, bin_spec in enumerate(schema):
                    lo = bin_spec.get("lower")
                    up = bin_spec.get("upper")
                    lo_f = -np.inf if lo is None else float(lo)
                    up_f = np.inf if up is None else float(up)
                    # For point bins (lower == upper) use a ±0.5 window
                    if lo is not None and up is not None and lo_f == up_f:
                        lo_f -= 0.5
                        up_f += 0.5
                    if lo_f <= x <= up_f:
                        matched = j
                        break
                if matched < 0:
                    # Assign to closest bin by midpoint
                    best, best_dist = 0, np.inf
                    for j, bin_spec in enumerate(schema):
                        lo = bin_spec.get("lower")
                        up = bin_spec.get("upper")
                        lo_f = 0.0 if lo is None else float(lo)
                        up_f = lo_f if up is None else float(up)
                        mid = (lo_f + up_f) / 2.0
                        d = abs(x - mid)
                        if d < best_dist:
                            best, best_dist = j, d
                    matched = best
                bin_rank_arr[i] = float(matched)
                bin_rank_norm_arr[i] = float(matched) / max(n - 1, 1)
                is_lower_tail_arr[i] = 1.0 if matched == 0 else 0.0
                is_upper_tail_arr[i] = 1.0 if matched == n - 1 else 0.0
                # Distance to nearest boundary
                bin_spec = schema[matched]
                lo = bin_spec.get("lower")
                up = bin_spec.get("upper")
                d_lo = abs(x - float(lo)) if lo is not None else 999.0
                d_up = abs(float(up) - x) if up is not None else 999.0
                bin_boundary_dist_arr[i] = min(d_lo, d_up)
            data["bin_rank"] = bin_rank_arr
            data["bin_rank_norm"] = bin_rank_norm_arr
            data["n_bins"] = n_bins_arr
            data["is_lower_tail_bin"] = is_lower_tail_arr
            data["is_upper_tail_bin"] = is_upper_tail_arr
            data["bin_boundary_dist"] = bin_boundary_dist_arr

        # Standalone boundary-distance feature: min(dist_to_lower_bound, dist_to_upper_bound).
        # Skipped if use_bin_position already computed it above.
        if self.use_bin_boundary_dist and not self.use_bin_position and "market_spec_json" in frame.columns:
            nwp_vals = np.asarray(data.get("ecmwf_ifs025_model_daily_max", np.zeros(len(frame))), dtype=float)
            bin_boundary_dist_arr = np.zeros(len(frame), dtype=float)
            _schema_cache2: dict[str, list] = {}
            for i, (spec_raw, x) in enumerate(
                zip(frame["market_spec_json"].tolist(), nwp_vals, strict=True)
            ):
                if spec_raw not in _schema_cache2:
                    try:
                        spec_dict = json.loads(spec_raw) if isinstance(spec_raw, str) else spec_raw
                        _schema_cache2[spec_raw] = spec_dict.get("outcome_schema", [])
                    except Exception:
                        _schema_cache2[spec_raw] = []
                schema = _schema_cache2[spec_raw]
                if not schema:
                    continue
                # Find best matching bin (same logic as use_bin_position)
                matched = 0
                best_dist = float("inf")
                for j, b in enumerate(schema):
                    lo = b.get("lower")
                    up = b.get("upper")
                    if lo is None and up is None:
                        continue
                    if lo is not None and up is not None and float(lo) == float(up):
                        lo_f = float(lo) - 0.5
                        up_f = float(up) + 0.5
                    else:
                        lo_f = float(lo) if lo is not None else -999.0
                        up_f = float(up) if up is not None else 999.0
                    if lo_f <= x < up_f:
                        matched = j
                        best_dist = 0.0
                        break
                    mid = (lo_f + up_f) / 2.0
                    d = abs(x - mid)
                    if d < best_dist:
                        matched = j
                        best_dist = d
                bin_spec = schema[matched]
                lo = bin_spec.get("lower")
                up = bin_spec.get("upper")
                d_lo = abs(x - float(lo)) if lo is not None else 999.0
                d_up = abs(float(up) - x) if up is not None else 999.0
                bin_boundary_dist_arr[i] = min(d_lo, d_up)
            data["bin_boundary_dist"] = bin_boundary_dist_arr

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

        return pd.DataFrame(data, index=frame.index, dtype=float)
