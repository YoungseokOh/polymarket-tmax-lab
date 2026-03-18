"""Feature engineering for temperature-max models."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd


def build_hourly_feature_frame(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize Open-Meteo hourly data into a DataFrame-backed feature package."""

    hourly = payload.get("hourly", {})
    frame = pd.DataFrame(hourly)
    if not frame.empty and "time" in frame:
        frame["time"] = pd.to_datetime(frame["time"])
        frame["hour"] = frame["time"].dt.hour
        frame["date"] = frame["time"].dt.date
    return {"raw": payload, "frame": frame}


def target_day_features(package: dict[str, Any], target_date: date) -> dict[str, float]:
    """Build direct-daily-max predictors from hourly target-day slices."""

    frame: pd.DataFrame = package["frame"]
    subset = frame.loc[frame["date"] == target_date].copy()
    if subset.empty:
        return {}
    temp = subset["temperature_2m"].astype(float)
    midday = subset.loc[subset["hour"].between(11, 15), "temperature_2m"].astype(float)
    return {
        "model_daily_max": float(temp.max()),
        "model_daily_mean": float(temp.mean()),
        "model_daily_min": float(temp.min()),
        "diurnal_amplitude": float(temp.max() - temp.min()),
        "midday_temp": float(midday.mean()) if not midday.empty else float(temp.mean()),
        "cloud_cover_mean": float(subset.get("cloud_cover", pd.Series([0.0])).mean()),
        "wind_speed_mean": float(subset.get("wind_speed_10m", pd.Series([0.0])).mean()),
        "humidity_mean": float(subset.get("relative_humidity_2m", pd.Series([0.0])).mean()),
        "dew_point_mean": float(subset.get("dew_point_2m", pd.Series([0.0])).mean()),
        "num_hours": float(len(subset)),
    }


def summarize_hourly_trajectory(package: dict[str, Any], target_date: date) -> np.ndarray:
    """Return hourly temperature trajectory for the target local date."""

    frame: pd.DataFrame = package["frame"]
    subset = frame.loc[frame["date"] == target_date].copy()
    if subset.empty:
        return np.array([], dtype=float)
    return np.asarray(subset["temperature_2m"].astype(float).to_numpy(), dtype=float)
