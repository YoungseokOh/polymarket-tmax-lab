from __future__ import annotations

import numpy as np
import pandas as pd

from pmtmax.markets.market_spec import FinalizationPolicy, MarketSpec, OutcomeBin, PrecisionRule
from pmtmax.modeling.tail_calibration import (
    TailCalibratedGaussianModel,
    TailCalibrationConfig,
    sentinel_daily_max_count,
)


class _FakeGaussianModel:
    feature_names: list[str] = []

    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return np.full(len(frame), self.mean), np.full(len(frame), self.std)


def _spec(unit: str) -> MarketSpec:
    return MarketSpec(
        market_id=f"m-{unit}",
        slug=f"m-{unit}",
        question="Highest temperature?",
        city="Atlanta",
        country="US",
        target_local_date="2026-02-18",
        timezone="America/New_York",
        official_source_name="Wunderground",
        official_source_url="https://example.test",
        station_id="KATL",
        station_name="Atlanta",
        unit=unit,  # type: ignore[arg-type]
        precision_rule=PrecisionRule(
            unit=unit,  # type: ignore[arg-type]
            step=1.0,
            source_precision_text="whole degrees",
        ),
        outcome_schema=[OutcomeBin(label="Any", lower=None, upper=None)],
        finalization_policy=FinalizationPolicy(),
    )


def test_sentinel_daily_max_count_is_unit_aware() -> None:
    frame = pd.DataFrame(
        [
            {
                "market_spec_json": _spec("F").model_dump_json(),
                "ecmwf_ifs025_model_daily_max": 32.0,
                "ecmwf_aifs025_single_model_daily_max": 64.0,
                "kma_gdps_model_daily_max": 32.0,
                "gfs_seamless_model_daily_max": 65.0,
            },
            {
                "market_spec_json": _spec("C").model_dump_json(),
                "ecmwf_ifs025_model_daily_max": 0.0,
                "ecmwf_aifs025_single_model_daily_max": 18.0,
                "kma_gdps_model_daily_max": 0.0,
                "gfs_seamless_model_daily_max": 21.0,
            },
        ]
    )

    assert sentinel_daily_max_count(frame).tolist() == [2, 2]


def test_tail_calibrated_model_blends_fallback_in_celsius_space_for_fahrenheit_market() -> None:
    frame = pd.DataFrame(
        [
            {
                "market_spec_json": _spec("F").model_dump_json(),
                "ecmwf_ifs025_model_daily_max": 32.0,
                "ecmwf_aifs025_single_model_daily_max": 70.0,
                "kma_gdps_model_daily_max": 32.0,
                "gfs_seamless_model_daily_max": 32.0,
            }
        ]
    )
    model = TailCalibratedGaussianModel(
        primary_model=_FakeGaussianModel(mean=50.0, std=3.6),  # 10C, 2C scale
        fallback_model=_FakeGaussianModel(mean=68.0, std=9.0),  # 20C, 5C scale
        config=TailCalibrationConfig(
            name="test",
            fallback_sentinel_min_count=3,
            fallback_mean_weight=0.5,
            fallback_std_mode="blend",
            scale_floor_c=1.0,
        ),
    )

    mean, std = model.predict(frame)

    assert np.allclose(mean, np.asarray([59.0]))  # 15C converted back to F
    assert np.allclose(std, np.asarray([6.3]))  # 3.5C converted back to F scale
