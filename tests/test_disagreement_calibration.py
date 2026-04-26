from __future__ import annotations

import numpy as np
import pandas as pd

from pmtmax.markets.market_spec import FinalizationPolicy, MarketSpec, OutcomeBin, PrecisionRule
from pmtmax.modeling.disagreement_calibration import (
    DisagreementCalibratedGaussianModel,
    DisagreementCalibrationConfig,
    source_range_celsius,
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


def test_source_range_celsius_is_unit_aware_for_fahrenheit_market() -> None:
    frame = pd.DataFrame(
        [
            {
                "market_spec_json": _spec("F").model_dump_json(),
                "ecmwf_ifs025_model_daily_max": 50.0,
                "ecmwf_aifs025_single_model_daily_max": 59.0,
                "kma_gdps_model_daily_max": np.nan,
                "gfs_seamless_model_daily_max": 68.0,
            }
        ]
    )

    assert np.allclose(source_range_celsius(frame), np.asarray([10.0]))


def test_positive_disagreement_blends_mean_and_inflates_std_in_celsius_space() -> None:
    frame = pd.DataFrame(
        [
            {
                "market_spec_json": _spec("F").model_dump_json(),
                "ecmwf_ifs025_model_daily_max": 50.0,
                "ecmwf_aifs025_single_model_daily_max": 68.0,
            }
        ]
    )
    model = DisagreementCalibratedGaussianModel(
        primary_model=_FakeGaussianModel(mean=68.0, std=3.6),  # 20C, 2C scale
        fallback_model=_FakeGaussianModel(mean=50.0, std=5.4),  # 10C, 3C scale
        config=DisagreementCalibrationConfig(
            name="test",
            disagreement_variance_weight=0.25,
            mean_blend_mode="positive",
            mean_blend_weight_per_c=0.1,
            max_mean_blend_weight=0.5,
        ),
    )

    mean, std = model.predict(frame)

    assert np.allclose(mean, np.asarray([59.0]))  # 15C
    expected_std_c = np.sqrt(2.0**2 + 0.25 * 10.0**2 + 0.5 * 0.5 * 10.0**2)
    assert np.allclose(std, np.asarray([expected_std_c * 1.8]))
