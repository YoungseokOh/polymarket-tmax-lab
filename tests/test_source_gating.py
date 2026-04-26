from __future__ import annotations

import numpy as np
import pandas as pd

from pmtmax.markets.market_spec import FinalizationPolicy, MarketSpec, OutcomeBin, PrecisionRule
from pmtmax.modeling.source_gating import (
    ConstantFallbackGate,
    SourceGatedGaussianModel,
    SourceGatingConfig,
    fit_source_gated_model,
)


class _FakeGaussianModel:
    feature_names: list[str] = []

    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return np.full(len(frame), self.mean), np.full(len(frame), self.std)


class _ColumnGaussianModel:
    feature_names: list[str] = []

    def __init__(self, mean_column: str, std: float = 1.0) -> None:
        self.mean_column = mean_column
        self.std = std

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return frame[self.mean_column].to_numpy(dtype=float), np.full(len(frame), self.std)


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


def test_source_gated_model_moment_blends_in_celsius_space_for_fahrenheit_market() -> None:
    frame = pd.DataFrame(
        [
            {
                "market_spec_json": _spec("F").model_dump_json(),
                "city": "Atlanta",
                "target_date": "2026-02-18",
                "decision_horizon": "market_open",
                "lead_hours": 6.0,
            }
        ]
    )
    model = SourceGatedGaussianModel(
        primary_model=_FakeGaussianModel(mean=50.0, std=3.6),  # 10C, 2C scale
        fallback_model=_FakeGaussianModel(mean=68.0, std=9.0),  # 20C, 5C scale
        gate_model=ConstantFallbackGate(0.5),
        config=SourceGatingConfig(name="test"),
    )

    mean, std = model.predict(frame)

    assert np.allclose(mean, np.asarray([59.0]))
    assert np.allclose(std, np.asarray([np.sqrt(39.5) * 1.8]))


def test_fit_source_gated_model_uses_constant_gate_when_fallback_always_better() -> None:
    frame = pd.DataFrame(
        {
            "market_spec_json": [_spec("C").model_dump_json()] * 24,
            "city": ["Atlanta"] * 24,
            "target_date": pd.date_range("2026-01-01", periods=24, freq="D"),
            "decision_horizon": ["market_open"] * 24,
            "lead_hours": np.linspace(1.0, 24.0, 24),
            "realized_daily_max": np.linspace(10.0, 33.0, 24),
        }
    )
    frame["primary_pred"] = frame["realized_daily_max"] - 8.0
    frame["fallback_pred"] = frame["realized_daily_max"]

    model, diagnostics = fit_source_gated_model(
        frame,
        primary_model=_ColumnGaussianModel("primary_pred"),
        fallback_model=_ColumnGaussianModel("fallback_pred"),
        config=SourceGatingConfig(name="test"),
    )

    weights = model.predict_fallback_weight(frame)

    assert diagnostics["fallback_better_rate"] == 1.0
    assert np.allclose(weights, np.ones(len(frame)))
