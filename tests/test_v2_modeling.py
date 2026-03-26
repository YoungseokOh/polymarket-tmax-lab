from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pmtmax.cli.main import _forecast_contract_rejection_reason
from pmtmax.examples import example_market_specs
from pmtmax.modeling.predict import predict_market
from pmtmax.storage.schemas import ProbForecast


class _MixtureModel:
    feature_names = ["model_daily_max"]

    def predict(
        self,
        frame: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert "model_daily_max" in frame.columns
        return (
            np.array([[0.5, 0.5]], dtype=float),
            np.array([[8.0, 10.0]], dtype=float),
            np.array([[0.05, 0.05]], dtype=float),
        )


def test_predict_market_preserves_gaussian_mixture_probabilities(monkeypatch) -> None:
    spec = example_market_specs(["Seoul"])[0]
    frame = pd.DataFrame([{"model_daily_max": 9.0}])

    monkeypatch.setattr("pmtmax.modeling.predict.load_model", lambda path: _MixtureModel())

    forecast = predict_market(
        Path("unused.pkl"),
        "flexible_flow_nn",
        spec,
        frame,
        calibrate=False,
    )

    assert forecast.contract_version == "v2"
    assert forecast.distribution_family == "gaussian_mixture"
    assert forecast.daily_max_distribution["family"] == "gaussian_mixture"
    assert forecast.distribution_payload["weights"] == [0.5, 0.5]
    assert forecast.outcome_probabilities["9°C"] < 0.05
    assert forecast.outcome_probabilities["8°C"] > 0.3
    assert forecast.outcome_probabilities["10°C or higher"] > 0.3


def test_forecast_contract_rejection_reason_requires_calibrated_contract() -> None:
    forecast = ProbForecast(
        target_market="m1",
        generated_at=datetime.now(tz=UTC),
        contract_version="v2",
        mean=9.0,
        std=1.0,
        outcome_probabilities={"8°C": 0.4, "9°C": 0.6},
        probability_source="raw",
    )

    assert _forecast_contract_rejection_reason(forecast) == "missing_calibrator"
    assert _forecast_contract_rejection_reason(forecast.model_copy(update={"probability_source": "calibrated"})) is None
