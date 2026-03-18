from pathlib import Path

import pandas as pd

from pmtmax.examples import example_market_specs
from pmtmax.modeling.predict import predict_market
from pmtmax.modeling.train import train_model


def test_train_and_predict_end_to_end(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "market_id": "m1",
                "station_id": "RKSI",
                "target_date": pd.Timestamp("2025-12-11"),
                "model_daily_max": 8.0,
                "ecmwf_ifs025_model_daily_max": 8.0,
                "ecmwf_aifs025_single_model_daily_max": 7.5,
                "lead_hours": 24.0,
                "neighbor_mean_temp": 8.0,
                "neighbor_spread": 0.5,
                "realized_daily_max": 8.0,
            },
            {
                "market_id": "m2",
                "station_id": "KLGA",
                "target_date": pd.Timestamp("2025-12-20"),
                "model_daily_max": 39.0,
                "ecmwf_ifs025_model_daily_max": 39.0,
                "ecmwf_aifs025_single_model_daily_max": 38.5,
                "lead_hours": 24.0,
                "neighbor_mean_temp": 39.0,
                "neighbor_spread": 0.5,
                "realized_daily_max": 40.0,
            },
        ]
    )
    artifacts = tmp_path / "artifacts"
    artifact = train_model("gaussian_emos", frame, artifacts)
    spec = example_market_specs(["Seoul"])[0]
    forecast = predict_market(Path(artifact.path), "gaussian_emos", spec, frame.iloc[[0]])
    assert forecast.mean > 0
    assert abs(sum(forecast.outcome_probabilities.values()) - 1.0) < 1e-6

