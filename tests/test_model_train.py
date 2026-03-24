from __future__ import annotations

from pathlib import Path

import pandas as pd

from pmtmax.modeling.predict import load_model
from pmtmax.modeling.train import default_feature_names, train_model


def test_default_feature_names_excludes_metadata_and_constant_columns() -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["1", "2"],
            "station_id": ["AAA", "AAA"],
            "target_date": [pd.Timestamp("2026-03-24"), pd.Timestamp("2026-03-25")],
            "realized_daily_max": [12.0, 14.0],
            "winning_outcome": ["11-12", "13-14"],
            "source_priority": [80.0, 90.0],
            "settlement_eligible": [0.0, 1.0],
            "lead_hours": [6.0, 18.0],
            "model_daily_max": [11.0, 13.0],
            "ecmwf_ifs025_cloud_cover_mean": [0.0, 0.0],
            "ecmwf_ifs025_humidity_mean": [0.0, 0.0],
            "ecmwf_ifs025_midday_temp": [10.5, 12.5],
            "ecmwf_ifs025_num_hours": [24.0, 24.0],
        }
    )

    features = default_feature_names(frame)

    assert "lead_hours" in features
    assert "model_daily_max" in features
    assert "ecmwf_ifs025_midday_temp" in features
    assert "source_priority" not in features
    assert "settlement_eligible" not in features
    assert "ecmwf_ifs025_cloud_cover_mean" not in features
    assert "ecmwf_ifs025_humidity_mean" not in features
    assert "ecmwf_ifs025_num_hours" not in features


def test_gaussian_emos_trains_with_empty_feature_set(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["1", "2", "3"],
            "station_id": ["AAA", "AAA", "AAA"],
            "target_date": [
                pd.Timestamp("2026-03-24"),
                pd.Timestamp("2026-03-25"),
                pd.Timestamp("2026-03-26"),
            ],
            "realized_daily_max": [10.0, 14.0, 16.0],
            "winning_outcome": ["9-10", "13-14", "15-16"],
            "source_priority": [80.0, 80.0, 80.0],
            "settlement_eligible": [1.0, 1.0, 1.0],
            "ecmwf_ifs025_cloud_cover_mean": [0.0, 0.0, 0.0],
            "model_daily_max": [12.0, 12.0, 12.0],
        }
    )

    artifact = train_model("gaussian_emos", frame, tmp_path)
    model = load_model(Path(artifact.path))
    prediction_frame = frame.iloc[[0]].copy()
    mean, std = model.predict(prediction_frame)
    expected_mean = float(frame["realized_daily_max"].mean())
    expected_std = float(max((frame["realized_daily_max"] - expected_mean).abs().mean(), 0.5))

    assert artifact.features == []
    assert mean.tolist() == [expected_mean]
    assert std.tolist() == [expected_std]
