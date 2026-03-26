from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmtmax.examples import example_market_specs
from pmtmax.modeling.predict import load_model, predict_market
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


def test_train_model_rejects_removed_models(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["1", "2"],
            "station_id": ["AAA", "AAA"],
            "target_date": [pd.Timestamp("2026-03-24"), pd.Timestamp("2026-03-25")],
            "realized_daily_max": [10.0, 12.0],
            "winning_outcome": ["9°C", "10°C"],
            "market_spec_json": ["{}", "{}"],
            "model_daily_max": [10.5, 11.5],
        }
    )

    with pytest.raises(ValueError, match="Unsupported model: flexible_flow_nn"):
        train_model("flexible_flow_nn", frame, tmp_path)


def test_train_model_persists_internal_variant_metadata(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "market_id": [f"m{i:03d}" for i in range(48)],
            "station_id": ["AAA"] * 48,
            "target_date": [pd.Timestamp("2026-01-01") + pd.Timedelta(days=i) for i in range(48)],
            "decision_horizon": ["morning_of"] * 48,
            "decision_time_utc": [pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=i) for i in range(48)],
            "realized_daily_max": np.linspace(8.0, 12.0, 48),
            "lead_hours": np.linspace(6.0, 18.0, 48),
            "model_daily_max": np.linspace(8.2, 11.8, 48),
            "ecmwf_ifs025_model_daily_max": np.linspace(8.1, 11.9, 48),
        }
    )

    artifact = train_model("tuned_ensemble", frame, tmp_path, variant="legacy_fixed2")

    assert artifact.variant == "legacy_fixed2"
    assert artifact.status == "experimental"
    assert "legacy_fixed2" in Path(artifact.path).name


def test_train_model_v2_persists_calibrator_and_predict_uses_calibrated_probs(tmp_path: Path) -> None:
    spec_template = example_market_specs(["Seoul"])[0]
    rows: list[dict[str, object]] = []
    for idx in range(50):
        target_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=idx)
        realized = 8.0 if idx % 2 == 0 else 9.0
        market_id = f"m{idx:03d}"
        spec = spec_template.model_copy(
            update={
                "market_id": market_id,
                "target_local_date": target_date.date(),
            }
        )
        rows.append(
            {
                "market_id": market_id,
                "station_id": spec.station_id,
                "target_date": target_date,
                "decision_horizon": "morning_of" if idx % 3 else "previous_evening",
                "decision_time_utc": target_date.tz_localize("UTC"),
                "market_spec_json": spec.model_dump_json(),
                "realized_daily_max": realized,
                "winning_outcome": "8°C" if realized == 8.0 else "9°C",
                "lead_hours": float(12 + (idx % 4)),
                "model_daily_max": realized - 0.2,
                "ecmwf_ifs025_model_daily_max": realized - 0.1,
            }
        )
    frame = pd.DataFrame(rows)

    artifact = train_model(
        "gaussian_emos",
        frame,
        tmp_path,
        split_policy="market_day",
        seed=123,
    )

    forecast = predict_market(
        Path(artifact.path),
        "gaussian_emos",
        spec_template,
        frame.iloc[[0]],
    )

    assert artifact.contract_version == "v2"
    assert artifact.split_policy == "market_day"
    assert artifact.seed == 123
    assert artifact.calibration_path is not None
    assert Path(artifact.calibration_path).exists()
    assert artifact.metrics["calibration_fitted"] == 1.0
    assert forecast.contract_version == "v2"
    assert forecast.probability_source == "calibrated"
    assert forecast.outcome_probabilities_calibrated
    assert abs(sum(forecast.outcome_probabilities.values()) - 1.0) < 1e-6
    assert forecast.feature_availability["model_daily_max"] is True
