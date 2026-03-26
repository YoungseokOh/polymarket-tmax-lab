from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pmtmax.examples import example_market_specs
from pmtmax.modeling.design_matrix import ContextualFeatureBuilder
from pmtmax.modeling.predict import load_model, predict_market
from pmtmax.modeling.train import sanitize_model_frame, train_model


def _winning_outcome(spec, value: float) -> str:
    for outcome in spec.outcome_schema:
        if outcome.contains(value):
            return outcome.label
    raise AssertionError(f"no outcome for value={value}")


def _training_frame() -> pd.DataFrame:
    seoul, nyc = example_market_specs(["Seoul", "NYC"])
    rows: list[dict[str, object]] = []
    for idx in range(72):
        target_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=idx)
        spec = seoul if idx % 2 == 0 else nyc
        spec = spec.model_copy(
            update={
                "market_id": f"m{idx:03d}",
                "target_local_date": target_date.date(),
            }
        )
        base = 9.0 + (idx % 5) + (0.8 if spec.city == "NYC" else 0.0)
        winning = _winning_outcome(spec, base)
        humidity = np.nan if idx % 5 == 0 else 45.0 + idx % 7
        feature_availability = {
            "model_daily_max": True,
            "ecmwf_ifs025_model_daily_max": idx % 4 != 0,
            "gfs_seamless_model_daily_max": idx % 3 != 0,
            "humidity_mean": not np.isnan(humidity),
        }
        rows.append(
            {
                "market_id": spec.market_id,
                "station_id": spec.station_id,
                "target_date": target_date,
                "decision_horizon": "market_open" if idx % 3 == 0 else "morning_of",
                "decision_time_utc": target_date.tz_localize("UTC"),
                "market_spec_json": spec.model_dump_json(),
                "feature_availability_json": json.dumps(feature_availability),
                "realized_daily_max": base,
                "winning_outcome": winning,
                "lead_hours": float(6 + (idx % 12)),
                "model_daily_max": base - 0.2,
                "ecmwf_ifs025_model_daily_max": (base - 0.1) if idx % 4 != 0 else np.nan,
                "gfs_seamless_model_daily_max": (base + 0.1) if idx % 3 != 0 else np.nan,
                "humidity_mean": humidity,
            }
        )
    return pd.DataFrame(rows)


def test_sanitize_model_frame_preserves_missing_values() -> None:
    frame = pd.DataFrame({"model_daily_max": [10.0, np.nan], "realized_daily_max": [9.5, 10.5]})

    clean = sanitize_model_frame(frame)

    assert pd.isna(clean.loc[1, "model_daily_max"])


def test_contextual_feature_builder_adds_contextual_columns() -> None:
    frame = _training_frame().iloc[:4].copy()
    builder = ContextualFeatureBuilder(
        [
            "lead_hours",
            "model_daily_max",
            "ecmwf_ifs025_model_daily_max",
            "humidity_mean",
        ]
    ).fit(frame)

    transformed = builder.transform(frame.iloc[[0, 1]])

    assert "miss__humidity_mean" in transformed.columns
    assert "avail__ecmwf_ifs025_model_daily_max" in transformed.columns
    assert "day_of_year_sin" in transformed.columns
    assert any(column.startswith("city__") for column in transformed.columns)
    assert any(column.startswith("horizon__") for column in transformed.columns)
    assert "available_feature_fraction" in transformed.columns


def test_tuned_ensemble_and_det2prob_return_mixture_forecasts(tmp_path: Path) -> None:
    frame = _training_frame()
    spec = example_market_specs(["Seoul"])[0]

    for model_name in ("tuned_ensemble", "det2prob_nn"):
        artifact = train_model(model_name, frame, tmp_path / model_name, split_policy="market_day", seed=42)
        model = load_model(Path(artifact.path))
        weights, means, scales = model.predict(frame.iloc[:3].copy())
        forecast = predict_market(Path(artifact.path), model_name, spec, frame.iloc[[0]])

        assert weights.shape[0] == 3
        assert means.shape == weights.shape
        assert scales.shape == weights.shape
        assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(scales > 0.0)
        assert forecast.distribution_family == "gaussian_mixture"
        assert abs(sum(forecast.outcome_probabilities.values()) - 1.0) < 1e-6


def test_ablation_variants_support_gaussian_and_mixture_contracts(tmp_path: Path) -> None:
    frame = _training_frame()

    tuned_artifact = train_model(
        "tuned_ensemble",
        frame,
        tmp_path / "tuned_legacy",
        split_policy="market_day",
        seed=42,
        variant="legacy_fixed2",
    )
    det_artifact = train_model(
        "det2prob_nn",
        frame,
        tmp_path / "det_legacy",
        split_policy="market_day",
        seed=42,
        variant="legacy_gaussian",
    )

    tuned_model = load_model(Path(tuned_artifact.path))
    det_model = load_model(Path(det_artifact.path))

    tuned_mean, tuned_std = tuned_model.predict(frame.iloc[:3].copy())
    det_mean, det_std = det_model.predict(frame.iloc[:3].copy())

    assert tuned_artifact.variant == "legacy_fixed2"
    assert det_artifact.variant == "legacy_gaussian"
    assert tuned_mean.shape == (3,)
    assert tuned_std.shape == (3,)
    assert det_mean.shape == (3,)
    assert det_std.shape == (3,)
    assert np.all(tuned_std > 0.0)
    assert np.all(det_std > 0.0)
