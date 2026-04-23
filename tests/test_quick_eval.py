from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmtmax.markets.repository import bundled_market_snapshots
from pmtmax.modeling import quick_eval
from pmtmax.modeling.bin_mapper import infer_winning_label
from pmtmax.modeling.quick_eval import _values_to_celsius


def test_values_to_celsius_converts_temperatures_and_scales_per_row() -> None:
    units = np.asarray(["F", "C"], dtype=object)

    temperatures = _values_to_celsius(np.asarray([50.0, 10.0]), units)
    scales = _values_to_celsius(np.asarray([9.0, 5.0]), units, scale=True)

    assert np.allclose(temperatures, np.asarray([10.0, 10.0]))
    assert np.allclose(scales, np.asarray([5.0, 5.0]))


def test_evaluate_saved_model_averages_only_rows_with_probability_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = bundled_market_snapshots(["Seoul"])[0].spec
    assert spec is not None
    winner = infer_winning_label(spec, 10.0)

    class FakeModel:
        def predict(self, _frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            return np.asarray([0.0, 10.0]), np.asarray([1.0, 1.0])

    monkeypatch.setattr(quick_eval, "load_model", lambda _: FakeModel())
    holdout = pd.DataFrame(
        [
            {
                "market_spec_json": "{bad json",
                "realized_daily_max": 100.0,
                "winning_outcome": winner,
            },
            {
                "market_spec_json": spec.model_dump_json(),
                "realized_daily_max": 10.0,
                "winning_outcome": winner,
            },
        ]
    )

    metrics = quick_eval.evaluate_saved_model(Path("unused.pkl"), holdout)

    assert metrics is not None
    assert metrics["n"] == 1.0
    assert metrics["mae_celsius_normalized"] == 0.0
