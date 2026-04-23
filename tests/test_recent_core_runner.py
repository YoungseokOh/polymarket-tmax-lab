from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_recent_core_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("run_recent_core_benchmark", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_backtest_split_count = _MODULE._backtest_split_count
_filter_prebuilt_city_frame = _MODULE._filter_prebuilt_city_frame
_panel_summary = _MODULE._panel_summary
_validate_retrain_stride = _MODULE._validate_retrain_stride


def test_filter_prebuilt_city_frame_keeps_last_market_days() -> None:
    frame = pd.DataFrame(
        {
            "city": ["London"] * 6 + ["NYC"],
            "target_date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-05",
                ]
            ),
            "decision_horizon": ["market_open"] * 7,
        }
    )

    result = _filter_prebuilt_city_frame(
        frame,
        city="London",
        label="dataset",
        last_n_market_days=2,
    )

    assert sorted(result["target_date"].dt.strftime("%Y-%m-%d").unique()) == ["2026-01-04", "2026-01-05"]
    assert set(result["city"]) == {"London"}


def test_validate_retrain_stride_rejects_single_retrain_recent_core() -> None:
    with pytest.raises(ValueError, match="Invalid recent-core retrain stride"):
        _validate_retrain_stride(retrain_stride=30, split_count=26, city="London")


def test_backtest_split_count_honors_last_n_groups() -> None:
    frame = pd.DataFrame(
        {
            "market_id": [f"m{i:03d}" for i in range(5) for _ in range(3)],
            "target_date": [pd.Timestamp("2026-01-01") + pd.Timedelta(days=i) for i in range(5) for _ in range(3)],
        }
    )

    assert _backtest_split_count(frame, backtest_last_n=0) == 4
    assert _backtest_split_count(frame, backtest_last_n=2) == 2


def test_panel_summary_filters_to_last_n_test_groups(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "market_id": ["m1", "m2", "m3"],
            "target_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "decision_time_utc": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
            "decision_horizon": ["morning_of", "morning_of", "morning_of"],
        }
    )
    panel = pd.DataFrame(
        {
            "market_id": ["m1", "m2", "m3"],
            "target_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "decision_horizon": ["morning_of", "morning_of", "morning_of"],
            "coverage_status": ["missing", "ok", "ok"],
        }
    )
    dataset_path = tmp_path / "dataset.parquet"
    panel_path = tmp_path / "panel.parquet"
    dataset.to_parquet(dataset_path)
    panel.to_parquet(panel_path)

    summary = _panel_summary(panel_path, dataset_path=dataset_path, backtest_last_n=2)

    assert summary["rows"] == 2
    assert summary["coverage"] == {"ok": 2}
