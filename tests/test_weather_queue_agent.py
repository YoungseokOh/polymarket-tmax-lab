from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from pmtmax.weather.queue_agent import (
    CollectionLogEntry,
    append_collection_log_entry,
    build_collection_note,
    classify_collection_outcome,
    infer_next_queue_start,
    load_weather_train_snapshot,
    parse_collection_log,
    refresh_weather_pretrain,
    render_status_markdown,
    should_refresh_pretrain,
)
from pmtmax.weather.training_data import (
    WeatherTrainingCollectionResult,
    WeatherTrainingPaths,
    default_weather_training_paths,
)


def test_infer_next_queue_start_prefers_status_hint(tmp_path: Path) -> None:
    status_path = tmp_path / "weather_train_status.md"
    status_path.write_text(
        "## Next Collection Queue\n"
        "1. Continue older gap-fill from `2024-07-02` forward with `7`-day chunks.\n",
        encoding="utf-8",
    )

    start_date = infer_next_queue_start(
        status_path=status_path,
        snapshot=load_weather_train_snapshot(tmp_path / "missing.parquet"),
        anchor_date=date(2024, 6, 1),
    )

    assert start_date == date(2024, 7, 2)


def test_append_collection_log_entry_round_trips(tmp_path: Path) -> None:
    log_path = tmp_path / "weather_train_collection_log.md"
    log_path.write_text(
        "# Weather Train Collection Log\n\n"
        "Append-only operational log for `weather_train`.\n\n"
        "| Run Date | Range | Mode | Outcome | Rows Added | Notes |\n"
        "| --- | --- | --- | --- | ---: | --- |\n",
        encoding="utf-8",
    )

    append_collection_log_entry(
        log_path,
        CollectionLogEntry(
            run_date="2026-04-24",
            range_text="2024-07-02..2024-07-08",
            mode="7-day queue agent",
            outcome="success",
            rows_added=210,
            notes="Full `210/210 available`; next older-backfill queue is `2024-07-09`.",
        ),
    )

    entries = parse_collection_log(log_path)

    assert len(entries) == 1
    assert entries[0].range_text == "2024-07-02..2024-07-08"
    assert entries[0].rows_added == 210


def test_render_status_markdown_reports_pretrain_gap(tmp_path: Path) -> None:
    paths = default_weather_training_paths(tmp_path / "parquet")
    paths.gold_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "station_id": ["A", "B", "A", "B", "A"],
            "target_date": [
                "2024-06-01",
                "2024-06-01",
                "2024-06-02",
                "2024-06-02",
                "2024-06-03",
            ],
        }
    )
    frame.to_parquet(paths.gold_path, index=False)
    snapshot = load_weather_train_snapshot(paths.gold_path)
    entries = [
        CollectionLogEntry(
            run_date="2026-04-24",
            range_text="2024-06-01..2024-06-02",
            mode="7-day queue agent",
            outcome="success",
            rows_added=4,
            notes="Full run",
        ),
        CollectionLogEntry(
            run_date="2026-04-24",
            range_text="2026-01-22..2026-01-28",
            mode="1-day chunks",
            outcome="retry-only",
            rows_added=0,
            notes="429",
        ),
    ]
    metadata = {
        "path": "artifacts/workspaces/weather_train/models/v2/gaussian_emos.pkl",
        "dataset_signature": "abc123",
        "trained_at": "2026-04-24T11:34:21Z",
        "weather_training_rows": 4,
    }

    markdown = render_status_markdown(
        snapshot=snapshot,
        entries=entries,
        pretrain_metadata=metadata,
        next_queue_start=date(2024, 6, 3),
        chunk_days=7,
    )

    assert "total rows: `5`" in markdown
    assert "weather pretrain artifact trails the current dataset by `1` rows" in markdown
    assert "`2026-01-22..2026-01-28`" in markdown


def test_render_status_markdown_drops_retry_only_dates_that_later_succeeded(tmp_path: Path) -> None:
    paths = default_weather_training_paths(tmp_path / "parquet")
    paths.gold_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "station_id": ["A", "B"],
            "target_date": ["2024-06-01", "2024-06-01"],
        }
    )
    frame.to_parquet(paths.gold_path, index=False)
    snapshot = load_weather_train_snapshot(paths.gold_path)
    entries = [
        CollectionLogEntry(
            run_date="2026-04-24",
            range_text="2024-06-01",
            mode="single-day probe",
            outcome="retry-only",
            rows_added=0,
            notes="429",
        ),
        CollectionLogEntry(
            run_date="2026-04-24",
            range_text="2024-06-01",
            mode="single-day probe",
            outcome="success",
            rows_added=2,
            notes="reopened",
        ),
    ]

    markdown = render_status_markdown(
        snapshot=snapshot,
        entries=entries,
        pretrain_metadata={},
        next_queue_start=date(2024, 6, 2),
        chunk_days=7,
    )

    assert "`2024-06-01`: `0/2 available`" not in markdown


def test_rate_limit_cancelled_outcome_is_logged_with_attempted_rows(tmp_path: Path) -> None:
    result = WeatherTrainingCollectionResult(
        paths=WeatherTrainingPaths(
            bronze_path=tmp_path / "bronze.parquet",
            silver_path=tmp_path / "silver.parquet",
            gold_path=tmp_path / "gold.parquet",
        ),
        requested_rows=210,
        attempted_rows=2,
        collected_rows=0,
        skipped_existing_rows=0,
        failed_rows=2,
        status_counts={"retryable_error": 2},
        failures=[],
        workers_requested=1,
        workers_effective=1,
        early_stop_reason="consecutive_429:2",
    )

    outcome = classify_collection_outcome(result)
    note = build_collection_note(result=result, outcome=outcome, next_queue_start=date(2025, 1, 28))

    assert outcome == "rate-limit-cancelled"
    assert "Cancelled after `2` consecutive `429` responses" in note
    assert "`2/210` planned requests attempted" in note


def test_should_refresh_pretrain_uses_row_gap_threshold(tmp_path: Path) -> None:
    paths = default_weather_training_paths(tmp_path / "parquet")
    paths.gold_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "station_id": ["A", "B", "A", "B", "A"],
            "target_date": ["2024-06-01", "2024-06-01", "2024-06-02", "2024-06-02", "2024-06-03"],
        }
    ).to_parquet(paths.gold_path, index=False)
    snapshot = load_weather_train_snapshot(paths.gold_path)

    assert should_refresh_pretrain(snapshot=snapshot, metadata={"weather_training_rows": 4}, threshold_rows=1) is True
    assert should_refresh_pretrain(snapshot=snapshot, metadata={"weather_training_rows": 4}, threshold_rows=2) is False
    assert should_refresh_pretrain(snapshot=snapshot, metadata={"weather_training_rows": 0}, threshold_rows=0) is False


def test_refresh_weather_pretrain_writes_sidecar_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "weather_training_set.parquet"
    frame = pd.DataFrame(
        {
            "station_id": ["RKSI", "RKSI", "KATL", "KATL"],
            "target_date": ["2024-06-01", "2024-06-02", "2024-06-01", "2024-06-02"],
            "forecast_temperature_2m_min": [20.0, 21.0, 24.0, 25.0],
            "forecast_temperature_2m_mean": [24.0, 25.0, 28.0, 29.0],
            "forecast_temperature_2m_max": [28.0, 29.0, 32.0, 33.0],
            "forecast_dew_point_2m_mean": [18.0, 19.0, 20.0, 21.0],
            "forecast_relative_humidity_2m_mean": [60.0, 62.0, 58.0, 57.0],
            "forecast_wind_speed_10m_mean": [3.0, 4.0, 5.0, 4.5],
            "forecast_cloud_cover_mean": [40.0, 45.0, 35.0, 30.0],
            "realized_daily_max": [27.0, 28.5, 31.0, 32.0],
        }
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dataset_path, index=False)

    payload = refresh_weather_pretrain(
        dataset_path=dataset_path,
        artifacts_dir=tmp_path / "artifacts" / "models" / "v2",
        dataset_profile="weather_real",
        workspace_name="weather_train",
        seed=42,
        model_name="gaussian_emos",
        variant=None,
    )

    sidecar = Path(payload["path"]).with_suffix(".json")
    assert Path(payload["path"]).exists()
    assert sidecar.exists()
    assert payload["weather_training_rows"] == 4
    assert payload["dataset_profile"] == "weather_real"
    assert payload["workspace_name"] == "weather_train"
