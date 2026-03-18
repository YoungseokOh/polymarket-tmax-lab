from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pmtmax.backfill import BackfillPipeline
from pmtmax.http import CachedHttpClient
from pmtmax.markets.repository import bundled_market_snapshots
from pmtmax.storage.warehouse import DataWarehouse


class _BrokenOpenMeteoClient:
    def historical_forecast(self, **_: object) -> dict:
        raise RuntimeError("force fixture fallback")

    def forecast(self, **_: object) -> dict:
        raise RuntimeError("force fixture fallback")


def _load_fixture(city: str) -> dict:
    fixture_name = city.lower().replace(" ", "_") + "_daily.json"
    return json.loads((Path("tests/fixtures/openmeteo") / fixture_name).read_text())


class _SingleRunOpenMeteoClient:
    def historical_forecast(self, *, latitude: float, longitude: float, model: str, hourly: list[str], start_date: str, end_date: str, timezone: str) -> dict:  # noqa: ARG002
        return _load_fixture("Seoul")

    def forecast(self, **_: object) -> dict:
        return _load_fixture("Seoul")

    def single_run(self, *, latitude: float, longitude: float, model: str, hourly: list[str], run: str, forecast_days: int, timezone: str) -> dict:  # noqa: ARG002
        payload = _load_fixture("Seoul")
        payload["hourly"]["temperature_2m"] = [value + 1.0 for value in payload["hourly"]["temperature_2m"]]
        return payload


def test_backfill_pipeline_builds_bronze_silver_gold_tables(tmp_path: Path) -> None:
    snapshots = bundled_market_snapshots()
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "test.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=CachedHttpClient(tmp_path / "cache"),
        openmeteo=_BrokenOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025", "ecmwf_aifs025_single"],
        truth_snapshot_dir=Path("tests/fixtures/truth"),
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )

    market_tables = pipeline.backfill_markets(snapshots, source_name="test")
    forecast_tables = pipeline.backfill_forecasts(
        snapshots,
        allow_fixture_fallback=True,
        strict_archive=False,
    )
    truth_tables = pipeline.backfill_truth(snapshots)
    gold = pipeline.materialize_training_set(
        snapshots,
        output_name="test_training_set",
        decision_horizons=["market_open", "morning_of"],
    )

    assert len(market_tables["bronze_market_snapshots"]) == 4
    assert len(market_tables["silver_market_specs"]) == 4
    assert len(forecast_tables["bronze_forecast_requests"]) == 16
    assert len(forecast_tables["silver_forecast_runs_hourly"]) > 0
    assert len(truth_tables["bronze_truth_snapshots"]) == 4
    assert len(truth_tables["silver_observations_daily"]) == 4
    assert len(gold) == 8
    assert (tmp_path / "parquet" / "silver" / "silver_forecast_runs_hourly.parquet").exists()
    assert (tmp_path / "parquet" / "gold" / "test_training_set.parquet").exists()
    assert (tmp_path / "parquet" / "gold" / "test_training_set_sequence.parquet").exists()
    assert (tmp_path / "manifests" / "warehouse_manifest.json").exists()
    assert any((tmp_path / "raw").rglob("*.json"))


def test_backfill_pipeline_strict_archive_skips_fixture_fallback(tmp_path: Path) -> None:
    snapshots = bundled_market_snapshots()
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "strict.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=CachedHttpClient(tmp_path / "cache"),
        openmeteo=_BrokenOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025"],
        truth_snapshot_dir=Path("tests/fixtures/truth"),
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )

    forecast_tables = pipeline.backfill_forecasts(
        snapshots,
        allow_fixture_fallback=False,
        strict_archive=True,
    )

    assert len(forecast_tables["bronze_forecast_requests"]) == 8
    assert forecast_tables["silver_forecast_runs_hourly"].empty
    assert set(forecast_tables["bronze_forecast_requests"]["request_kind"]) == {"probe", "full"}


def test_backfill_pipeline_single_run_horizon_overrides_generic_rows(tmp_path: Path) -> None:
    snapshots = bundled_market_snapshots(["Seoul"])
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "single_run.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=CachedHttpClient(tmp_path / "cache"),
        openmeteo=_SingleRunOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025"],
        truth_snapshot_dir=Path("tests/fixtures/truth"),
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )

    pipeline.backfill_markets(snapshots, source_name="single_run_test")
    forecast_tables = pipeline.backfill_forecasts(
        snapshots,
        strict_archive=True,
        single_run_horizons=["morning_of"],
    )
    pipeline.backfill_truth(snapshots)
    gold = pipeline.materialize_training_set(
        snapshots,
        output_name="single_run_training_set",
        decision_horizons=["morning_of"],
    )

    single_run_rows = forecast_tables["silver_forecast_runs_hourly"].loc[
        forecast_tables["silver_forecast_runs_hourly"]["endpoint_kind"] == "single_run"
    ]
    assert not single_run_rows.empty
    assert set(single_run_rows["decision_horizon"].dropna().astype(str)) == {"morning_of"}
    assert len(gold) == 1
    row = gold.iloc[0]
    assert row["forecast_source_kind"] == "single_run"
    assert json.loads(str(row["selected_models_json"])) == ["ecmwf_ifs025"]
    assert pd.notna(row["issue_time_utc"])

    availability = pipeline.summarize_forecast_availability(top_k=2)
    assert not availability["summary"].empty
    assert not availability["recommended"].empty
