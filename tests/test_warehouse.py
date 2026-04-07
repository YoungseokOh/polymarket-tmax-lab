from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pmtmax.storage.warehouse import DataWarehouse, ordered_legacy_paths


def _warehouse(tmp_path: Path, name: str) -> DataWarehouse:
    return DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / f"{name}.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
        recovery_root=tmp_path / "recovery",
    )


def test_warehouse_rejects_second_writer_lock(tmp_path: Path) -> None:
    primary = _warehouse(tmp_path, "warehouse")
    with pytest.raises(RuntimeError):
        _warehouse(tmp_path, "warehouse")
    primary.close()


def test_warehouse_migrates_legacy_gold_table(tmp_path: Path) -> None:
    legacy = _warehouse(tmp_path, "legacy")
    legacy.write_gold_table(
        "legacy_training_set",
        pd.DataFrame(
            [
                {
                    "market_id": "m1",
                    "decision_horizon": "morning_of",
                    "realized_daily_max": 8.0,
                }
            ]
        ),
        relative_path="warehouse/legacy_training_set.parquet",
    )
    legacy.close()

    canonical = _warehouse(tmp_path, "canonical")
    result = canonical.migrate_legacy_databases([tmp_path / "duckdb" / "legacy.duckdb"], archive_legacy=True)

    assert result["gold_training_examples_tabular"] == 1
    migrated = canonical.read_table("gold_training_examples_tabular")
    assert len(migrated) == 1
    assert migrated.iloc[0]["market_id"] == "m1"
    assert (tmp_path / "archive" / "legacy-duckdb").exists()
    canonical.close()


def test_ordered_legacy_paths_uses_cutover_precedence(tmp_path: Path) -> None:
    names = [
        "warehouse_smoke.duckdb",
        "pmtmax.duckdb",
        "openmeteo_fix_smoke2.duckdb",
        "build_dataset_single_run.duckdb",
    ]
    paths = [tmp_path / name for name in names]
    ordered = ordered_legacy_paths(paths)

    assert [path.name for path in ordered] == [
        "pmtmax.duckdb",
        "openmeteo_fix_smoke2.duckdb",
        "build_dataset_single_run.duckdb",
        "warehouse_smoke.duckdb",
    ]


def test_warehouse_validation_and_standard_exports(tmp_path: Path) -> None:
    legacy = _warehouse(tmp_path, "legacy")
    legacy.write_gold_table(
        "legacy_training_set",
        pd.DataFrame(
            [
                {
                    "market_id": "m1",
                    "decision_horizon": "morning_of",
                    "realized_daily_max": 8.0,
                }
            ]
        ),
        relative_path="warehouse/legacy_training_set.parquet",
    )
    legacy.close()

    canonical = _warehouse(tmp_path, "canonical")
    inventory = canonical.inventory_legacy_databases([tmp_path / "duckdb" / "legacy.duckdb"])
    canonical.migrate_legacy_databases([tmp_path / "duckdb" / "legacy.duckdb"], archive_legacy=False)
    canonical.compact()
    validation = canonical.validate_migration(inventory)

    assert validation["ok"] is True
    assert (tmp_path / "parquet" / "gold" / "historical_training_set.parquet").exists()
    canonical.close()


def test_warehouse_normalizes_missing_legacy_key_columns(tmp_path: Path) -> None:
    legacy = _warehouse(tmp_path, "legacy_forecast")
    legacy.duckdb_store.write_frame(
        "bronze_forecast_requests",
        pd.DataFrame(
            [
                {
                    "market_id": "m1",
                    "model_name": "ecmwf_ifs025",
                    "endpoint_kind": "historical_forecast",
                    "target_local_date": "2025-12-11",
                }
            ]
        ),
    )
    legacy.close()

    canonical = _warehouse(tmp_path, "canonical_forecast")
    canonical.migrate_legacy_databases([tmp_path / "duckdb" / "legacy_forecast.duckdb"], archive_legacy=False)
    frame = canonical.read_table("bronze_forecast_requests")

    assert len(frame) == 1
    assert frame.iloc[0]["request_kind"] == "full"
    assert "decision_horizon" in frame.columns
    canonical.close()


def test_warehouse_sql_upsert_replaces_null_key_rows_and_aligns_schema(tmp_path: Path) -> None:
    warehouse = _warehouse(tmp_path, "upsert")
    initial = pd.DataFrame(
        [
            {
                "market_id": "m0",
                "model_name": "ecmwf_ifs025",
                "endpoint_kind": "historical_forecast",
                "decision_horizon": None,
                "forecast_time_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
                "temperature_2m": 8.0,
            },
            {
                "market_id": "m1",
                "model_name": "ecmwf_ifs025",
                "endpoint_kind": "historical_forecast",
                "decision_horizon": None,
                "forecast_time_utc": pd.Timestamp("2025-01-01T01:00:00Z"),
                "temperature_2m": 10.0,
            },
        ]
    )
    warehouse.upsert_table("silver_forecast_runs_hourly", initial, return_frame=False)

    incremental = pd.DataFrame(
        [
            {
                "market_id": "m1",
                "model_name": "ecmwf_ifs025",
                "endpoint_kind": "historical_forecast",
                "decision_horizon": None,
                "forecast_time_utc": pd.Timestamp("2025-01-01T01:00:00Z"),
                "temperature_2m": 12.0,
                "requested_run_time_utc": pd.Timestamp("2024-12-31T18:00:00Z"),
            },
            {
                "market_id": "m1",
                "model_name": "ecmwf_ifs025",
                "endpoint_kind": "historical_forecast",
                "decision_horizon": None,
                "forecast_time_utc": pd.Timestamp("2025-01-01T01:00:00Z"),
                "temperature_2m": 14.0,
                "requested_run_time_utc": pd.Timestamp("2024-12-31T19:00:00Z"),
            },
            {
                "market_id": "m2",
                "model_name": "ecmwf_ifs025",
                "endpoint_kind": "historical_forecast",
                "decision_horizon": "morning_of",
                "forecast_time_utc": pd.Timestamp("2025-01-01T02:00:00Z"),
                "temperature_2m": 16.0,
                "requested_run_time_utc": pd.Timestamp("2024-12-31T20:00:00Z"),
            },
        ]
    )
    row_count = warehouse.upsert_table("silver_forecast_runs_hourly", incremental, return_frame=False)
    frame = warehouse.read_table("silver_forecast_runs_hourly").sort_values(["market_id", "forecast_time_utc"]).reset_index(drop=True)

    assert int(row_count) == 3
    assert list(frame["market_id"]) == ["m0", "m1", "m2"]
    assert float(frame.loc[frame["market_id"] == "m1", "temperature_2m"].iloc[0]) == 14.0
    assert pd.isna(frame.loc[frame["market_id"] == "m0", "requested_run_time_utc"].iloc[0])
    assert pd.Timestamp(frame.loc[frame["market_id"] == "m1", "requested_run_time_utc"].iloc[0]).tz_convert("UTC") == pd.Timestamp(
        "2024-12-31T19:00:00Z"
    )
    warehouse.close()


def test_warehouse_sql_upsert_handles_all_null_stage_key_without_demoting_target_type(tmp_path: Path) -> None:
    warehouse = _warehouse(tmp_path, "upsert_null_stage")
    initial = pd.DataFrame(
        [
            {
                "market_id": "m1",
                "model_name": "ecmwf_ifs025",
                "endpoint_kind": "historical_forecast",
                "request_kind": "full",
                "decision_horizon": "market_open",
                "target_local_date": pd.Timestamp("2025-01-01"),
            }
        ]
    )
    warehouse.upsert_table("bronze_forecast_requests", initial, return_frame=False)

    incremental = pd.DataFrame(
        [
            {
                "market_id": "m2",
                "model_name": "ecmwf_ifs025",
                "endpoint_kind": "historical_forecast",
                "request_kind": "probe",
                "decision_horizon": None,
                "target_local_date": pd.Timestamp("2025-01-02"),
            }
        ]
    )
    row_count = warehouse.upsert_table("bronze_forecast_requests", incremental, return_frame=False)
    frame = warehouse.read_table("bronze_forecast_requests").sort_values(["market_id"]).reset_index(drop=True)

    assert int(row_count) == 2
    assert list(frame["market_id"]) == ["m1", "m2"]
    assert str(frame.loc[0, "decision_horizon"]) == "market_open"
    assert pd.isna(frame.loc[1, "decision_horizon"])
    warehouse.close()


def test_write_manifest_uses_duckdb_metadata_without_table_reads(tmp_path: Path) -> None:
    warehouse = _warehouse(tmp_path, "manifest")
    warehouse.upsert_table(
        "bronze_market_snapshots",
        pd.DataFrame(
            [
                {
                    "market_id": "m1",
                    "captured_at": pd.Timestamp("2025-12-10T00:00:00Z"),
                    "question": "Highest temperature in Seoul on Dec 11?",
                }
            ]
        ),
        return_frame=False,
    )

    def _fail_read(_: str) -> pd.DataFrame:
        raise AssertionError("write_manifest should not read full tables")

    warehouse.read_table = _fail_read  # type: ignore[method-assign]
    manifest_path = warehouse.write_manifest()
    payload = json.loads(manifest_path.read_text())

    assert payload["tables"]["bronze_market_snapshots"]["rows"] == 1
    assert payload["tables"]["bronze_market_snapshots"]["columns"] == ["market_id", "captured_at", "question"]
    warehouse.close()


def test_compact_rewrites_parquet_without_table_readbacks(tmp_path: Path) -> None:
    warehouse = _warehouse(tmp_path, "compact")
    warehouse.upsert_table(
        "bronze_market_snapshots",
        pd.DataFrame(
            [
                {
                    "market_id": "m1",
                    "captured_at": pd.Timestamp("2025-12-10T00:00:00Z"),
                    "question": "Highest temperature in Seoul on Dec 11?",
                }
            ]
        ),
        return_frame=False,
    )
    warehouse.write_gold_table(
        "gold_training_examples_tabular",
        pd.DataFrame(
            [
                {
                    "market_id": "m1",
                    "decision_horizon": "morning_of",
                    "realized_daily_max": 8.0,
                    "forecast_source_kind": "fixture",
                    "available_models_json": "[]",
                    "selected_models_json": "[]",
                }
            ]
        ),
    )

    def _fail_read(_: str) -> pd.DataFrame:
        raise AssertionError("compact should not read full tables")

    warehouse.read_table = _fail_read  # type: ignore[method-assign]
    counts = warehouse.compact()

    assert counts["bronze_market_snapshots"] == 1
    assert counts["gold_training_examples_tabular"] == 1
    assert (tmp_path / "parquet" / "bronze" / "bronze_market_snapshots.parquet").exists()
    assert (tmp_path / "parquet" / "gold" / "historical_training_set.parquet").exists()
    warehouse.close()


def test_warehouse_blocks_canonical_gold_overwrite_without_unlock(tmp_path: Path) -> None:
    warehouse = _warehouse(tmp_path, "guarded")
    frame = pd.DataFrame([{"market_id": "m1", "decision_horizon": "morning_of", "realized_daily_max": 8.0}])
    warehouse.write_gold_table(
        "gold_training_examples_tabular",
        frame,
        relative_path="gold/v2/historical_training_set.parquet",
        allow_canonical_overwrite=True,
    )

    with pytest.raises(ValueError, match="Protected canonical output already exists"):
        warehouse.write_gold_table(
            "gold_training_examples_tabular",
            frame,
            relative_path="gold/v2/historical_training_set.parquet",
        )
    warehouse.close()


def test_warehouse_unlocks_canonical_overwrite_and_creates_recovery_backup(tmp_path: Path) -> None:
    warehouse = _warehouse(tmp_path, "recovery")
    first = pd.DataFrame([{"market_id": "m1", "decision_horizon": "morning_of", "realized_daily_max": 8.0}])
    second = pd.DataFrame([{"market_id": "m2", "decision_horizon": "morning_of", "realized_daily_max": 9.0}])
    warehouse.write_gold_table(
        "gold_training_examples_tabular",
        first,
        relative_path="gold/v2/historical_training_set.parquet",
        allow_canonical_overwrite=True,
    )
    warehouse.write_manifest()

    warehouse.write_gold_table(
        "gold_training_examples_tabular",
        second,
        relative_path="gold/v2/historical_training_set.parquet",
        allow_canonical_overwrite=True,
    )

    backups = list((tmp_path / "recovery").rglob("historical_training_set.parquet"))
    manifests = list((tmp_path / "recovery").rglob("warehouse_manifest.json"))
    assert backups
    assert manifests
    warehouse.close()


def test_write_gold_table_uses_parquet_metadata_for_shrinkage_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    warehouse = _warehouse(tmp_path, "gold_metadata")
    first = pd.DataFrame(
        [
            {"market_id": "m1", "decision_horizon": "morning_of", "realized_daily_max": 8.0},
            {"market_id": "m2", "decision_horizon": "morning_of", "realized_daily_max": 9.0},
        ]
    )
    second = pd.DataFrame(
        [
            {"market_id": "m3", "decision_horizon": "morning_of", "realized_daily_max": 10.0},
            {"market_id": "m4", "decision_horizon": "morning_of", "realized_daily_max": 11.0},
        ]
    )
    warehouse.write_gold_table(
        "gold_training_examples_tabular",
        first,
        relative_path="gold/v2/custom_training_set.parquet",
    )
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("write_gold_table should not call pandas.read_parquet")),
    )

    warehouse.write_gold_table(
        "gold_training_examples_tabular",
        second,
        relative_path="gold/v2/custom_training_set.parquet",
    )

    frame = warehouse.read_table("gold_training_examples_tabular").sort_values("market_id").reset_index(drop=True)
    assert list(frame["market_id"]) == ["m3", "m4"]
    warehouse.close()
