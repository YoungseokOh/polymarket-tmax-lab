from __future__ import annotations

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
