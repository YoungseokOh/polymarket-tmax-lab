from __future__ import annotations

from pathlib import Path

import pandas as pd

from pmtmax.storage.lab_bootstrap import (
    archive_legacy_runs,
    export_seed_bundle,
    inventory_legacy_runs,
    restore_seed_bundle,
    restore_warehouse_from_seed,
)
from pmtmax.storage.warehouse import DataWarehouse


def _warehouse(data_root: Path, name: str = "warehouse") -> DataWarehouse:
    return DataWarehouse.from_paths(
        duckdb_path=data_root / "duckdb" / f"{name}.duckdb",
        parquet_root=data_root / "parquet",
        raw_root=data_root / "raw",
        manifest_root=data_root / "manifests",
        archive_root=data_root / "archive",
    )


def test_seed_export_and_restore_rebuilds_warehouse(tmp_path: Path) -> None:
    source_root = tmp_path / "source_data"
    source = _warehouse(source_root)
    source.raw_store.write_json("markets/snapshot/test_market.json", {"id": "m1"})
    source.upsert_table(
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
    )
    source.write_gold_table(
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
    source.compact()
    source.close()

    seed_path = tmp_path / "seed" / "pmtmax_seed.tar.gz"
    manifest = export_seed_bundle(
        data_root=source_root,
        raw_root=source_root / "raw",
        parquet_root=source_root / "parquet",
        manifest_root=source_root / "manifests",
        seed_path=seed_path,
    )

    assert seed_path.exists()
    assert manifest.file_count > 0

    target_root = tmp_path / "target_data"
    restored_manifest = restore_seed_bundle(seed_path=seed_path, data_root=target_root)
    target = _warehouse(target_root)
    original_read_parquet = pd.read_parquet

    def _fail_read_parquet(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError(f"restore_warehouse_from_seed should not call pandas.read_parquet: {args} {kwargs}")

    pd.read_parquet = _fail_read_parquet  # type: ignore[assignment]
    try:
        counts = restore_warehouse_from_seed(
            warehouse=target,
            parquet_root=target_root / "parquet",
            manifest_root=target_root / "manifests",
        )
    finally:
        pd.read_parquet = original_read_parquet  # type: ignore[assignment]

    assert restored_manifest.archive_path == str(seed_path)
    assert counts["bronze_market_snapshots"] == 1
    assert counts["gold_training_examples_tabular"] == 1
    assert len(target.read_table("gold_training_examples_tabular")) == 1
    assert (target_root / "parquet" / "gold" / "historical_training_set.parquet").exists()
    target.close()


def test_inventory_and_archive_legacy_runs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    (data_root / "raw" / "bronze").mkdir(parents=True)
    (data_root / "raw" / "backfill_smoke" / "markets").mkdir(parents=True)
    (data_root / "raw" / "backfill_smoke" / "markets" / "snapshot.json").write_text("{}")
    (data_root / "parquet" / "bronze").mkdir(parents=True)
    (data_root / "parquet" / "backfill_smoke").mkdir(parents=True)
    (data_root / "parquet" / "backfill_smoke" / "dataset.parquet").write_text("legacy")
    (data_root / "parquet" / "historical_training_set.parquet").write_text("legacy")
    (data_root / "manifests").mkdir(parents=True)
    (data_root / "manifests" / "warehouse_manifest.json").write_text("{}")

    inventory = inventory_legacy_runs(
        raw_root=data_root / "raw",
        parquet_root=data_root / "parquet",
        manifest_root=data_root / "manifests",
    )

    assert len(inventory.entries) == 3
    assert sorted(Path(entry.path).name for entry in inventory.entries) == [
        "backfill_smoke",
        "backfill_smoke",
        "historical_training_set.parquet",
    ]

    report = archive_legacy_runs(
        inventory=inventory,
        archive_root=data_root / "archive",
        dry_run=False,
    )

    assert len(report["archived_paths"]) == 3
    assert not (data_root / "raw" / "backfill_smoke").exists()
    assert not (data_root / "parquet" / "backfill_smoke").exists()
    assert (data_root / "archive" / "legacy-runs" / "raw" / "backfill_smoke").exists()
    assert (data_root / "archive" / "legacy-runs" / "parquet" / "historical_training_set.parquet").exists()
