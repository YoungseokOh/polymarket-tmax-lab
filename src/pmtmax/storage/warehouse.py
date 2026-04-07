"""Warehouse helpers for canonical bronze/silver/gold research tables."""

from __future__ import annotations

import atexit
import datetime as dt
import fcntl
import shutil
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from pmtmax.storage.duckdb_store import DuckDBStore
from pmtmax.storage.parquet_store import ParquetStore
from pmtmax.storage.raw_store import RawStore
from pmtmax.storage.schemas import IngestRun, WarehouseManifest
from pmtmax.utils import dump_json, stable_hash

TABLE_KEYS: dict[str, list[str]] = {
    "bronze_ingest_runs": ["run_id"],
    "bronze_market_snapshots": ["market_id", "captured_at"],
    "bronze_forecast_requests": [
        "market_id",
        "model_name",
        "endpoint_kind",
        "request_kind",
        "decision_horizon",
        "target_local_date",
    ],
    "bronze_truth_snapshots": ["market_id", "target_local_date"],
    "bronze_price_snapshots": ["market_id", "token_id", "captured_at"],
    "bronze_price_history_requests": ["market_id", "token_id", "requested_at"],
    "silver_market_specs": ["market_id"],
    "silver_forecast_runs_hourly": [
        "market_id",
        "model_name",
        "endpoint_kind",
        "decision_horizon",
        "forecast_time_utc",
    ],
    "silver_observations_daily": ["market_id", "target_local_date"],
    "silver_price_timeseries": ["market_id", "token_id", "timestamp"],
    "dim_station": ["station_id"],
    "dim_model": ["model_name"],
    "dim_source": ["source_key"],
    "gold_training_examples": ["market_id", "decision_horizon"],
    "gold_training_examples_tabular": ["market_id", "decision_horizon"],
    "gold_training_examples_sequence": ["market_id", "decision_horizon", "model_name"],
    "gold_backtest_panel": ["market_id", "decision_horizon", "token_id"],
}

LEGACY_GOLD_COLUMNS = {
    "market_id",
    "decision_horizon",
    "realized_daily_max",
}

LEGACY_PRECEDENCE: dict[str, int] = {
    "pmtmax.duckdb": 0,
    "smoke.duckdb": 1,
    "backfill_smoke.duckdb": 2,
    "backfill_smoke2.duckdb": 3,
    "backfill_smoke3.duckdb": 4,
    "openmeteo_fix_smoke.duckdb": 5,
    "openmeteo_fix_smoke2.duckdb": 6,
    "single_run_cli.duckdb": 7,
    "build_dataset_single_run.duckdb": 8,
    "warehouse_smoke.duckdb": 9,
}

PROTECTED_CANONICAL_GOLD_PATHS = frozenset(
    {
        "gold/v2/historical_training_set.parquet",
        "gold/v2/historical_training_set_compat.parquet",
        "gold/v2/historical_training_set_sequence.parquet",
        "gold/v2/historical_backtest_panel.parquet",
    }
)

NUMERIC_DUCKDB_TYPE_PREFIXES = (
    "BIGINT",
    "DECIMAL",
    "DOUBLE",
    "FLOAT",
    "HUGEINT",
    "INTEGER",
    "REAL",
    "SMALLINT",
    "TINYINT",
    "UBIGINT",
    "UHUGEINT",
    "UINTEGER",
    "USMALLINT",
    "UTINYINT",
)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def parquet_relative_path(table_name: str) -> str:
    """Map a table to its canonical parquet mirror path."""

    if table_name.startswith("bronze_"):
        return f"bronze/{table_name}.parquet"
    if table_name.startswith("silver_"):
        return f"silver/{table_name}.parquet"
    if table_name.startswith("gold_"):
        return f"gold/{table_name}.parquet"
    if table_name.startswith("dim_"):
        return f"silver/dim/{table_name}.parquet"
    return f"warehouse/{table_name}.parquet"


def ordered_legacy_paths(paths: list[Path]) -> list[Path]:
    """Sort legacy paths by the cutover precedence order."""

    return sorted(paths, key=lambda path: (LEGACY_PRECEDENCE.get(path.name, 10_000), path.name))


def backup_duckdb_file(duckdb_path: Path, archive_root: Path) -> Path | None:
    """Copy an existing canonical DuckDB file into the pre-migration archive."""

    if not duckdb_path.exists() or duckdb_path.stat().st_size == 0:
        return None
    timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    destination_dir = archive_root / "pre-migration"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{duckdb_path.stem}_{timestamp}{duckdb_path.suffix}"
    shutil.copy2(duckdb_path, destination)
    return destination


@dataclass
class DataWarehouse:
    """Coordinate DuckDB tables, Parquet exports, raw payload archives, and manifests."""

    duckdb_store: DuckDBStore
    parquet_store: ParquetStore
    raw_store: RawStore
    manifest_root: Path
    archive_root: Path
    recovery_root: Path
    lock_path: Path
    _lock_handle: Any = field(repr=False)
    _recovery_session_dir: Path | None = field(default=None, repr=False)
    _recovery_manifest_copied: bool = field(default=False, repr=False)

    @classmethod
    def from_paths(
        cls,
        *,
        duckdb_path: Path,
        parquet_root: Path,
        raw_root: Path,
        manifest_root: Path,
        archive_root: Path,
        recovery_root: Path | None = None,
    ) -> DataWarehouse:
        """Create a warehouse from concrete storage paths and acquire a writer lock."""

        duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_root.mkdir(parents=True, exist_ok=True)
        raw_root.mkdir(parents=True, exist_ok=True)
        manifest_root.mkdir(parents=True, exist_ok=True)
        archive_root.mkdir(parents=True, exist_ok=True)
        resolved_recovery_root = recovery_root or (archive_root / "recovery")
        resolved_recovery_root.mkdir(parents=True, exist_ok=True)
        lock_path = duckdb_path.with_suffix(duckdb_path.suffix + ".lock")
        lock_handle = cls._acquire_lock(lock_path)
        warehouse = cls(
            duckdb_store=DuckDBStore(duckdb_path),
            parquet_store=ParquetStore(parquet_root),
            raw_store=RawStore(raw_root / "bronze"),
            manifest_root=manifest_root,
            archive_root=archive_root,
            recovery_root=resolved_recovery_root,
            lock_path=lock_path,
            _lock_handle=lock_handle,
        )
        atexit.register(warehouse.close)
        return warehouse

    @staticmethod
    def _acquire_lock(lock_path: Path) -> Any:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            msg = f"Warehouse lock already held for {lock_path}."
            raise RuntimeError(msg) from exc
        return handle

    def close(self) -> None:
        """Close the warehouse lock and the underlying DuckDB connection."""

        if getattr(self, "_lock_handle", None) is not None:
            with suppress(Exception):
                self.duckdb_store.close()
            with suppress(Exception):
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            with suppress(Exception):
                self._lock_handle.close()
            self._lock_handle = None

    def __del__(self) -> None:
        self.close()

    def start_run(
        self,
        *,
        command: str,
        config_hash: str,
        code_version: str = "workspace",
        notes: str = "",
    ) -> IngestRun:
        """Create or update a running ingest/materialization run record."""

        started_at = dt.datetime.now(tz=dt.UTC)
        run_id = stable_hash(f"{command}|{config_hash}|{started_at.isoformat()}")[:16]
        record = IngestRun(
            run_id=run_id,
            command=command,
            status="running",
            config_hash=config_hash,
            code_version=code_version,
            started_at=started_at,
            notes=notes,
        )
        self.upsert_table("bronze_ingest_runs", pd.DataFrame([record.model_dump(mode="json")]))
        return record

    def finish_run(self, run: IngestRun, *, status: str, notes: str = "") -> IngestRun:
        """Persist the terminal state of an ingest/materialization run."""

        updated = run.model_copy(
            update={
                "status": status,
                "completed_at": dt.datetime.now(tz=dt.UTC),
                "notes": notes or run.notes,
            }
        )
        self.upsert_table("bronze_ingest_runs", pd.DataFrame([updated.model_dump(mode="json")]))
        return updated

    def upsert_table(
        self,
        table_name: str,
        frame: pd.DataFrame,
        *,
        export_parquet: bool = True,
        return_frame: bool = True,
    ) -> pd.DataFrame | int:
        """Upsert a DataFrame into a managed warehouse table."""

        keys = TABLE_KEYS.get(table_name)
        if keys is None:
            msg = f"Unsupported warehouse table: {table_name}"
            raise ValueError(msg)
        if frame.empty:
            if export_parquet and self.table_exists(table_name):
                self.export_table(table_name)
            return self.read_table(table_name) if return_frame else self.table_row_count(table_name)
        row_count = self.duckdb_store.upsert_frame(table_name, frame, subset=keys)
        if export_parquet:
            self.export_table(table_name)
        return self.read_table(table_name) if return_frame else row_count

    def write_gold_table(
        self,
        table_name: str,
        frame: pd.DataFrame,
        relative_path: str | None = None,
        *,
        allow_canonical_overwrite: bool = False,
    ) -> Path:
        """Replace a gold table and export it to Parquet."""

        if frame.empty:
            msg = f"{table_name} is empty; nothing to materialize"
            raise ValueError(msg)

        target = relative_path or parquet_relative_path(table_name)

        # Shrinkage guard: refuse to overwrite if new frame is <50% of existing rows.
        # Prevents accidental dataset destruction (e.g. build-dataset without --markets-path).
        existing_path = self.parquet_store.root / target
        if existing_path.exists():
            if target in PROTECTED_CANONICAL_GOLD_PATHS and not allow_canonical_overwrite:
                msg = (
                    f"Protected canonical output already exists: {target}. "
                    "Pass --allow-canonical-overwrite to replace it. "
                    "A timestamped recovery backup will be created first."
                )
                raise ValueError(msg)
            try:
                existing_rows = self.duckdb_store.parquet_row_count(existing_path)
                if existing_rows > 0 and len(frame) < existing_rows * 0.5:
                    msg = (
                        f"Shrinkage guard: refusing to overwrite {target} "
                        f"({existing_rows} rows → {len(frame)} rows, "
                        f"{len(frame)/existing_rows:.0%} of original). "
                        "Use a complete --markets-path and only unlock canonical overwrite "
                        "when you intend to promote a rebuilt dataset."
                    )
                    raise ValueError(msg)
            except (ImportError, Exception) as _e:
                if "Shrinkage guard" in str(_e):
                    raise
                # If we can't read the existing file, proceed normally

            if target in PROTECTED_CANONICAL_GOLD_PATHS:
                self._backup_protected_output(existing_path, target)

            # Backup existing parquet before overwriting
            import shutil as _shutil
            _shutil.copy2(existing_path, str(existing_path) + ".bak")

        self.duckdb_store.write_frame(table_name, frame)
        return self.duckdb_store.export_table(table_name, self.parquet_store.root / target)

    def _backup_protected_output(self, existing_path: Path, relative_path: str) -> None:
        recovery_dir = self._recovery_session_path()
        destination = recovery_dir / "parquet" / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(existing_path, destination)

        if self._recovery_manifest_copied:
            return
        manifest_path = self.manifest_root / "warehouse_manifest.json"
        if manifest_path.exists():
            manifest_destination = recovery_dir / "manifests" / "warehouse_manifest.json"
            manifest_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest_path, manifest_destination)
        self._recovery_manifest_copied = True

    def _recovery_session_path(self) -> Path:
        if self._recovery_session_dir is None:
            timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
            self._recovery_session_dir = self.recovery_root / timestamp
            self._recovery_session_dir.mkdir(parents=True, exist_ok=True)
        return self._recovery_session_dir

    def read_table(self, table_name: str) -> pd.DataFrame:
        """Read a warehouse table or return an empty DataFrame if absent."""

        if not self.duckdb_store.table_exists(table_name):
            return pd.DataFrame()
        return self.duckdb_store.read_table(table_name)

    def read_query(self, query: str, parameters: list[Any] | None = None) -> pd.DataFrame:
        """Read an arbitrary DuckDB query as a DataFrame."""

        return self.duckdb_store.read_frame(query, parameters)

    def table_exists(self, table_name: str) -> bool:
        """Return whether a warehouse table exists."""

        return self.duckdb_store.table_exists(table_name)

    def table_row_count(self, table_name: str) -> int:
        """Return the row count for a warehouse table."""

        if not self.table_exists(table_name):
            return 0
        return self.duckdb_store.row_count(table_name)

    def export_table(self, table_name: str) -> Path | None:
        """Rewrite a managed parquet mirror directly from DuckDB."""

        if not self.table_exists(table_name):
            return None
        return self.duckdb_store.export_table(table_name, self.parquet_store.root / parquet_relative_path(table_name))

    def list_tables(self) -> list[str]:
        """List currently materialized tables in the canonical warehouse."""

        return sorted(self.duckdb_store.list_tables())

    def compact(self) -> dict[str, int]:
        """Rewrite parquet mirrors for all known tables currently present."""

        counts: dict[str, int] = {}
        for table_name in self.list_tables():
            counts[table_name] = self.table_row_count(table_name)
            self.export_table(table_name)
        self._write_standard_gold_exports()
        self.write_manifest()
        return counts

    def write_manifest(self) -> Path:
        """Write a JSON manifest describing the canonical warehouse state."""

        tables: dict[str, dict[str, Any]] = {}
        for table_name in self.list_tables():
            tables[table_name] = {
                "rows": self.table_row_count(table_name),
                "columns": self.duckdb_store.column_names(table_name),
                "parquet_path": parquet_relative_path(table_name),
            }
        manifest = WarehouseManifest(
            generated_at=dt.datetime.now(tz=dt.UTC),
            duckdb_path=str(self.duckdb_store.path),
            raw_root=str(self.raw_store.root),
            parquet_root=str(self.parquet_store.root),
            manifest_root=str(self.manifest_root),
            tables=tables,
        )
        path = self.manifest_root / "warehouse_manifest.json"
        dump_json(path, manifest.model_dump(mode="json"))
        return path

    def inventory_legacy_databases(self, legacy_paths: list[Path]) -> dict[str, Any]:
        """Inspect legacy DuckDB files and summarize recognized tables."""

        inventory: dict[str, Any] = {
            "generated_at": dt.datetime.now(tz=dt.UTC).isoformat(),
            "databases": [],
            "max_rows_by_target_table": {},
        }
        canonical_path = self.duckdb_store.path.resolve()
        for path in ordered_legacy_paths(legacy_paths):
            if not path.exists() or path.resolve() == canonical_path:
                continue
            connection = duckdb.connect(str(path), read_only=True)
            try:
                table_rows: list[dict[str, Any]] = []
                for legacy_table in [row[0] for row in connection.execute("show tables").fetchall()]:
                    target_table = self._resolve_legacy_table_name(connection, legacy_table)
                    row = connection.execute(f"select count(*) from {legacy_table}").fetchone()  # noqa: S608
                    count = int(row[0]) if row is not None else 0
                    table_rows.append(
                        {
                            "legacy_table": legacy_table,
                            "target_table": target_table,
                            "rows": count,
                        }
                    )
                    if target_table is not None:
                        current_max = inventory["max_rows_by_target_table"].get(target_table, 0)
                        inventory["max_rows_by_target_table"][target_table] = max(current_max, count)
                inventory["databases"].append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "precedence": LEGACY_PRECEDENCE.get(path.name, 10_000),
                        "tables": table_rows,
                    }
                )
            finally:
                connection.close()
        return inventory

    def migrate_legacy_databases(
        self,
        legacy_paths: list[Path],
        *,
        archive_legacy: bool = False,
    ) -> dict[str, int]:
        """Merge recognized tables from legacy DuckDB files into the canonical warehouse."""

        migrated_rows: dict[str, int] = {}
        canonical_path = self.duckdb_store.path.resolve()
        for path in ordered_legacy_paths(legacy_paths):
            if path.resolve() == canonical_path or not path.exists():
                continue
            connection = duckdb.connect(str(path), read_only=True)
            try:
                legacy_tables = [row[0] for row in connection.execute("show tables").fetchall()]
                for legacy_table in legacy_tables:
                    target_table = self._resolve_legacy_table_name(connection, legacy_table)
                    if target_table is None:
                        continue
                    frame = connection.execute(f"select * from {legacy_table}").fetch_df()  # noqa: S608
                    if frame.empty:
                        continue
                    frame = self._normalize_legacy_frame(frame, target_table=target_table, source_path=path)
                    current_rows = self.upsert_table(target_table, frame, return_frame=False)
                    migrated_rows[target_table] = int(current_rows)
            finally:
                connection.close()
            if archive_legacy:
                self._archive_legacy_database(path)
        self.write_manifest()
        return migrated_rows

    def archive_legacy_databases(self, legacy_paths: list[Path]) -> list[str]:
        """Move migrated legacy databases into the archive directory."""

        archived: list[str] = []
        canonical_path = self.duckdb_store.path.resolve()
        for path in ordered_legacy_paths(legacy_paths):
            if not path.exists() or path.resolve() == canonical_path:
                continue
            self._archive_legacy_database(path)
            archived.append(str(path))
        return archived

    def validate_migration(self, inventory: dict[str, Any]) -> dict[str, Any]:
        """Validate canonical table counts and duplicate constraints after migration."""

        canonical_counts: dict[str, int] = {}
        duplicates: dict[str, int] = {}
        for table_name in self.list_tables():
            frame = self.read_table(table_name)
            canonical_counts[table_name] = len(frame)
            keys = TABLE_KEYS.get(table_name)
            if keys:
                duplicates[table_name] = int(frame.duplicated(subset=keys).sum())
        threshold_failures: dict[str, dict[str, int]] = {}
        for table_name, legacy_max in inventory.get("max_rows_by_target_table", {}).items():
            canonical_count = canonical_counts.get(table_name, 0)
            if canonical_count < legacy_max:
                threshold_failures[table_name] = {
                    "canonical_rows": canonical_count,
                    "required_min_rows": legacy_max,
                }
        duplicate_failures = {table_name: count for table_name, count in duplicates.items() if count > 0}
        required_tables = sorted(set(inventory.get("max_rows_by_target_table", {}).keys()))
        missing_required = [table_name for table_name in required_tables if table_name not in canonical_counts]
        validation = {
            "ok": not threshold_failures and not duplicate_failures and not missing_required,
            "canonical_counts": canonical_counts,
            "duplicate_failures": duplicate_failures,
            "threshold_failures": threshold_failures,
            "missing_required_tables": missing_required,
        }
        return validation

    def _archive_legacy_database(self, path: Path) -> None:
        destination_dir = self.archive_root / "legacy-duckdb"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / path.name
        if destination.exists():
            suffix = stable_hash(path.name + str(dt.datetime.now(tz=dt.UTC).timestamp()))[:8]
            destination = destination_dir / f"{path.stem}_{suffix}{path.suffix}"
        shutil.move(str(path), str(destination))

    def _normalize_legacy_frame(self, frame: pd.DataFrame, *, target_table: str, source_path: Path) -> pd.DataFrame:
        normalized = frame.copy()
        created_at = dt.datetime.fromtimestamp(source_path.stat().st_mtime, tz=dt.UTC)
        for column in TABLE_KEYS.get(target_table, []):
            if column not in normalized.columns:
                if target_table == "bronze_forecast_requests" and column == "request_kind":
                    normalized[column] = "full"
                else:
                    normalized[column] = None
        if "run_id" not in normalized.columns:
            normalized["run_id"] = f"legacy::{source_path.stem}"
        if "data_version" not in normalized.columns:
            normalized["data_version"] = "legacy-v0"
        if "created_at" not in normalized.columns:
            normalized["created_at"] = pd.Timestamp(created_at)
        if "source_priority" not in normalized.columns:
            normalized["source_priority"] = LEGACY_PRECEDENCE.get(source_path.name, 10)
        if target_table == "gold_training_examples_tabular":
            if "forecast_source_kind" not in normalized.columns:
                normalized["forecast_source_kind"] = "legacy"
            if "available_models_json" not in normalized.columns:
                normalized["available_models_json"] = "[]"
            if "selected_models_json" not in normalized.columns:
                normalized["selected_models_json"] = normalized["available_models_json"]
        return normalized

    def _write_standard_gold_exports(self) -> None:
        if self.duckdb_store.table_exists("gold_training_examples_tabular"):
            self.duckdb_store.export_query(
                self._sanitized_gold_export_query("gold_training_examples_tabular"),
                self.parquet_store.root / "gold/historical_training_set.parquet",
            )
        if self.duckdb_store.table_exists("gold_training_examples_sequence"):
            self.duckdb_store.export_table(
                "gold_training_examples_sequence",
                self.parquet_store.root / "gold/historical_training_sequence.parquet",
            )

    def _sanitized_gold_export_query(self, table_name: str) -> str:
        select_fragments: list[str] = []
        for column_name, data_type in self.duckdb_store.table_schema(table_name):
            quoted_name = _quote_identifier(column_name)
            if data_type.upper().startswith(NUMERIC_DUCKDB_TYPE_PREFIXES):
                select_fragments.append(f"coalesce({quoted_name}, 0) as {quoted_name}")
            else:
                select_fragments.append(quoted_name)
        return f"select {', '.join(select_fragments)} from {table_name}"

    def _resolve_legacy_table_name(self, connection: duckdb.DuckDBPyConnection, table_name: str) -> str | None:
        if table_name in TABLE_KEYS:
            return table_name
        frame = connection.execute(f"select * from {table_name} limit 1").fetch_df()  # noqa: S608
        if LEGACY_GOLD_COLUMNS.issubset(set(frame.columns)):
            return "gold_training_examples_tabular"
        return None
