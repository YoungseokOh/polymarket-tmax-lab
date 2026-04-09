"""DuckDB-backed storage for datasets and metrics."""

from __future__ import annotations

import re
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


def _validate_identifier(identifier: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        msg = f"Unsafe identifier: {identifier}"
        raise ValueError(msg)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class DuckDBStore:
    """Simple DuckDB helper for persisting tabular artifacts."""

    _ARROW_EXPORT_BATCH_ROWS = 100_000

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(path))

    def write_frame(self, table_name: str, frame: pd.DataFrame) -> None:
        """Write a DataFrame to a DuckDB table, replacing prior contents."""

        _validate_identifier(table_name)
        self.connection.register("frame_view", frame)
        try:
            self.connection.execute(
                f"create or replace table {table_name} as select * from frame_view"  # noqa: S608
            )
        finally:
            with suppress(Exception):
                self.connection.unregister("frame_view")

    def table_exists(self, table_name: str) -> bool:
        """Return whether a table already exists."""

        _validate_identifier(table_name)
        query = """
            select count(*)
            from information_schema.tables
            where table_schema = current_schema()
              and table_name = ?
        """
        row = self.connection.execute(query, [table_name]).fetchone()
        return bool(row[0]) if row is not None else False

    def read_table(self, table_name: str) -> pd.DataFrame:
        """Read an entire table as a pandas DataFrame."""

        _validate_identifier(table_name)
        return self.connection.execute(f"select * from {table_name}").fetch_df()  # noqa: S608

    def row_count(self, table_name: str) -> int:
        """Return the row count for a table."""

        _validate_identifier(table_name)
        row = self.connection.execute(f"select count(*) from {table_name}").fetchone()  # noqa: S608
        return int(row[0]) if row is not None else 0

    def table_schema(self, table_name: str) -> list[tuple[str, str]]:
        """Return ordered column metadata for a table."""

        _validate_identifier(table_name)
        rows = self.connection.execute(
            f"select name, type from pragma_table_info('{table_name}') order by cid"  # noqa: S608
        ).fetchall()
        return [(str(name), str(data_type)) for name, data_type in rows]

    def column_names(self, table_name: str) -> list[str]:
        """Return ordered column names for a table."""

        return [name for name, _ in self.table_schema(table_name)]

    def export_query(self, query: str, path: Path) -> Path:
        """Export query results to Parquet in bounded Arrow batches."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.parent / f".{path.name}.tmp"
        reader = self.connection.execute(query).to_arrow_reader(self._ARROW_EXPORT_BATCH_ROWS)
        writer = None
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            for batch in reader:
                if writer is None:
                    writer = pq.ParquetWriter(temp_path, batch.schema)
                writer.write_batch(batch)
            if writer is None:
                writer = pq.ParquetWriter(temp_path, reader.schema)
                writer.write_table(pa.Table.from_batches([], schema=reader.schema))
        finally:
            if writer is not None:
                writer.close()
        shutil.move(str(temp_path), str(path))
        return path

    def export_table(self, table_name: str, path: Path) -> Path:
        """Export a table directly to Parquet via DuckDB."""

        _validate_identifier(table_name)
        return self.export_query(f"select * from {table_name}", path)

    def parquet_row_count(self, path: Path) -> int:
        """Return the row count for a Parquet file without loading it into pandas."""

        row = self.connection.execute("select count(*) from read_parquet(?)", [str(path)]).fetchone()
        return int(row[0]) if row is not None else 0

    def load_parquet(self, table_name: str, path: Path) -> int:
        """Replace a table with rows loaded directly from Parquet."""

        _validate_identifier(table_name)
        self.connection.execute(
            f"create or replace table {table_name} as select * from read_parquet(?)",  # noqa: S608
            [str(path)],
        )
        return self.row_count(table_name)

    def upsert_frame(self, table_name: str, frame: pd.DataFrame, subset: list[str]) -> int:
        """Append rows and keep the latest row for each primary-key subset in DuckDB."""

        if not subset:
            msg = "subset must not be empty"
            raise ValueError(msg)
        _validate_identifier(table_name)
        incoming = frame.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
        if incoming.empty:
            return self.row_count(table_name) if self.table_exists(table_name) else 0
        if not self.table_exists(table_name):
            self.write_frame(table_name, incoming)
            return len(incoming)

        stage_name = f"__stage_{table_name}"
        resolved_name = f"__resolved_{table_name}"
        self.connection.register("frame_view", incoming)
        try:
            self.connection.execute(
                f"create or replace temp table {stage_name} as select * from frame_view"  # noqa: S608
            )
        finally:
            with suppress(Exception):
                self.connection.unregister("frame_view")

        try:
            target_schema = self.table_schema(table_name)
            stage_schema = dict(self.table_schema(stage_name))
            target_columns = {name for name, _ in target_schema}
            for column_name, data_type in stage_schema.items():
                if column_name in target_columns:
                    continue
                self.connection.execute(
                    f"alter table {table_name} add column {_quote_identifier(column_name)} {data_type}"  # noqa: S608
                )

            self.connection.execute(
                
                    f"create or replace temp table {resolved_name} as "
                    f"select * from {table_name} union all by name select * from {stage_name} limit 0"
                  # noqa: S608
            )
            resolved_schema_map = dict(self.table_schema(resolved_name))
            target_schema_map = dict(self.table_schema(table_name))
            for column_name, resolved_type in resolved_schema_map.items():
                target_type = target_schema_map.get(column_name)
                if target_type is None or target_type == resolved_type:
                    continue
                with suppress(duckdb.Error):
                    self.connection.execute(
                        
                            f"alter table {table_name} alter column {_quote_identifier(column_name)} "
                            f"set data type {resolved_type}"
                          # noqa: S608
                    )

            target_schema = self.table_schema(table_name)
            select_fragments: list[str] = []
            for column_name, data_type in target_schema:
                quoted_name = _quote_identifier(column_name)
                if column_name in stage_schema:
                    select_fragments.append(f"cast(stage.{quoted_name} as {data_type}) as {quoted_name}")
                else:
                    select_fragments.append(f"cast(NULL as {data_type}) as {quoted_name}")

            join_conditions = " and ".join(
                [
                    (
                        f"target.{_quote_identifier(column_name)} is not distinct from "
                        f"cast(stage.{_quote_identifier(column_name)} as {dict(target_schema)[column_name]})"
                    )
                    for column_name in subset
                ]
            )
            self.connection.execute(
                f"delete from {table_name} as target using {stage_name} as stage where {join_conditions}"  # noqa: S608
            )
            self.connection.execute(
                
                    f"insert into {table_name} "
                    f"select {', '.join(select_fragments)} from {stage_name} as stage"
                  # noqa: S608
            )
        finally:
            self.connection.execute(f"drop table if exists {resolved_name}")  # noqa: S608
            self.connection.execute(f"drop table if exists {stage_name}")  # noqa: S608

        return self.row_count(table_name)

    def read_frame(self, query: str, parameters: list[Any] | None = None) -> pd.DataFrame:
        """Read query results as a pandas DataFrame."""

        return self.connection.execute(query, parameters or []).fetch_df()

    def list_tables(self) -> list[str]:
        """Return user tables in the current DuckDB schema."""

        rows = self.connection.execute("show tables").fetchall()
        return [str(row[0]) for row in rows]

    def execute(self, sql: str, parameters: list[Any] | None = None) -> None:
        """Execute arbitrary SQL."""

        self.connection.execute(sql, parameters or [])

    def close(self) -> None:
        """Close the underlying connection."""

        self.connection.close()
