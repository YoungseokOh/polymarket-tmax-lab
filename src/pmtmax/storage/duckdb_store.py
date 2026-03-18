"""DuckDB-backed storage for datasets and metrics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


class DuckDBStore:
    """Simple DuckDB helper for persisting tabular artifacts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(path))

    def write_frame(self, table_name: str, frame: pd.DataFrame) -> None:
        """Write a DataFrame to a DuckDB table, replacing prior contents."""

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
            msg = f"Unsafe table name: {table_name}"
            raise ValueError(msg)
        self.connection.register("frame_view", frame)
        self.connection.execute(
            f"create or replace table {table_name} as select * from frame_view"  # noqa: S608
        )
        self.connection.unregister("frame_view")

    def table_exists(self, table_name: str) -> bool:
        """Return whether a table already exists."""

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

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
            msg = f"Unsafe table name: {table_name}"
            raise ValueError(msg)
        return self.connection.execute(f"select * from {table_name}").fetch_df()  # noqa: S608

    def upsert_frame(self, table_name: str, frame: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
        """Append rows and keep the latest row for each primary-key subset."""

        if not subset:
            msg = "subset must not be empty"
            raise ValueError(msg)
        existing = self.read_table(table_name) if self.table_exists(table_name) else pd.DataFrame(columns=frame.columns)
        combined = pd.concat([existing, frame], ignore_index=True, sort=False)
        deduped = combined.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
        self.write_frame(table_name, deduped)
        return deduped

    def read_frame(self, query: str) -> pd.DataFrame:
        """Read query results as a pandas DataFrame."""

        return self.connection.execute(query).fetch_df()

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
