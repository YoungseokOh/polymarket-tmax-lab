"""Parquet storage helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetStore:
    """Simple Parquet read/write helper."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_frame(self, relative_path: str, frame: pd.DataFrame) -> Path:
        """Write DataFrame to a Parquet file under the store root."""

        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return path

    def read_frame(self, relative_path: str) -> pd.DataFrame:
        """Read DataFrame from a Parquet file under the store root."""

        return pd.read_parquet(self.root / relative_path)

