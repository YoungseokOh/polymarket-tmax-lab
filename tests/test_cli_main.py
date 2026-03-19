from __future__ import annotations

from pathlib import Path

import pytest

from pmtmax.cli.main import _bootstrap_snapshots, _load_snapshots


def test_load_snapshots_raises_for_missing_markets_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing_snapshots.json"

    with pytest.raises(FileNotFoundError, match="Market snapshot file does not exist"):
        _load_snapshots(markets_path=missing)


def test_bootstrap_snapshots_raises_for_missing_markets_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing_snapshots.json"

    with pytest.raises(FileNotFoundError, match="Market snapshot file does not exist"):
        _bootstrap_snapshots(markets_path=missing, cities=None)
