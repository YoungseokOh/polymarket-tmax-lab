from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pmtmax.cli.main import _bootstrap_snapshots, _collection_preflight_report, _load_snapshots, summarize_truth_coverage
from pmtmax.config.settings import EnvSettings
from pmtmax.markets.repository import bundled_market_snapshots


def test_load_snapshots_raises_for_missing_markets_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing_snapshots.json"

    with pytest.raises(FileNotFoundError, match="Market snapshot file does not exist"):
        _load_snapshots(markets_path=missing)


def test_bootstrap_snapshots_raises_for_missing_markets_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing_snapshots.json"

    with pytest.raises(FileNotFoundError, match="Market snapshot file does not exist"):
        _bootstrap_snapshots(markets_path=missing, cities=None)


def test_collection_preflight_defaults_to_public_archive_for_wu_markets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PMTMAX_WU_API_KEY", raising=False)

    report = _collection_preflight_report(
        bundled_market_snapshots(["Seoul", "Hong Kong"]),
        EnvSettings(),
    )

    assert report["ready"] is True
    assert report["missing_env"] == []
    assert report["source_counts"] == {"hko": 1, "wunderground": 1}
    assert report["truth_track_counts"] == {"exact_public": 1, "research_public": 1}
    assert report["settlement_eligible_count"] == 1
    assert json.loads(json.dumps(report))["optional_env"] == ["PMTMAX_WU_API_KEY"]


def test_summarize_truth_coverage_command_writes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeWarehouse:
        def close(self) -> None:
            return None

    class _FakePipeline:
        warehouse = _FakeWarehouse()

        def summarize_truth_coverage(self) -> dict[str, pd.DataFrame]:
            return {
                "summary": pd.DataFrame(
                    [{"status": "lag", "truth_track": "research_public", "city": "Seoul", "count": 1}]
                ),
                "details": pd.DataFrame(
                    [{"city": "Seoul", "station_id": "RKSI", "status": "lag", "lag_days": 8}]
                ),
            }

    monkeypatch.setattr("pmtmax.cli.main._runtime", lambda include_stores=False: (None, None, None, None, None, None))
    monkeypatch.setattr("pmtmax.cli.main._backfill_pipeline", lambda config, http, openmeteo: _FakePipeline())

    output = tmp_path / "truth_coverage.json"
    summarize_truth_coverage(output)
    payload = json.loads(output.read_text())
    assert payload["summary"][0]["status"] == "lag"
    assert payload["details"][0]["station_id"] == "RKSI"
