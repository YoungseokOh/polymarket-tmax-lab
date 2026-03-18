from __future__ import annotations

from pathlib import Path

from pmtmax.backtest.dataset_builder import DatasetBuilder
from pmtmax.http import CachedHttpClient
from pmtmax.markets.repository import bundled_market_snapshots
from pmtmax.storage.duckdb_store import DuckDBStore
from pmtmax.storage.parquet_store import ParquetStore


class _FixtureOpenMeteoClient:
    def historical_forecast(self, **_: object) -> dict:
        raise RuntimeError("force fixture fallback")


def test_dataset_builder_creates_rows_for_all_cities_and_horizons(tmp_path: Path) -> None:
    http = CachedHttpClient(tmp_path / "cache")
    builder = DatasetBuilder(
        http=http,
        openmeteo=_FixtureOpenMeteoClient(),  # type: ignore[arg-type]
        duckdb_store=DuckDBStore(tmp_path / "db" / "test.duckdb"),
        parquet_store=ParquetStore(tmp_path / "parquet"),
        snapshot_dir=Path("tests/fixtures/truth"),
        fixture_dir=Path("tests/fixtures/openmeteo"),
    )

    frame = builder.build(
        bundled_market_snapshots(),
        output_name="test_training_set",
        decision_horizons=["market_open", "morning_of"],
    )

    assert len(frame) == 8
    assert set(frame["city"]) == {"Seoul", "NYC", "Hong Kong", "Taipei"}
    assert {"winning_outcome", "market_spec_json", "market_prices_json"}.issubset(frame.columns)
