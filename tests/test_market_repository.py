from __future__ import annotations

from pmtmax.markets.repository import bundled_market_snapshots


def test_bundled_market_snapshots_parse_all_supported_cities() -> None:
    snapshots = bundled_market_snapshots()

    assert len(snapshots) == 4
    assert all(snapshot.spec is not None for snapshot in snapshots)
    assert all(snapshot.outcome_prices for snapshot in snapshots)
    assert all(snapshot.clob_token_ids for snapshot in snapshots)
