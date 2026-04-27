from __future__ import annotations

from pathlib import Path

from pmtmax.markets.repository import (
    SHARDED_MARKET_SNAPSHOTS_FORMAT,
    count_market_snapshot_payloads,
    load_market_snapshot_payloads,
    load_market_snapshots,
    market_snapshot_inventory_contains,
)
from pmtmax.utils import dump_json


def test_load_market_snapshots_supports_sharded_manifest(tmp_path: Path) -> None:
    shard_dir = tmp_path / "inventory.d"
    shard_dir.mkdir()
    first = [{"captured_at": "2026-01-01T00:00:00Z", "market": {"id": "1"}}]
    second = [{"captured_at": "2026-01-02T00:00:00Z", "market": {"id": "2"}}]
    dump_json(shard_dir / "part-000.json", first)
    dump_json(shard_dir / "part-001.json", second)
    manifest = tmp_path / "full_training_set_snapshots.json"
    dump_json(
        manifest,
        {
            "format": SHARDED_MARKET_SNAPSHOTS_FORMAT,
            "snapshot_count": 2,
            "shards": [
                {"path": "inventory.d/part-000.json", "snapshot_count": 1},
                {"path": "inventory.d/part-001.json", "snapshot_count": 1},
            ],
        },
    )

    payloads = load_market_snapshot_payloads(manifest)
    snapshots = load_market_snapshots(manifest)

    assert [row["market"]["id"] for row in payloads] == ["1", "2"]
    assert [snapshot.market["id"] for snapshot in snapshots] == ["1", "2"]
    assert count_market_snapshot_payloads(manifest) == 2
    assert market_snapshot_inventory_contains(manifest, b'"captured_at"') is True
