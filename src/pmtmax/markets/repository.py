"""Helpers for bundled and scanned market snapshots."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pmtmax.examples import EXAMPLE_MARKETS
from pmtmax.markets.normalization import (
    extract_clob_token_ids,
    extract_outcome_prices,
)
from pmtmax.markets.rule_parser import parse_market_spec
from pmtmax.storage.schemas import MarketSnapshot
from pmtmax.utils import dump_json, load_json


SHARDED_MARKET_SNAPSHOTS_FORMAT = "pmtmax.market_snapshots.sharded.v1"


def snapshot_from_market(market: dict[str, Any], *, captured_at: datetime | None = None) -> MarketSnapshot:
    """Build a parsed snapshot from a raw market payload."""

    captured = captured_at or datetime.now(tz=UTC)
    outcome_prices = extract_outcome_prices(market)
    clob_token_ids = extract_clob_token_ids(market)
    try:
        spec = parse_market_spec(market.get("description", ""), market=market)
        return MarketSnapshot(
            captured_at=captured,
            market=market,
            spec=spec,
            outcome_prices=outcome_prices,
            clob_token_ids=clob_token_ids,
        )
    except Exception as exc:  # noqa: BLE001
        return MarketSnapshot(
            captured_at=captured,
            market=market,
            spec=None,
            parse_error=str(exc),
            outcome_prices=outcome_prices,
            clob_token_ids=clob_token_ids,
        )


def bundled_market_snapshots(cities: list[str] | None = None) -> list[MarketSnapshot]:
    """Return reproducible historical market snapshots for bundled city templates."""

    selected = cities or list(EXAMPLE_MARKETS)
    captured_at = datetime(2026, 1, 1, tzinfo=UTC)
    return [snapshot_from_market(EXAMPLE_MARKETS[city], captured_at=captured_at) for city in selected]


def save_market_snapshots(path: Path, snapshots: list[MarketSnapshot]) -> Path:
    """Serialize market snapshots to disk."""

    dump_json(path, [snapshot.model_dump(mode="json") for snapshot in snapshots])
    return path


def _is_sharded_market_snapshots(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("format") == SHARDED_MARKET_SNAPSHOTS_FORMAT


def _shard_path(manifest_path: Path, shard: dict[str, Any]) -> Path:
    shard_ref = shard.get("path")
    if not isinstance(shard_ref, str) or not shard_ref:
        msg = f"Invalid market snapshot shard entry in {manifest_path}: {shard!r}"
        raise ValueError(msg)
    return (manifest_path.parent / shard_ref).resolve()


def _starts_with_json_object(path: Path) -> bool:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4096), b""):
            stripped = chunk.lstrip()
            if stripped:
                return stripped.startswith(b"{")
    return False


def _count_marker(path: Path, marker: bytes) -> int:
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(marker)
    return count


def _contains_marker(path: Path, marker: bytes) -> bool:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if marker in chunk:
                return True
    return False


def load_market_snapshot_payloads(path: Path) -> list[dict[str, Any]]:
    """Load market snapshot payload dictionaries from a flat or sharded inventory."""

    payload = load_json(path)
    if isinstance(payload, list):
        return payload
    if not _is_sharded_market_snapshots(payload):
        msg = f"Unsupported market snapshot inventory format: {path}"
        raise ValueError(msg)

    rows: list[dict[str, Any]] = []
    shards = payload.get("shards")
    if not isinstance(shards, list):
        msg = f"Sharded market snapshot inventory is missing shards: {path}"
        raise ValueError(msg)
    for shard in shards:
        if not isinstance(shard, dict):
            msg = f"Invalid market snapshot shard entry in {path}: {shard!r}"
            raise ValueError(msg)
        shard_payload = load_json(_shard_path(path, shard))
        if not isinstance(shard_payload, list):
            msg = f"Market snapshot shard must contain a JSON list: {_shard_path(path, shard)}"
            raise ValueError(msg)
        rows.extend(shard_payload)

    expected_count = payload.get("snapshot_count")
    if isinstance(expected_count, int) and expected_count != len(rows):
        msg = f"Sharded inventory {path} expected {expected_count} snapshots, loaded {len(rows)}"
        raise ValueError(msg)
    return rows


def count_market_snapshot_payloads(path: Path) -> int | None:
    """Count flat or sharded snapshot records without loading every large shard."""

    if not path.exists():
        return None
    if _starts_with_json_object(path):
        payload = load_json(path)
        if _is_sharded_market_snapshots(payload):
            snapshot_count = payload.get("snapshot_count")
            if isinstance(snapshot_count, int):
                return snapshot_count
            shards = payload.get("shards")
            if isinstance(shards, list):
                return sum(int(shard.get("snapshot_count", 0)) for shard in shards if isinstance(shard, dict))
        return None
    return _count_marker(path, b'"captured_at"')


def market_snapshot_inventory_contains(path: Path, marker: bytes) -> bool:
    """Return whether a flat or sharded market inventory contains a byte marker."""

    if not path.exists():
        return False
    if _contains_marker(path, marker):
        return True
    if not _starts_with_json_object(path):
        return False
    payload = load_json(path)
    if not _is_sharded_market_snapshots(payload):
        return False
    shards = payload.get("shards")
    if not isinstance(shards, list):
        return False
    return any(
        _contains_marker(_shard_path(path, shard), marker)
        for shard in shards
        if isinstance(shard, dict)
    )


def load_market_snapshots(path: Path) -> list[MarketSnapshot]:
    """Load serialized market snapshots from disk."""

    payload = load_market_snapshot_payloads(path)
    return [MarketSnapshot.model_validate(item) for item in payload]
