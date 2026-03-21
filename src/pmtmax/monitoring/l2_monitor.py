"""L2 order-book time-series monitor for settlement-window analysis."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from pmtmax.logging_utils import get_logger
from pmtmax.markets.book_utils import fetch_book
from pmtmax.markets.clob_read_client import ClobReadClient
from pmtmax.storage.schemas import L2TimeSeriesRecord, MarketSnapshot

LOGGER = get_logger(__name__)


def _hours_to_settlement(target_local_date: date, now: datetime) -> float:
    """Estimate hours until target_local_date midnight UTC (conservative proxy)."""

    settlement_dt = datetime(
        target_local_date.year,
        target_local_date.month,
        target_local_date.day,
        23,
        59,
        59,
        tzinfo=UTC,
    )
    delta = settlement_dt - now
    return max(delta.total_seconds() / 3600.0, 0.0)


def collect_l2_snapshots(
    clob: ClobReadClient,
    snapshots: list[MarketSnapshot],
    *,
    settlement_window_hours: float = 48.0,
) -> list[L2TimeSeriesRecord]:
    """Collect L2 book data for all outcomes of markets within the settlement window."""

    now = datetime.now(tz=UTC)
    records: list[L2TimeSeriesRecord] = []

    for snapshot in snapshots:
        spec = snapshot.spec
        if spec is None:
            continue
        hours_left = _hours_to_settlement(spec.target_local_date, now)
        if hours_left > settlement_window_hours or hours_left <= 0:
            continue

        for outcome_label, token_id in zip(spec.outcome_labels(), spec.token_ids, strict=False):
            book = fetch_book(clob, snapshot, token_id, outcome_label)
            best_bid = book.best_bid()
            best_ask = book.best_ask()
            bid_depth = sum(level.size for level in book.bids)
            ask_depth = sum(level.size for level in book.asks)
            records.append(
                L2TimeSeriesRecord(
                    market_id=spec.market_id,
                    token_id=token_id,
                    outcome_label=outcome_label,
                    city=spec.city,
                    target_local_date=spec.target_local_date,
                    captured_at=now,
                    hours_to_settlement=round(hours_left, 2),
                    best_bid=best_bid,
                    best_ask=best_ask,
                    spread=round(best_ask - best_bid, 4),
                    bid_depth=bid_depth,
                    ask_depth=ask_depth,
                    num_bid_levels=len(book.bids),
                    num_ask_levels=len(book.asks),
                )
            )

    LOGGER.info("Collected %d L2 records from %d markets", len(records), len(snapshots))
    return records


def append_records_jsonl(records: list[L2TimeSeriesRecord], output_dir: Path) -> Path:
    """Append L2 records to a date-partitioned JSONL file."""

    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{today}.jsonl"
    with path.open("a") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    return path


def analyze_l2_timeseries(output_dir: Path) -> dict:
    """Analyze collected L2 data by hours_to_settlement buckets."""

    import json

    buckets = [48, 24, 12, 6, 3, 1]
    all_records: list[dict] = []

    if not output_dir.exists():
        return {"buckets": [], "error": "No L2 data directory found"}

    for jsonl_file in sorted(output_dir.glob("*.jsonl")):
        with jsonl_file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))

    if not all_records:
        return {"buckets": [], "error": "No L2 records found"}

    results = []
    for upper in buckets:
        lower = buckets[buckets.index(upper) + 1] if buckets.index(upper) < len(buckets) - 1 else 0
        bucket_records = [
            r for r in all_records if lower < r["hours_to_settlement"] <= upper
        ]
        if not bucket_records:
            results.append({
                "bucket": f"{lower}-{upper}h",
                "count": 0,
                "median_spread": None,
                "mean_bid_depth": None,
                "mean_ask_depth": None,
                "tradeable_pct": None,
            })
            continue

        spreads = sorted(r["spread"] for r in bucket_records)
        median_spread = spreads[len(spreads) // 2]
        mean_bid = sum(r["bid_depth"] for r in bucket_records) / len(bucket_records)
        mean_ask = sum(r["ask_depth"] for r in bucket_records) / len(bucket_records)
        tradeable = sum(1 for r in bucket_records if r["spread"] < 0.10) / len(bucket_records)

        results.append({
            "bucket": f"{lower}-{upper}h",
            "count": len(bucket_records),
            "median_spread": round(median_spread, 4),
            "mean_bid_depth": round(mean_bid, 2),
            "mean_ask_depth": round(mean_ask, 2),
            "tradeable_pct": round(tradeable * 100, 1),
        })

    return {
        "total_records": len(all_records),
        "buckets": results,
    }
