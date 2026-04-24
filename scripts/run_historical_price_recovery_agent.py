#!/usr/bin/env python3
"""Run one or more daily historical_real price-recovery shards and update checker state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pmtmax.backfill.price_recovery_agent import run_historical_price_recovery_agent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-path", type=Path, default=Path("checker/historical_price_status.md"))
    parser.add_argument(
        "--collection-log-path",
        type=Path,
        default=Path("checker/historical_price_collection_log.md"),
    )
    parser.add_argument(
        "--markets-path",
        type=Path,
        default=Path("configs/market_inventory/full_training_set_snapshots.json"),
    )
    parser.add_argument(
        "--coverage-output-path",
        type=Path,
        default=Path("artifacts/workspaces/historical_real/coverage/latest_price_history_coverage.json"),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/workspaces/historical_real/parquet/gold/historical_training_set.parquet"),
    )
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--shard-start", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--interval", default="max")
    parser.add_argument("--fidelity", type=int, default=60)
    parser.add_argument("--only-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--price-no-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-name", default="historical_backtest_panel")
    parser.add_argument("--max-price-age-minutes", type=int, default=720)
    parser.add_argument("--allow-canonical-overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_historical_price_recovery_agent(
        status_path=args.status_path,
        collection_log_path=args.collection_log_path,
        markets_path=args.markets_path,
        coverage_output_path=args.coverage_output_path,
        dataset_path=args.dataset_path,
        shard_size=args.shard_size,
        shard_start_override=args.shard_start,
        max_shards=args.max_shards,
        sleep_seconds=args.sleep_seconds,
        interval=args.interval,
        fidelity=args.fidelity,
        only_missing=bool(args.only_missing),
        price_no_cache=bool(args.price_no_cache),
        output_name=args.output_name,
        max_price_age_minutes=args.max_price_age_minutes,
        allow_canonical_overwrite=bool(args.allow_canonical_overwrite),
    )
    print(json.dumps(summary.__dict__, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
