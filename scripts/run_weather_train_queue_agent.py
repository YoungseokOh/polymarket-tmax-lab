#!/usr/bin/env python3
"""Run the weather_train queue agent until throttling appears."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from pmtmax.weather.queue_agent import run_weather_train_queue_agent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-path", type=Path, default=Path("checker/weather_train_status.md"))
    parser.add_argument("--collection-log-path", type=Path, default=Path("checker/weather_train_collection_log.md"))
    parser.add_argument("--queue-anchor-date", type=date.fromisoformat, default=date(2024, 6, 1))
    parser.add_argument("--queue-start", type=date.fromisoformat, default=None)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--model", default="gfs_seamless")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rate-limit-profile", default="free")
    parser.add_argument("--http-timeout-seconds", type=int, default=15)
    parser.add_argument("--http-retries", type=int, default=1)
    parser.add_argument("--http-retry-wait-min-seconds", type=float, default=1.0)
    parser.add_argument("--http-retry-wait-max-seconds", type=float, default=8.0)
    parser.add_argument("--pretrain-refresh-threshold-rows", type=int, default=500)
    parser.add_argument("--pretrain-model-name", default="gaussian_emos")
    parser.add_argument("--pretrain-variant", default=None)
    parser.add_argument("--missing-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_weather_train_queue_agent(
        status_path=args.status_path,
        collection_log_path=args.collection_log_path,
        queue_anchor_date=args.queue_anchor_date,
        queue_start_override=args.queue_start,
        chunk_days=args.chunk_days,
        max_chunks=args.max_chunks,
        sleep_seconds=args.sleep_seconds,
        model=args.model,
        missing_only=bool(args.missing_only),
        workers=args.workers,
        rate_limit_profile=args.rate_limit_profile,
        http_timeout_seconds=args.http_timeout_seconds,
        http_retries=args.http_retries,
        http_retry_wait_min_seconds=args.http_retry_wait_min_seconds,
        http_retry_wait_max_seconds=args.http_retry_wait_max_seconds,
        progress=bool(args.progress),
        pretrain_refresh_threshold_rows=args.pretrain_refresh_threshold_rows,
        pretrain_model_name=args.pretrain_model_name,
        pretrain_variant=args.pretrain_variant,
    )
    print(json.dumps(summary.__dict__, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
