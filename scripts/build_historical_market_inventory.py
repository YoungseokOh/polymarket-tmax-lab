"""Build a curated historical market inventory from Polymarket event pages."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from pmtmax.config.settings import load_settings
from pmtmax.http import CachedHttpClient
from pmtmax.markets.inventory import (
    HistoricalEventPage,
    build_historical_inventory_from_pages,
    preserve_existing_capture_times,
)
from pmtmax.markets.repository import load_market_snapshots, save_market_snapshots
from pmtmax.utils import dump_json, load_json

DEFAULT_INPUT = Path("configs/market_inventory/historical_temperature_event_urls.json")
DEFAULT_OUTPUT = Path("configs/market_inventory/historical_temperature_snapshots.json")
DEFAULT_REPORT = Path("data/manifests/historical_inventory_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="JSON array of Polymarket event URLs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the aggregated MarketSnapshot[] JSON.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Where to write the inventory build report JSON.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the shared HTTP cache while fetching event pages.",
    )
    args = parser.parse_args()

    config, _ = load_settings()
    urls = [str(item).strip() for item in load_json(args.input) if str(item).strip()]
    http = CachedHttpClient(config.app.cache_dir, config.weather.timeout_seconds, config.weather.retries)
    try:
        pages = [
            HistoricalEventPage(
                url=url,
                html=http.get_text(url, use_cache=not args.no_cache),
                fetched_at=datetime.now(tz=UTC),
            )
            for url in urls
        ]
    finally:
        http.close()

    snapshots, report = build_historical_inventory_from_pages(
        pages,
        supported_cities=config.app.supported_cities,
        source_manifest=str(args.input),
    )
    existing_snapshot_paths: list[Path] = []
    if args.output.exists():
        existing_snapshot_paths.append(args.output)
    elif args.output != DEFAULT_OUTPUT and DEFAULT_OUTPUT.exists():
        existing_snapshot_paths.append(DEFAULT_OUTPUT)
    existing_snapshots = []
    for path in existing_snapshot_paths:
        existing_snapshots.extend(load_market_snapshots(path))
    if existing_snapshots:
        snapshots = preserve_existing_capture_times(
            snapshots,
            existing_snapshots=existing_snapshots,
        )
    save_market_snapshots(args.output, snapshots)
    dump_json(args.report, report.model_dump(mode="json"))
    print(
        f"Built {len(snapshots)} curated snapshots from {len(urls)} URLs -> {args.output} "
        f"(issues: {len(report.issues)})"
    )


if __name__ == "__main__":
    main()
