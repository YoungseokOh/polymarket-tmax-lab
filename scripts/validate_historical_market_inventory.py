"""Validate a curated historical market snapshot inventory."""

from __future__ import annotations

import argparse
from pathlib import Path

from pmtmax.config.settings import load_settings
from pmtmax.http import CachedHttpClient
from pmtmax.markets.inventory import probe_truth_readiness, validate_historical_inventory
from pmtmax.markets.repository import load_market_snapshots
from pmtmax.utils import dump_json

DEFAULT_INPUT = Path("configs/market_inventory/historical_temperature_snapshots.json")
DEFAULT_REPORT = Path("data/manifests/historical_inventory_validate_report.json")
DEFAULT_TRUTH_WORKERS = 4
DEFAULT_TRUTH_PER_SOURCE_LIMIT = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Curated MarketSnapshot[] JSON to validate.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Where to write the validation report JSON.",
    )
    parser.add_argument(
        "--truth-workers",
        type=int,
        default=DEFAULT_TRUTH_WORKERS,
        help="Bounded concurrency for truth-readiness probes during validation.",
    )
    parser.add_argument(
        "--truth-per-source-limit",
        type=int,
        default=DEFAULT_TRUTH_PER_SOURCE_LIMIT,
        help="Maximum concurrent truth probes per official source family.",
    )
    parser.add_argument(
        "--truth-no-cache",
        action="store_true",
        help="Disable cache reads while probing official truth readiness.",
    )
    args = parser.parse_args()

    config, _ = load_settings()
    snapshots = load_market_snapshots(args.input)
    http = CachedHttpClient(config.app.cache_dir, config.weather.timeout_seconds, config.weather.retries)
    try:
        report = validate_historical_inventory(
            snapshots,
            supported_cities=config.app.supported_cities,
            source_manifest=str(args.input),
            truth_probe=lambda snapshot: probe_truth_readiness(
                snapshot,
                http,
                snapshot_dir=config.app.raw_dir / "bronze",
                use_cache=not args.truth_no_cache,
            ),
            truth_workers=args.truth_workers,
            truth_per_source_limit=args.truth_per_source_limit,
        )
    finally:
        http.close()
    dump_json(args.report, report.model_dump(mode="json"))
    print(
        f"Validated {len(snapshots)} curated snapshots -> {args.report} "
        f"(issues: {len(report.issues)}, issue_counts: {report.issue_counts})"
    )


if __name__ == "__main__":
    main()
