"""Build a curated historical market inventory from Polymarket event pages."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from pmtmax.config.settings import load_settings
from pmtmax.http import CachedHttpClient
from pmtmax.markets.inventory import (
    HistoricalCollectionStatusReport,
    HistoricalEventCandidateReport,
    HistoricalEventPage,
    build_historical_inventory_from_pages,
    preserve_existing_capture_times,
    probe_truth_readiness,
    sync_historical_collection_status_report,
)
from pmtmax.markets.repository import load_market_snapshots, save_market_snapshots
from pmtmax.utils import dump_json, load_json

DEFAULT_INPUT = Path("configs/market_inventory/historical_temperature_event_urls.json")
DEFAULT_OUTPUT = Path("configs/market_inventory/historical_temperature_snapshots.json")
DEFAULT_REPORT = Path("data/manifests/historical_inventory_build_report.json")
DEFAULT_CANDIDATE_REPORT = Path("data/manifests/historical_event_candidates.json")
DEFAULT_STATUS_REPORT = Path("data/manifests/historical_collection_status.json")
DEFAULT_TRUTH_WORKERS = 4
DEFAULT_TRUTH_PER_SOURCE_LIMIT = 1


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
        "--candidate-report",
        type=Path,
        default=None,
        help="Optional candidate manifest used to sync the canonical collection status report.",
    )
    parser.add_argument(
        "--status-report",
        type=Path,
        default=None,
        help="Optional collection status report path to sync after rebuilding the curated inventory.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the shared HTTP cache while fetching event pages.",
    )
    parser.add_argument(
        "--truth-workers",
        type=int,
        default=DEFAULT_TRUTH_WORKERS,
        help="Bounded concurrency for truth-readiness probes while building the curated inventory.",
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
        snapshots, report = build_historical_inventory_from_pages(
            pages,
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

    default_canonical_build = args.input == DEFAULT_INPUT and args.output == DEFAULT_OUTPUT
    candidate_report_path = args.candidate_report or (DEFAULT_CANDIDATE_REPORT if default_canonical_build else None)
    status_report_path = args.status_report or (DEFAULT_STATUS_REPORT if default_canonical_build else None)
    synced_status_path: Path | None = None
    if status_report_path is not None:
        candidate_report = None
        if candidate_report_path is not None and candidate_report_path.exists():
            candidate_report = HistoricalEventCandidateReport.model_validate(load_json(candidate_report_path))
        existing_status_report = None
        if status_report_path.exists():
            existing_status_report = HistoricalCollectionStatusReport.model_validate(load_json(status_report_path))
        synced_status = sync_historical_collection_status_report(
            snapshots,
            supported_cities=config.app.supported_cities,
            source_manifest=str(args.input),
            candidate_report=candidate_report,
            existing_report=existing_status_report,
        )
        dump_json(status_report_path, synced_status.model_dump(mode="json"))
        synced_status_path = status_report_path
    print(
        f"Built {len(snapshots)} curated snapshots from {len(urls)} URLs -> {args.output} "
        f"(issues: {len(report.issues)}, issue_counts: {report.issue_counts}, "
        f"status_report: {synced_status_path or 'not-synced'})"
    )


if __name__ == "__main__":
    main()
