"""Run the staged, resumable historical closed-event refresh pipeline."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from pmtmax.config.settings import load_settings
from pmtmax.http import CachedHttpClient
from pmtmax.markets.gamma_client import GammaClient
from pmtmax.markets.inventory import (
    NON_TERMINAL_COLLECTION_STATUSES,
    TERMINAL_COLLECTION_STATUSES,
    HistoricalCollectionStatus,
    HistoricalCollectionStatusReport,
    HistoricalEventCandidateReport,
    HistoricalEventPageFetchReport,
    build_fetches_from_report,
    build_historical_collection_status,
    collection_status_matches_filter,
    discover_temperature_event_refs_from_gamma,
    fetch_historical_event_page_report,
    filter_historical_event_candidates,
    is_retryable_collection_entry,
    merge_historical_collection_status_reports,
    merge_historical_event_candidates,
    probe_truth_readiness,
)
from pmtmax.utils import dump_json, load_json

DEFAULT_OUTPUT = Path("configs/market_inventory/historical_temperature_event_urls.json")
DEFAULT_CANDIDATES = Path("data/manifests/historical_event_candidates.json")
DEFAULT_FETCH_REPORT = Path("data/manifests/historical_event_page_fetches.json")
DEFAULT_REPORT = Path("data/manifests/historical_collection_status.json")
DEFAULT_FETCH_WORKERS = 8
DEFAULT_TRUTH_WORKERS = 4
DEFAULT_TRUTH_PER_SOURCE_LIMIT = 1
DEFAULT_CHECKPOINT_EVERY = 25
STAGE_CHOICES = ("discover", "fetch-pages", "classify", "publish", "all")

def _load_model[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT | None:
    if not path.exists():
        return None
    return model_type.model_validate(load_json(path))


def _require_model[ModelT: BaseModel](path: Path, model_type: type[ModelT], *, help_text: str) -> ModelT:
    model = _load_model(path, model_type)
    if model is None:
        msg = f"Missing required manifest: {path}. {help_text}"
        raise FileNotFoundError(msg)
    return model


def _selected_fetch_report(
    fetch_report: HistoricalEventPageFetchReport,
    *,
    status_report: HistoricalCollectionStatusReport | None,
    supported_cities: list[str],
    resume: bool,
    status_filters: set[HistoricalCollectionStatus],
    max_events: int | None,
    already_selected_urls: set[str] | None = None,
) -> HistoricalEventPageFetchReport:
    supported = {city.lower() for city in supported_cities}
    existing_by_url = {entry.url: entry for entry in (status_report.entries if status_report is not None else [])}
    processed_urls = already_selected_urls or set()
    selected_entries = []
    for entry in fetch_report.entries:
        if entry.fetch_status != "fetched":
            continue
        if entry.city.lower() not in supported:
            continue
        if entry.url in processed_urls:
            continue
        existing = existing_by_url.get(entry.url)
        if status_filters:
            if existing is None or not collection_status_matches_filter(existing, status_filters):
                continue
        elif resume and existing is not None and not is_retryable_collection_entry(existing):
            continue
        selected_entries.append(entry)

    if max_events is not None:
        selected_entries = selected_entries[:max_events]

    return HistoricalEventPageFetchReport(
        generated_at=fetch_report.generated_at,
        supported_cities=supported_cities,
        total_candidates=fetch_report.total_candidates,
        processed_this_run=len(selected_entries),
        status_counts=fetch_report.status_counts,
        entries=selected_entries,
    )


def _existing_collected_market_ids(
    status_report: HistoricalCollectionStatusReport | None,
    *,
    selected_urls: set[str],
) -> set[str]:
    if status_report is None:
        return set()
    return {
        entry.market_id
        for entry in status_report.entries
        if entry.status == "collected" and entry.market_id and entry.url not in selected_urls
    }


def _publish_collected_urls(
    *,
    status_report: HistoricalCollectionStatusReport,
    output_path: Path,
) -> tuple[int, int]:
    existing_urls = [str(item).strip() for item in load_json(output_path)] if output_path.exists() else []
    manifest_urls = list(existing_urls)
    for url in [entry.url for entry in status_report.entries if entry.status == "collected"]:
        if url not in manifest_urls:
            manifest_urls.append(url)
    dump_json(output_path, manifest_urls)
    return len(manifest_urls), len(manifest_urls) - len(existing_urls)


def _with_processed_this_run[ModelT: BaseModel](report: ModelT, processed_this_run: int) -> ModelT:
    return report.model_copy(update={"processed_this_run": processed_this_run})


def _run_fetch_stage(
    *,
    http: CachedHttpClient,
    candidate_report: HistoricalEventCandidateReport,
    fetch_report: HistoricalEventPageFetchReport | None,
    supported_cities: list[str],
    use_cache: bool,
    resume: bool,
    max_workers: int,
    max_events: int | None,
    checkpoint_every: int,
    fetch_report_path: Path,
) -> HistoricalEventPageFetchReport:
    remaining = max_events
    total_processed = 0
    current_report = fetch_report
    while True:
        batch_limit = checkpoint_every if remaining is None else min(checkpoint_every, remaining)
        if batch_limit <= 0:
            break
        current_report = fetch_historical_event_page_report(
            http,
            filter_historical_event_candidates(candidate_report, supported_cities=supported_cities),
            existing_report=current_report,
            use_cache=use_cache,
            resume=resume,
            max_workers=max_workers,
            max_events=batch_limit,
        )
        processed = current_report.processed_this_run
        total_processed += processed
        dump_json(fetch_report_path, _with_processed_this_run(current_report, total_processed).model_dump(mode="json"))
        if processed == 0:
            break
        if remaining is not None:
            remaining -= processed
            if remaining <= 0:
                break
    if current_report is None:
        current_report = HistoricalEventPageFetchReport(
            generated_at=datetime.now(tz=UTC),
            supported_cities=supported_cities,
            total_candidates=0,
            processed_this_run=0,
            status_counts={},
            entries=[],
        )
    return _with_processed_this_run(current_report, total_processed)


def _run_classify_stage(
    *,
    http: CachedHttpClient,
    fetch_report: HistoricalEventPageFetchReport,
    status_report: HistoricalCollectionStatusReport | None,
    supported_cities: list[str],
    resume: bool,
    status_filters: set[HistoricalCollectionStatus],
    max_events: int | None,
    checkpoint_every: int,
    output_path: Path,
    report_path: Path,
    truth_workers: int,
    truth_per_source_limit: int,
    truth_snapshot_dir: Path | None,
    use_truth_cache: bool,
) -> HistoricalCollectionStatusReport:
    remaining = max_events
    total_processed = 0
    current_report = status_report
    processed_urls: set[str] = set()
    while True:
        batch_limit = checkpoint_every if remaining is None else min(checkpoint_every, remaining)
        if batch_limit <= 0:
            break
        selected_fetch_report = _selected_fetch_report(
            fetch_report,
            status_report=current_report,
            supported_cities=supported_cities,
            resume=resume,
            status_filters=status_filters,
            max_events=batch_limit,
            already_selected_urls=processed_urls,
        )
        if selected_fetch_report.processed_this_run == 0:
            break
        selected_urls = {entry.url for entry in selected_fetch_report.entries}
        processed_urls.update(selected_urls)
        existing_status_by_url = {entry.url: entry for entry in (current_report.entries if current_report is not None else [])}
        attempted_at = datetime.now(tz=UTC)
        partial_fetches = build_fetches_from_report(http, selected_fetch_report, supported_cities=supported_cities)
        _, partial_status_report = build_historical_collection_status(
            partial_fetches,
            supported_cities=supported_cities,
            truth_probe=lambda snapshot: probe_truth_readiness(
                snapshot,
                http,
                snapshot_dir=truth_snapshot_dir,
                use_cache=use_truth_cache,
            ),
            source_manifest=str(output_path),
            existing_market_ids=_existing_collected_market_ids(current_report, selected_urls=selected_urls),
            attempt_counts={
                entry.url: (existing_status_by_url[entry.url].attempt_count if entry.url in existing_status_by_url else 0) + 1
                for entry in selected_fetch_report.entries
            },
            discovered_at_by_url={entry.url: entry.discovered_at for entry in selected_fetch_report.entries},
            attempted_at_by_url={entry.url: attempted_at for entry in selected_fetch_report.entries},
            truth_workers=truth_workers,
            truth_per_source_limit=truth_per_source_limit,
        )
        current_report = merge_historical_collection_status_reports(
            existing_report=current_report,
            updated_report=partial_status_report,
            total_discovered=fetch_report.total_candidates,
            supported_cities=supported_cities,
            source_manifest=str(output_path),
        )
        total_processed += selected_fetch_report.processed_this_run
        dump_json(report_path, _with_processed_this_run(current_report, total_processed).model_dump(mode="json"))
        if remaining is not None:
            remaining -= selected_fetch_report.processed_this_run
            if remaining <= 0:
                break
    if current_report is None:
        current_report = HistoricalCollectionStatusReport(
            generated_at=datetime.now(tz=UTC),
            source_manifest=str(output_path),
            supported_cities=supported_cities,
            total_discovered=fetch_report.total_candidates,
            processed_this_run=0,
            collected_urls=[],
            status_counts={},
            entries=[],
        )
    return _with_processed_this_run(current_report, total_processed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=STAGE_CHOICES,
        default="all",
        help="Which stage to execute. 'all' runs discover -> fetch-pages -> classify -> publish.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the append-only JSON array of collected event URLs.",
    )
    parser.add_argument(
        "--candidates-path",
        type=Path,
        default=DEFAULT_CANDIDATES,
        help="Where to persist the grouped closed-event backlog manifest.",
    )
    parser.add_argument(
        "--fetch-report",
        type=Path,
        default=DEFAULT_FETCH_REPORT,
        help="Where to persist grouped event page fetch state.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Where to write the closed-event collection status report JSON.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of Gamma event pages to scan during discovery. Defaults to config.polymarket.max_pages.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum number of candidate or fetched events to process in the current stage.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Persist fetch/classify progress after each batch of this many events.",
    )
    parser.add_argument(
        "--city",
        action="append",
        default=None,
        help="Limit discovery and processing to specific supported cities. Can be provided multiple times.",
    )
    parser.add_argument(
        "--fetch-workers",
        type=int,
        default=DEFAULT_FETCH_WORKERS,
        help="Bounded concurrency for Polymarket event page fetches.",
    )
    parser.add_argument(
        "--truth-workers",
        type=int,
        default=DEFAULT_TRUTH_WORKERS,
        help="Bounded concurrency for exact-source truth readiness probes.",
    )
    parser.add_argument(
        "--truth-per-source-limit",
        type=int,
        default=DEFAULT_TRUTH_PER_SOURCE_LIMIT,
        help="Maximum concurrent truth probes per official source family.",
    )
    parser.add_argument(
        "--status-filter",
        action="append",
        choices=sorted(NON_TERMINAL_COLLECTION_STATUSES | TERMINAL_COLLECTION_STATUSES),
        default=None,
        help="When classifying, only reprocess URLs whose existing status matches one of these values.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing candidate/fetch/status manifests for the selected stage.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache reads while fetching new candidate pages.",
    )
    parser.add_argument(
        "--truth-no-cache",
        action="store_true",
        help="Disable cache reads while probing official truth readiness.",
    )
    parser.add_argument(
        "--skip-discover",
        action="store_true",
        help="When running --stage all, reuse the existing candidate manifest and only fill fetch/classify/publish gaps.",
    )
    args = parser.parse_args()

    config, _ = load_settings()
    supported_cities = args.city or config.app.supported_cities
    resume = not args.no_resume
    status_filters = set(args.status_filter or [])

    candidate_report = _load_model(args.candidates_path, HistoricalEventCandidateReport) if resume else None
    fetch_report = _load_model(args.fetch_report, HistoricalEventPageFetchReport) if resume else None
    status_report = _load_model(args.report, HistoricalCollectionStatusReport) if resume else None

    http = CachedHttpClient(config.app.cache_dir, config.weather.timeout_seconds, config.weather.retries)
    try:
        checkpoint_every = max(1, args.checkpoint_every)

        if args.stage in {"discover", "all"} and not args.skip_discover:
            gamma = GammaClient(http, config.polymarket.gamma_base_url)
            refs = discover_temperature_event_refs_from_gamma(
                gamma,
                supported_cities=supported_cities,
                closed=True,
                max_pages=args.max_pages or config.polymarket.max_pages,
            )
            candidate_report = merge_historical_event_candidates(
                refs,
                supported_cities=supported_cities,
                existing_report=candidate_report if resume else None,
            )
            dump_json(args.candidates_path, candidate_report.model_dump(mode="json"))
            print(
                f"[discover] scanned {len(refs)} supported closed events -> "
                f"{candidate_report.candidate_count} candidates persisted at {args.candidates_path}"
            )
        elif args.stage in {"fetch-pages", "classify", "publish", "all"}:
            candidate_report = candidate_report or _require_model(
                args.candidates_path,
                HistoricalEventCandidateReport,
                help_text="Run --stage discover first or point --candidates-path at an existing manifest.",
            )

        if args.stage in {"fetch-pages", "all"}:
            fetch_report = _run_fetch_stage(
                http=http,
                candidate_report=candidate_report,
                fetch_report=fetch_report if resume else None,
                supported_cities=supported_cities,
                use_cache=not args.no_cache,
                resume=resume,
                max_workers=max(1, args.fetch_workers),
                max_events=args.max_events,
                checkpoint_every=checkpoint_every,
                fetch_report_path=args.fetch_report,
            )
            print(
                f"[fetch-pages] processed {fetch_report.processed_this_run} candidates -> "
                f"{fetch_report.status_counts.get('fetched', 0)} fetched, "
                f"{fetch_report.status_counts.get('fetch_failed', 0)} failed"
            )

        if args.stage in {"classify", "all"}:
            candidate_report = candidate_report or _require_model(
                args.candidates_path,
                HistoricalEventCandidateReport,
                help_text="Run --stage discover first or point --candidates-path at an existing manifest.",
            )
            fetch_report = fetch_report or _require_model(
                args.fetch_report,
                HistoricalEventPageFetchReport,
                help_text="Run --stage fetch-pages first or point --fetch-report at an existing manifest.",
            )
            status_report = _run_classify_stage(
                http=http,
                fetch_report=fetch_report,
                status_report=status_report if resume else None,
                supported_cities=supported_cities,
                resume=resume,
                status_filters=status_filters,
                max_events=args.max_events,
                checkpoint_every=checkpoint_every,
                output_path=args.output,
                report_path=args.report,
                truth_workers=max(1, args.truth_workers),
                truth_per_source_limit=max(1, args.truth_per_source_limit),
                truth_snapshot_dir=config.app.raw_dir / "bronze",
                use_truth_cache=not args.truth_no_cache,
            )
            print(
                f"[classify] processed {status_report.processed_this_run} fetched events -> "
                f"{status_report.status_counts.get('collected', 0)} collected total, "
                f"{status_report.status_counts.get('truth_source_lag', 0)} source-lag, "
                f"{status_report.status_counts.get('truth_parse_failed', 0)} parse-failed, "
                f"{status_report.status_counts.get('truth_request_failed', 0)} request-failed"
            )

        if args.stage in {"publish", "all"}:
            status_report = status_report or _require_model(
                args.report,
                HistoricalCollectionStatusReport,
                help_text="Run --stage classify first or point --report at an existing status report.",
            )
            manifest_count, appended = _publish_collected_urls(status_report=status_report, output_path=args.output)
            print(
                f"[publish] wrote {manifest_count} manifest URLs to {args.output} "
                f"(appended {appended})"
            )
    finally:
        http.close()


if __name__ == "__main__":
    main()
