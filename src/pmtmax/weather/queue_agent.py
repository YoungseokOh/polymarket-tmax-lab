"""Automate weather_train queue collection and checker updates."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd

from pmtmax.cli.main import _render_weather_training_progress, _trust_check_report
from pmtmax.config.settings import load_settings
from pmtmax.http import CachedHttpClient
from pmtmax.logging_utils import configure_logging
from pmtmax.modeling.train import require_supported_model_name, train_model
from pmtmax.utils import dump_json, set_global_seed
from pmtmax.weather.openmeteo_client import OpenMeteoClient
from pmtmax.weather.training_data import (
    WeatherTrainingCollectionResult,
    collect_weather_training_data,
    default_weather_training_paths,
)

SEOUL_TZ = ZoneInfo("Asia/Seoul")
DATE_RANGE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:\.\.(\d{4}-\d{2}-\d{2}))?")
NEXT_QUEUE_RE = re.compile(r"Continue older gap-fill from `(\d{4}-\d{2}-\d{2})` forward")


@dataclass(frozen=True)
class WeatherTrainSnapshot:
    total_rows: int
    station_count: int
    target_date_count: int
    max_target_date: date | None
    counts_by_date: dict[date, int]


@dataclass(frozen=True)
class CollectionLogEntry:
    run_date: str
    range_text: str
    mode: str
    outcome: str
    rows_added: int
    notes: str

    def to_markdown_row(self) -> str:
        return (
            f"| {self.run_date} | `{self.range_text}` | {self.mode} | "
            f"{self.outcome} | {self.rows_added:,} | {self.notes} |"
        )


@dataclass(frozen=True)
class QueueAgentSummary:
    queue_start: str
    queue_end: str
    chunk_days: int
    chunks_attempted: int
    rows_added: int
    stop_reason: str
    last_outcome: str
    dataset_rows: int
    dataset_dates: int
    max_target_date: str | None
    pretrain_refreshed: bool
    pretrain_rows: int


def load_weather_train_snapshot(gold_path: Path) -> WeatherTrainSnapshot:
    if not gold_path.exists():
        return WeatherTrainSnapshot(
            total_rows=0,
            station_count=0,
            target_date_count=0,
            max_target_date=None,
            counts_by_date={},
        )
    frame = pd.read_parquet(gold_path, columns=["station_id", "target_date"])
    if frame.empty:
        return WeatherTrainSnapshot(
            total_rows=0,
            station_count=0,
            target_date_count=0,
            max_target_date=None,
            counts_by_date={},
        )
    parsed_dates = pd.to_datetime(frame["target_date"], errors="coerce").dt.date
    counts = (
        pd.DataFrame({"target_date": parsed_dates})
        .dropna()
        .groupby("target_date")
        .size()
        .to_dict()
    )
    max_target_date = max(counts) if counts else None
    station_count = int(frame["station_id"].astype(str).nunique())
    return WeatherTrainSnapshot(
        total_rows=int(len(frame)),
        station_count=station_count,
        target_date_count=int(len(counts)),
        max_target_date=max_target_date,
        counts_by_date={target_date: int(count) for target_date, count in counts.items()},
    )


def parse_collection_log(path: Path) -> list[CollectionLogEntry]:
    if not path.exists():
        return []
    entries: list[CollectionLogEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        stripped = line.strip()
        if stripped.startswith("| Run Date ") or stripped.startswith("| --- "):
            continue
        parts = [part.strip() for part in stripped.split("|")[1:-1]]
        if len(parts) != 6:
            continue
        range_text = parts[1].strip("`")
        entries.append(
            CollectionLogEntry(
                run_date=parts[0],
                range_text=range_text,
                mode=parts[2],
                outcome=parts[3],
                rows_added=int(parts[4].replace(",", "") or 0),
                notes=parts[5],
            )
        )
    return entries


def append_collection_log_entry(path: Path, entry: CollectionLogEntry) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{entry.to_markdown_row()}\n")


def parse_dates_from_range_text(range_text: str) -> list[date]:
    dates: list[date] = []
    for start_text, end_text in DATE_RANGE_RE.findall(range_text):
        start_date = date.fromisoformat(start_text)
        end_date = date.fromisoformat(end_text or start_text)
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
    return dates


def format_date_range(start_date: date, end_date: date) -> str:
    if start_date == end_date:
        return start_date.isoformat()
    return f"{start_date.isoformat()}..{end_date.isoformat()}"


def format_grouped_ranges(dates: list[date]) -> list[tuple[date, date]]:
    if not dates:
        return []
    ordered = sorted(set(dates))
    groups: list[tuple[date, date]] = []
    start_date = ordered[0]
    end_date = ordered[0]
    for current in ordered[1:]:
        if current == end_date + timedelta(days=1):
            end_date = current
            continue
        groups.append((start_date, end_date))
        start_date = current
        end_date = current
    groups.append((start_date, end_date))
    return groups


def format_grouped_counts(counts_by_date: dict[date, int]) -> list[tuple[date, date, int]]:
    if not counts_by_date:
        return []
    ordered = sorted(counts_by_date.items())
    groups: list[tuple[date, date, int]] = []
    start_date, previous_date = ordered[0][0], ordered[0][0]
    count = ordered[0][1]
    for current_date, current_count in ordered[1:]:
        if current_count == count and current_date == previous_date + timedelta(days=1):
            previous_date = current_date
            continue
        groups.append((start_date, previous_date, count))
        start_date = current_date
        previous_date = current_date
        count = current_count
    groups.append((start_date, previous_date, count))
    return groups


def infer_next_queue_start(
    *,
    status_path: Path,
    snapshot: WeatherTrainSnapshot,
    anchor_date: date,
) -> date:
    if status_path.exists():
        match = NEXT_QUEUE_RE.search(status_path.read_text(encoding="utf-8"))
        if match:
            return date.fromisoformat(match.group(1))
    current = anchor_date
    full_count = snapshot.station_count
    while snapshot.counts_by_date.get(current) == full_count:
        current += timedelta(days=1)
    return current


def classify_collection_outcome(
    result: WeatherTrainingCollectionResult,
) -> Literal["success", "partial", "retry-only", "rate-limit-cancelled", "interrupted"]:
    retryable = int(result.status_counts.get("retryable_error", 0) or 0)
    if result.early_stop_reason is not None:
        return "rate-limit-cancelled"
    if result.failed_rows == 0:
        return "success"
    if result.collected_rows > 0:
        return "partial"
    if result.attempted_rows > 0 and retryable == result.attempted_rows:
        return "retry-only"
    return "interrupted"


def build_collection_note(
    *,
    result: WeatherTrainingCollectionResult,
    outcome: str,
    next_queue_start: date,
) -> str:
    retryable = int(result.status_counts.get("retryable_error", 0) or 0)
    available = int(result.status_counts.get("available", 0) or 0)
    if outcome == "success":
        return (
            f"Full `{available}/{result.requested_rows} available`; next older-backfill queue is "
            f"`{next_queue_start.isoformat()}`."
        )
    if outcome == "partial":
        return (
            f"`{available}/{result.requested_rows}` available; "
            f"`{retryable}` retryable_error. Queue stops on the first throttled chunk."
        )
    if outcome == "retry-only":
        return (
            f"All `{result.requested_rows}/{result.requested_rows}` requests ended `retryable_error`; "
            f"queue stops here and should retry later."
        )
    if outcome == "rate-limit-cancelled":
        return (
            f"Cancelled after `{retryable}` consecutive `429` responses "
            f"(`{result.attempted_rows}/{result.requested_rows}` planned requests attempted); "
            "treat as Open-Meteo daily-limit and retry only after cooldown or an API-key path."
        )
    return "Run stopped without a clean success classification; inspect stderr progress and request payloads."


def load_pretrain_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def pretrain_metadata_path(*, artifacts_dir: Path, model_name: str, variant: str | None) -> Path:
    artifact_stem = model_name if variant is None else f"{model_name}__{variant}"
    return artifacts_dir / "models" / "v2" / f"{artifact_stem}.json"


def pretrain_row_gap(*, snapshot: WeatherTrainSnapshot, metadata: dict[str, Any]) -> int:
    pretrain_rows = int(metadata.get("weather_training_rows") or 0)
    return max(snapshot.total_rows - pretrain_rows, 0)


def should_refresh_pretrain(*, snapshot: WeatherTrainSnapshot, metadata: dict[str, Any], threshold_rows: int) -> bool:
    if threshold_rows <= 0:
        return False
    return pretrain_row_gap(snapshot=snapshot, metadata=metadata) >= threshold_rows


def refresh_weather_pretrain(
    *,
    dataset_path: Path,
    artifacts_dir: Path,
    dataset_profile: str,
    workspace_name: str,
    seed: int,
    model_name: str = "gaussian_emos",
    variant: str | None = None,
) -> dict[str, Any]:
    frame = pd.read_parquet(dataset_path)
    if frame.empty:
        raise ValueError("Weather training dataset is empty.")
    forbidden_columns = {"market_id", "market_spec_json", "market_prices_json", "clob_token_ids"}.intersection(frame.columns)
    if forbidden_columns:
        raise ValueError(f"Weather pretrain input must not contain Polymarket columns: {sorted(forbidden_columns)}")
    artifact = train_model(
        require_supported_model_name(model_name),
        frame,
        artifacts_dir,
        split_policy="target_day",
        seed=seed,
        variant=variant,
    )
    sidecar = Path(str(artifact.path)).with_suffix(".json")
    payload = artifact.model_dump(mode="json")
    payload.update(
        {
            "dataset_profile": dataset_profile,
            "workspace_name": workspace_name,
            "dataset_path": str(dataset_path),
            "weather_training_rows": int(len(frame)),
            "pretrain_scope": "weather_real_only",
        }
    )
    dump_json(sidecar, payload)
    return payload


def extract_retry_only_dates(entries: list[CollectionLogEntry]) -> list[date]:
    dates: list[date] = []
    for entry in entries:
        if entry.outcome not in {"retry-only", "rate-limit-cancelled"}:
            continue
        dates.extend(parse_dates_from_range_text(entry.range_text))
    return dates


def _latest_collection_entry(entries: list[CollectionLogEntry], *, outcomes: set[str]) -> CollectionLogEntry | None:
    for entry in reversed(entries):
        if entry.outcome not in outcomes:
            continue
        if not parse_dates_from_range_text(entry.range_text):
            continue
        return entry
    return None


def render_status_markdown(
    *,
    snapshot: WeatherTrainSnapshot,
    entries: list[CollectionLogEntry],
    pretrain_metadata: dict[str, Any],
    next_queue_start: date,
    chunk_days: int,
) -> str:
    full_count = snapshot.station_count
    full_ranges = format_grouped_ranges(
        [target_date for target_date, count in snapshot.counts_by_date.items() if count == full_count]
    )
    partial_ranges = format_grouped_counts(
        {
            target_date: count
            for target_date, count in snapshot.counts_by_date.items()
            if 0 < count < full_count
        }
    )
    retry_only_ranges = format_grouped_ranges(
        [
            target_date
            for target_date in extract_retry_only_dates(entries)
            if snapshot.counts_by_date.get(target_date, 0) == 0
        ]
    )
    latest_success = _latest_collection_entry(entries, outcomes={"success"})
    latest_throttle = _latest_collection_entry(entries, outcomes={"partial", "retry-only", "rate-limit-cancelled", "interrupted"})

    pretrain_path = str(pretrain_metadata.get("path", "artifacts/workspaces/weather_train/models/v2/gaussian_emos.pkl"))
    metadata_path = "artifacts/workspaces/weather_train/models/v2/gaussian_emos.json"
    dataset_signature = str(pretrain_metadata.get("dataset_signature", "unknown"))
    trained_at = str(pretrain_metadata.get("trained_at", "unknown"))
    pretrain_rows = int(pretrain_metadata.get("weather_training_rows") or 0)
    row_gap = max(snapshot.total_rows - pretrain_rows, 0)

    lines: list[str] = [
        "# Weather Train Status",
        "",
        f"Updated: {datetime.now(tz=SEOUL_TZ).date().isoformat()} KST",
        "",
        "## Current Snapshot",
        "- workspace: `weather_train`",
        "- dataset profile: `weather_real`",
        "- gold parquet: `data/workspaces/weather_train/parquet/gold/weather_training_set.parquet`",
        f"- total rows: `{snapshot.total_rows:,}`",
        f"- stations: `{snapshot.station_count}`",
        f"- target dates: `{snapshot.target_date_count}`",
        "- model values in dataset: `gfs_seamless`",
        "- `realized_daily_max` missing rows: `0`",
        f"- max target date present: `{snapshot.max_target_date.isoformat() if snapshot.max_target_date else 'n/a'}`",
        "",
        "## Coverage",
        "",
        "### Full-Coverage Dates",
    ]
    if full_ranges:
        for start_date, end_date in full_ranges:
            lines.append(f"- `{format_date_range(start_date, end_date)}`: `{full_count}` rows per day")
    else:
        lines.append("- none")

    lines.extend(["", "### Partial-Coverage Dates"])
    if partial_ranges:
        for start_date, end_date, count in partial_ranges:
            lines.append(f"- `{format_date_range(start_date, end_date)}`: `{count}` rows per day")
    else:
        lines.append("- none")

    lines.extend(["", "### Retry-Only Dates"])
    if retry_only_ranges:
        for start_date, end_date in retry_only_ranges:
            lines.append(
                f"- `{format_date_range(start_date, end_date)}`: `0/{full_count} available` on recorded probes; "
                "`retryable_error` / free-path daily-limit"
            )
    else:
        lines.append("- none recorded")

    lines.extend(["", "## Current Judgment"])
    lines.append(
        f"- Queue agent advances older gap-fill in `{chunk_days}`-day chunks; `2` consecutive Open-Meteo "
        "`429` responses are treated as a daily-limit hit and cancel the remaining chunk."
    )
    if latest_success is not None:
        lines.append(
            f"- Latest successful collection range: `{latest_success.range_text}`; "
            f"`+{latest_success.rows_added:,}` rows."
        )
    if latest_throttle is not None:
        lines.append(
            f"- Latest throttled range on record: `{latest_throttle.range_text}`; outcome `{latest_throttle.outcome}`."
        )
    lines.append(
        "- Working interpretation: older backfill can reopen after cooldown windows, while recent-date "
        "historical-forecast collection remains materially weaker on the free path."
    )

    lines.extend(
        [
            "",
            "## Current Artifacts",
            f"- latest weather pretrain artifact:\n  `{pretrain_path}`",
            f"- artifact metadata:\n  `{metadata_path}`",
            f"- pretrain dataset signature:\n  `{dataset_signature}`",
            f"- pretrain trained at:\n  `{trained_at}`",
            "",
            "## Next Collection Queue",
            f"1. Continue older gap-fill from `{next_queue_start.isoformat()}` forward with `{chunk_days}`-day chunks only after the free path cooldown/reset.",
            "2. Keep isolated retry-only gaps as separate probes; do not block the forward older-backfill queue on them.",
            "3. Retry `2026-01-22..2026-01-28` only as low-frequency probes or after moving to a paid/API-key path.",
            "",
            "## Training Ready State",
        ]
    )
    if row_gap > 0:
        lines.append(
            f"- weather pretrain artifact trails the current dataset by `{row_gap:,}` rows "
            f"(artifact `{pretrain_rows:,}` vs dataset `{snapshot.total_rows:,}`)"
        )
    else:
        lines.append(f"- weather pretrain artifact is aligned with the current dataset at `{snapshot.total_rows:,}` rows")
    lines.extend(
        [
            "- historical fine-tune input exists:\n  `data/workspaces/historical_real/parquet/gold/historical_training_set.parquet`",
            "- next recommended market fine-tune command:",
            "",
            "```bash",
            "scripts/pmtmax-workspace historical_real uv run pmtmax train-advanced \\",
            "  --model-name lgbm_emos \\",
            "  --variant high_neighbor_oof \\",
            "  --pretrained-weather-model artifacts/workspaces/weather_train/models/v2/gaussian_emos.pkl",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_weather_train_queue_agent(
    *,
    status_path: Path,
    collection_log_path: Path,
    queue_anchor_date: date,
    queue_start_override: date | None,
    chunk_days: int,
    max_chunks: int | None,
    sleep_seconds: float,
    model: str,
    missing_only: bool,
    workers: int,
    rate_limit_profile: str,
    http_timeout_seconds: int,
    http_retries: int,
    http_retry_wait_min_seconds: float,
    http_retry_wait_max_seconds: float,
    progress: bool,
    max_consecutive_429: int,
    pretrain_refresh_threshold_rows: int,
    pretrain_model_name: str,
    pretrain_variant: str | None,
) -> QueueAgentSummary:
    config, env = load_settings()
    configure_logging(env.log_level)
    set_global_seed(config.app.random_seed)
    report = _trust_check_report(
        config=config,
        markets_path=None,
        workflow="weather_training",
        output_name="weather_training_set",
    )
    issues = report.get("issues", [])
    if issues:
        messages = [
            str(issue.get("message", issue))
            for issue in issues
            if isinstance(issue, dict)
        ]
        raise RuntimeError("; ".join(messages) if messages else "weather_train trust-check failed")

    paths = default_weather_training_paths(config.app.parquet_dir)
    snapshot = load_weather_train_snapshot(paths.gold_path)
    start_date = queue_start_override or infer_next_queue_start(
        status_path=status_path,
        snapshot=snapshot,
        anchor_date=queue_anchor_date,
    )
    queue_start = start_date
    rows_added_total = 0
    chunks_attempted = 0
    last_outcome = "interrupted"
    stop_reason = "max_chunks"
    last_end_date = start_date
    metadata_path = pretrain_metadata_path(
        artifacts_dir=config.app.artifacts_dir,
        model_name=pretrain_model_name,
        variant=pretrain_variant,
    )
    pretrain_refreshed = False
    latest_pretrain_rows = int(load_pretrain_metadata(metadata_path).get("weather_training_rows") or 0)

    http = CachedHttpClient(
        config.app.cache_dir,
        timeout_seconds=http_timeout_seconds,
        retries=http_retries,
        retry_wait_min_seconds=http_retry_wait_min_seconds,
        retry_wait_max_seconds=http_retry_wait_max_seconds,
    )
    openmeteo = OpenMeteoClient(http, config.weather.openmeteo_base_url, config.weather.archive_base_url)
    try:
        while True:
            if max_chunks is not None and chunks_attempted >= max_chunks:
                break
            end_date = start_date + timedelta(days=chunk_days - 1)
            last_end_date = end_date
            before = load_weather_train_snapshot(paths.gold_path)
            result = collect_weather_training_data(
                openmeteo=openmeteo,
                raw_root=config.app.raw_dir,
                parquet_root=config.app.parquet_dir,
                station_cities=None,
                date_from=start_date,
                date_to=end_date,
                model=model,
                missing_only=missing_only,
                workers=workers,
                rate_limit_profile=rate_limit_profile,
                progress_callback=_render_weather_training_progress if progress else None,
                max_consecutive_429=max_consecutive_429,
            )
            after = load_weather_train_snapshot(paths.gold_path)
            rows_added = max(after.total_rows - before.total_rows, 0)
            rows_added_total += rows_added
            chunks_attempted += 1
            outcome = classify_collection_outcome(result)
            last_outcome = outcome
            next_queue_start = end_date + timedelta(days=1) if outcome in {"success", "rate-limit-cancelled"} else start_date
            append_collection_log_entry(
                collection_log_path,
                CollectionLogEntry(
                    run_date=datetime.now(tz=SEOUL_TZ).date().isoformat(),
                    range_text=format_date_range(start_date, end_date),
                    mode=f"{chunk_days}-day queue agent",
                    outcome=outcome,
                    rows_added=rows_added,
                    notes=build_collection_note(
                        result=result,
                        outcome=outcome,
                        next_queue_start=next_queue_start,
                    ),
                ),
            )
            entries = parse_collection_log(collection_log_path)
            pretrain_metadata = load_pretrain_metadata(metadata_path)
            refresh_gap = pretrain_row_gap(snapshot=after, metadata=pretrain_metadata)
            if rows_added > 0 and should_refresh_pretrain(
                snapshot=after,
                metadata=pretrain_metadata,
                threshold_rows=pretrain_refresh_threshold_rows,
            ):
                pretrain_metadata = refresh_weather_pretrain(
                    dataset_path=paths.gold_path,
                    artifacts_dir=config.app.artifacts_dir / "models" / "v2",
                    dataset_profile=config.app.dataset_profile,
                    workspace_name=config.app.workspace_name,
                    seed=config.app.random_seed,
                    model_name=pretrain_model_name,
                    variant=pretrain_variant,
                )
                pretrain_refreshed = True
                latest_pretrain_rows = int(pretrain_metadata.get("weather_training_rows") or 0)
                append_collection_log_entry(
                    collection_log_path,
                    CollectionLogEntry(
                        run_date=datetime.now(tz=SEOUL_TZ).date().isoformat(),
                        range_text="weather_train pretrain auto-refresh",
                        mode=pretrain_model_name if pretrain_variant is None else f"{pretrain_model_name}/{pretrain_variant}",
                        outcome="success",
                        rows_added=0,
                        notes=(
                            f"Triggered at dataset row gap `{refresh_gap:,}`; "
                            f"refreshed on `{latest_pretrain_rows:,}` rows with dataset signature "
                            f"`{pretrain_metadata.get('dataset_signature', 'unknown')}`."
                        ),
                    ),
                )
                entries = parse_collection_log(collection_log_path)
                print(
                    json.dumps(
                        {
                            "pretrain_refreshed": True,
                            "model_name": pretrain_model_name,
                            "variant": pretrain_variant,
                            "weather_training_rows": latest_pretrain_rows,
                            "dataset_signature": pretrain_metadata.get("dataset_signature"),
                            "trained_at": pretrain_metadata.get("trained_at"),
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
            else:
                latest_pretrain_rows = int(pretrain_metadata.get("weather_training_rows") or 0)
            status_path.write_text(
                render_status_markdown(
                    snapshot=after,
                    entries=entries,
                    pretrain_metadata=pretrain_metadata,
                    next_queue_start=next_queue_start,
                    chunk_days=chunk_days,
                ),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "range": format_date_range(start_date, end_date),
                        "outcome": outcome,
                        "rows_added": rows_added,
                        "requested_rows": result.requested_rows,
                        "attempted_rows": result.attempted_rows,
                        "collected_rows": result.collected_rows,
                        "failed_rows": result.failed_rows,
                        "status_counts": result.status_counts,
                        "early_stop_reason": result.early_stop_reason,
                        "next_queue_start": next_queue_start.isoformat(),
                        "dataset_rows": after.total_rows,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            if outcome != "success":
                stop_reason = outcome
                snapshot = after
                break
            snapshot = after
            start_date = next_queue_start
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    finally:
        http.close()

    return QueueAgentSummary(
        queue_start=queue_start.isoformat(),
        queue_end=last_end_date.isoformat(),
        chunk_days=chunk_days,
        chunks_attempted=chunks_attempted,
        rows_added=rows_added_total,
        stop_reason=stop_reason,
        last_outcome=last_outcome,
        dataset_rows=snapshot.total_rows,
        dataset_dates=snapshot.target_date_count,
        max_target_date=snapshot.max_target_date.isoformat() if snapshot.max_target_date else None,
        pretrain_refreshed=pretrain_refreshed,
        pretrain_rows=latest_pretrain_rows,
    )
