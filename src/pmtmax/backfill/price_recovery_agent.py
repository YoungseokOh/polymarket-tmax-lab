"""Automate daily historical_real price-history recovery and checker updates."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd

from pmtmax.cli.main import _backfill_pipeline, _bootstrap_snapshots, _runtime, _trust_check_report
from pmtmax.markets.clob_read_client import ClobReadClient
from pmtmax.storage.schemas import MarketSnapshot
from pmtmax.utils import dump_json

SEOUL_TZ = ZoneInfo("Asia/Seoul")
NEXT_SHARD_RE = re.compile(r"Continue missing-price recovery from shard `(\d+)\.\.(\d+)`")
REAL_HISTORY_METRICS_PATH = Path("artifacts/workspaces/historical_real/backtests/v2/backtest_metrics_real_history.json")


@dataclass(frozen=True)
class MarketShard:
    offset_start: int
    offset_end: int
    selected_markets: int
    total_markets: int
    date_from: date | None
    date_to: date | None
    cities: tuple[str, ...]

    @property
    def range_text(self) -> str:
        return f"{self.offset_start}..{self.offset_end}"

    @property
    def queue_text(self) -> str:
        return f"{self.range_text} / {self.total_markets}"

    @property
    def date_span_text(self) -> str:
        if self.date_from is None or self.date_to is None:
            return "n/a"
        if self.date_from == self.date_to:
            return self.date_from.isoformat()
        return f"{self.date_from.isoformat()}..{self.date_to.isoformat()}"

    @property
    def city_text(self) -> str:
        if not self.cities:
            return "n/a"
        if len(self.cities) <= 3:
            return ", ".join(self.cities)
        head = ", ".join(self.cities[:3])
        return f"{head}, +{len(self.cities) - 3} more"


@dataclass(frozen=True)
class HistoricalPriceSnapshot:
    total_markets: int
    request_count: int
    request_status_counts: dict[str, int]
    empty_with_last_trade_count: int
    last_trade_probe_count: int
    panel_token_counts: dict[str, int]
    decision_ready_rows: int
    decision_total_rows: int
    latest_backtest_priced_decision_rows: int | None
    latest_backtest_pnl: float | None
    latest_backtest_path: str | None
    latest_backtest_updated_at: str | None


@dataclass(frozen=True)
class HistoricalPriceLogEntry:
    run_date: str
    shard_text: str
    mode: str
    outcome: str
    markets_processed: int
    decision_ready_delta: int
    notes: str

    def to_markdown_row(self) -> str:
        return (
            f"| {self.run_date} | `{self.shard_text}` | {self.mode} | {self.outcome} | "
            f"{self.markets_processed} | {self.decision_ready_delta:+d} | {self.notes} |"
        )


@dataclass(frozen=True)
class HistoricalPriceRecoverySummary:
    shard_start: int
    shard_end: int
    shards_attempted: int
    markets_processed: int
    decision_ready_delta: int
    stop_reason: str
    last_outcome: str
    next_shard_start: int
    decision_ready_rows: int
    decision_total_rows: int
    latest_backtest_priced_decision_rows: int | None


def describe_shard(snapshots: list[MarketSnapshot], *, offset_start: int, shard_size: int) -> MarketShard:
    if not snapshots:
        raise ValueError("No parsed market snapshots are available for price recovery.")
    normalized_start = 0 if offset_start >= len(snapshots) else max(offset_start, 0)
    offset_end = min(normalized_start + shard_size, len(snapshots)) - 1
    selected = snapshots[normalized_start : offset_end + 1]
    target_dates = [snapshot.spec.target_local_date for snapshot in selected if snapshot.spec is not None]
    cities = tuple(sorted({snapshot.spec.city for snapshot in selected if snapshot.spec is not None}))
    return MarketShard(
        offset_start=normalized_start,
        offset_end=offset_end,
        selected_markets=len(selected),
        total_markets=len(snapshots),
        date_from=min(target_dates) if target_dates else None,
        date_to=max(target_dates) if target_dates else None,
        cities=cities,
    )


def next_shard_start(shard: MarketShard) -> int:
    candidate = shard.offset_end + 1
    if candidate >= shard.total_markets:
        return 0
    return candidate


def ensure_collection_log(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Historical Price Recovery Log\n\n"
        "Append-only operational log for `historical_real` official price-history recovery.\n\n"
        "| Run Date | Shard | Mode | Outcome | Markets | Ready Delta | Notes |\n"
        "| --- | --- | --- | --- | ---: | ---: | --- |\n",
        encoding="utf-8",
    )


def parse_collection_log(path: Path) -> list[HistoricalPriceLogEntry]:
    if not path.exists():
        return []
    entries: list[HistoricalPriceLogEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        stripped = line.strip()
        if stripped.startswith("| Run Date ") or stripped.startswith("| --- "):
            continue
        parts = [part.strip() for part in stripped.split("|")[1:-1]]
        if len(parts) != 7:
            continue
        entries.append(
            HistoricalPriceLogEntry(
                run_date=parts[0],
                shard_text=parts[1].strip("`"),
                mode=parts[2],
                outcome=parts[3],
                markets_processed=int(parts[4] or 0),
                decision_ready_delta=int(parts[5] or 0),
                notes=parts[6],
            )
        )
    return entries


def append_collection_log_entry(path: Path, entry: HistoricalPriceLogEntry) -> None:
    ensure_collection_log(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{entry.to_markdown_row()}\n")


def infer_next_shard_offset(status_path: Path, *, total_markets: int) -> int:
    if status_path.exists():
        match = NEXT_SHARD_RE.search(status_path.read_text(encoding="utf-8"))
        if match:
            offset = int(match.group(1))
            if offset < total_markets:
                return offset
    return 0


def _status_counts(frame: pd.DataFrame, *, status_column: str, value_column: str) -> dict[str, int]:
    if frame.empty or status_column not in frame.columns or value_column not in frame.columns:
        return {}
    grouped = frame.groupby(status_column, dropna=False)[value_column].sum()
    return {str(key): int(value) for key, value in grouped.to_dict().items()}


def _load_real_history_metrics(path: Path) -> tuple[int | None, float | None, str | None, str | None]:
    if not path.exists():
        return None, None, None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    priced_rows = payload.get("priced_decision_rows")
    pnl = payload.get("pnl")
    priced_value = None if priced_rows is None else int(round(float(priced_rows)))
    pnl_value = None if pnl is None else float(pnl)
    updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=SEOUL_TZ).isoformat()
    return priced_value, pnl_value, str(path), updated_at


def _coverage_payload(result: dict[str, pd.DataFrame]) -> dict[str, list[dict[str, object]]]:
    return {
        "request_summary": result["request_summary"].to_dict(orient="records"),
        "request_details": result["request_details"].to_dict(orient="records"),
        "panel_summary": result["panel_summary"].to_dict(orient="records"),
        "market_summary": result["market_summary"].to_dict(orient="records"),
        "details": result["details"].to_dict(orient="records"),
    }


def build_historical_price_snapshot(
    *,
    summary: dict[str, pd.DataFrame],
    total_markets: int,
    backtest_metrics_path: Path,
) -> HistoricalPriceSnapshot:
    request_summary = summary["request_summary"]
    panel_summary = summary["panel_summary"]
    market_summary = summary["market_summary"]

    request_count = int(request_summary["request_count"].sum()) if "request_count" in request_summary.columns else 0
    empty_with_last_trade_count = (
        int(request_summary["history_empty_with_last_trade_count"].sum())
        if "history_empty_with_last_trade_count" in request_summary.columns
        else 0
    )
    last_trade_probe_count = (
        int(request_summary["last_trade_probe_count"].sum())
        if "last_trade_probe_count" in request_summary.columns
        else 0
    )
    request_status_counts = _status_counts(request_summary, status_column="status", value_column="request_count")
    panel_token_counts = _status_counts(panel_summary, status_column="coverage_status", value_column="token_count")

    decision_ready_rows = 0
    decision_total_rows = 0
    if not market_summary.empty:
        decision_total_rows = int(len(market_summary))
        ready_mask = (
            market_summary["missing_token_count"].fillna(0).astype(int).eq(0)
            & market_summary["stale_token_count"].fillna(0).astype(int).eq(0)
            & market_summary["ok_token_count"].fillna(0).astype(int).eq(market_summary["token_count"].fillna(0).astype(int))
        )
        decision_ready_rows = int(ready_mask.sum())

    priced_rows, pnl, metrics_path_str, updated_at = _load_real_history_metrics(backtest_metrics_path)
    return HistoricalPriceSnapshot(
        total_markets=total_markets,
        request_count=request_count,
        request_status_counts=request_status_counts,
        empty_with_last_trade_count=empty_with_last_trade_count,
        last_trade_probe_count=last_trade_probe_count,
        panel_token_counts=panel_token_counts,
        decision_ready_rows=decision_ready_rows,
        decision_total_rows=decision_total_rows,
        latest_backtest_priced_decision_rows=priced_rows,
        latest_backtest_pnl=pnl,
        latest_backtest_path=metrics_path_str,
        latest_backtest_updated_at=updated_at,
    )


def classify_recovery_outcome(
    *,
    batch_request_count: int,
    status_counts: dict[str, int],
) -> Literal["success", "partial", "retry-only"]:
    if batch_request_count == 0:
        return "success"
    hard_errors = int(status_counts.get("http_error", 0) or 0) + int(status_counts.get("error", 0) or 0)
    completed = int(status_counts.get("ok", 0) or 0) + int(status_counts.get("empty", 0) or 0)
    if hard_errors == 0:
        return "success"
    if completed > 0:
        return "partial"
    return "retry-only"


def _latest_log_entry(entries: list[HistoricalPriceLogEntry], *, outcomes: set[str]) -> HistoricalPriceLogEntry | None:
    for entry in reversed(entries):
        if entry.outcome in outcomes:
            return entry
    return None


def render_status_markdown(
    *,
    snapshot: HistoricalPriceSnapshot,
    entries: list[HistoricalPriceLogEntry],
    next_shard: MarketShard,
    shard_size: int,
    markets_path: Path,
    coverage_output_path: Path,
) -> str:
    request_counts = snapshot.request_status_counts
    panel_counts = snapshot.panel_token_counts
    ok_requests = int(request_counts.get("ok", 0) or 0)
    empty_requests = int(request_counts.get("empty", 0) or 0)
    empty_ratio = 0.0 if snapshot.request_count == 0 else empty_requests / snapshot.request_count
    decision_ratio = 0.0 if snapshot.decision_total_rows == 0 else snapshot.decision_ready_rows / snapshot.decision_total_rows
    latest_success = _latest_log_entry(entries, outcomes={"success"})
    latest_throttle = _latest_log_entry(entries, outcomes={"partial", "retry-only"})

    lines: list[str] = [
        "# Historical Price Status",
        "",
        f"Updated: {datetime.now(tz=SEOUL_TZ).date().isoformat()} KST",
        "",
        "## Current Snapshot",
        "- workspace: `historical_real`",
        "- dataset profile: `real_market`",
        f"- inventory: `{markets_path}`",
        f"- inventory markets: `{snapshot.total_markets:,}`",
        "- gold panel: `data/workspaces/historical_real/parquet/gold/historical_backtest_panel.parquet`",
        f"- coverage artifact: `{coverage_output_path}`",
        f"- tracked token request rows: `{snapshot.request_count:,}`",
        f"- tracked decision rows: `{snapshot.decision_total_rows:,}`",
        "",
        "## Request Coverage",
        f"- status `ok`: `{ok_requests:,}`",
        f"- status `empty`: `{empty_requests:,}`",
        f"- status `http_error`: `{int(request_counts.get('http_error', 0) or 0):,}`",
        f"- status `error`: `{int(request_counts.get('error', 0) or 0):,}`",
        f"- `empty` with `last_trade_present=true`: `{snapshot.empty_with_last_trade_count:,}` / `{empty_requests:,}`",
        f"- last-trade probes recorded: `{snapshot.last_trade_probe_count:,}`",
        "",
        "## Panel Readiness",
        f"- token coverage `ok`: `{int(panel_counts.get('ok', 0) or 0):,}`",
        f"- token coverage `missing`: `{int(panel_counts.get('missing', 0) or 0):,}`",
        f"- token coverage `stale`: `{int(panel_counts.get('stale', 0) or 0):,}`",
        (
            f"- panel-ready decision rows: `{snapshot.decision_ready_rows:,}` / "
            f"`{snapshot.decision_total_rows:,}` (`{decision_ratio:.1%}`)"
        ),
    ]
    if snapshot.latest_backtest_priced_decision_rows is not None:
        lines.extend(
            [
                f"- latest backtest `priced_decision_rows`: `{snapshot.latest_backtest_priced_decision_rows:,}`",
                f"- latest backtest `PnL`: `{snapshot.latest_backtest_pnl:.2f}`" if snapshot.latest_backtest_pnl is not None else "- latest backtest `PnL`: `n/a`",
                f"- latest backtest metrics artifact: `{snapshot.latest_backtest_path}`",
                f"- latest backtest artifact updated at: `{snapshot.latest_backtest_updated_at}`",
            ]
        )
    else:
        lines.append("- latest backtest `priced_decision_rows`: `n/a`")

    lines.extend(["", "## Current Judgment"])
    lines.append(
        f"- Daily price agent runs one shard at a time (`{shard_size}` markets default) and keeps "
        "`backfill-price-history -> materialize-backtest-panel -> summarize-price-history-coverage` serialized."
    )
    if latest_success is not None:
        lines.append(
            f"- Latest successful shard: `{latest_success.shard_text}`; decision-ready delta `{latest_success.decision_ready_delta:+d}`."
        )
    if latest_throttle is not None:
        lines.append(
            f"- Latest shard with upstream errors: `{latest_throttle.shard_text}`; outcome `{latest_throttle.outcome}`."
        )
    if empty_ratio >= 0.5 and snapshot.empty_with_last_trade_count > 0:
        lines.append(
            "- Dominant blocker: official `/prices-history` empties still outweigh recovered rows, and many empties "
            "show `last_trade_present`, so retention-limited history remains the main constraint."
        )
    else:
        lines.append("- Current blocker mix is not dominated by retention-limited empties; continue shard rotation.")

    lines.extend(
        [
            "",
            "## Next Recovery Queue",
            (
                f"1. Continue missing-price recovery from shard `{next_shard.range_text}` "
                f"(`{next_shard.selected_markets}` markets, dates `{next_shard.date_span_text}`, "
                f"cities `{next_shard.city_text}`)."
            ),
            "2. `weather_train` queue can run in parallel in a separate session; do not overlap another mutating `historical_real` job.",
            "3. Re-run `real_history` evaluation after a meaningful panel-ready gain or after one full shard cycle completes.",
            "",
            "## Daily Agent Command",
            "",
            "```bash",
            "scripts/pmtmax-workspace historical_real uv run python scripts/run_historical_price_recovery_agent.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _build_log_note(
    *,
    batch_request_count: int,
    status_counts: dict[str, int],
    token_ok_delta: int,
    decision_ready_delta: int,
    next_shard: MarketShard,
) -> str:
    if batch_request_count == 0:
        return (
            f"No missing token requests in shard; token `ok` delta `{token_ok_delta:+d}`, "
            f"decision-ready delta `{decision_ready_delta:+d}`; next shard `{next_shard.range_text}`."
        )
    status_text = ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items()))
    return (
        f"`{status_text}`; token `ok` delta `{token_ok_delta:+d}`, "
        f"decision-ready delta `{decision_ready_delta:+d}`; next shard `{next_shard.range_text}`."
    )


def run_historical_price_recovery_agent(
    *,
    status_path: Path,
    collection_log_path: Path,
    markets_path: Path,
    coverage_output_path: Path,
    dataset_path: Path,
    shard_size: int,
    shard_start_override: int | None,
    max_shards: int | None,
    sleep_seconds: float,
    interval: str,
    fidelity: int,
    only_missing: bool,
    price_no_cache: bool,
    output_name: str,
    max_price_age_minutes: int,
    allow_canonical_overwrite: bool,
) -> HistoricalPriceRecoverySummary:
    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    report = _trust_check_report(
        config=config,
        markets_path=markets_path,
        allow_canonical_overwrite=allow_canonical_overwrite,
        output_name=output_name,
        workflow="real_market",
    )
    issues = report.get("issues", [])
    if issues:
        messages = [str(issue.get("message", issue)) for issue in issues if isinstance(issue, dict)]
        raise RuntimeError("; ".join(messages) if messages else "historical_real trust-check failed")

    if not dataset_path.exists():
        raise FileNotFoundError(f"training dataset missing: {dataset_path}")

    raw_snapshots = _bootstrap_snapshots(markets_path=markets_path, cities=None)
    snapshots = [snapshot for snapshot in raw_snapshots if snapshot.spec is not None]
    if not snapshots:
        raise RuntimeError("No parsed snapshots were loaded from the training inventory.")

    ensure_collection_log(collection_log_path)
    coverage_output_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    start_offset = shard_start_override
    if start_offset is None:
        start_offset = infer_next_shard_offset(status_path, total_markets=len(snapshots))

    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    pipeline = _backfill_pipeline(config, http, openmeteo)
    shard_start = start_offset
    last_shard = describe_shard(snapshots, offset_start=start_offset, shard_size=shard_size)
    summary = pipeline.summarize_price_history_coverage({snapshot.spec.market_id for snapshot in snapshots if snapshot.spec is not None})
    dump_json(coverage_output_path, _coverage_payload(summary))
    snapshot_before = build_historical_price_snapshot(
        summary=summary,
        total_markets=len(snapshots),
        backtest_metrics_path=REAL_HISTORY_METRICS_PATH,
    )
    shards_attempted = 0
    markets_processed = 0
    decision_ready_delta_total = 0
    last_outcome = "interrupted"
    stop_reason = "max_shards"
    next_offset = start_offset

    try:
        while True:
            if max_shards is not None and shards_attempted >= max_shards:
                break
            shard = describe_shard(snapshots, offset_start=start_offset, shard_size=shard_size)
            last_shard = shard
            batch_snapshots = snapshots[shard.offset_start : shard.offset_end + 1]
            result = pipeline.backfill_price_history(
                batch_snapshots,
                clob=clob,
                interval=interval,
                fidelity=fidelity,
                use_cache=not price_no_cache,
                only_missing=only_missing,
            )
            frame = pd.read_parquet(dataset_path)
            pipeline.materialize_backtest_panel(
                frame,
                output_name=output_name,
                max_price_age_minutes=max_price_age_minutes,
                allow_canonical_overwrite=allow_canonical_overwrite,
            )
            coverage = pipeline.summarize_price_history_coverage({snapshot.spec.market_id for snapshot in snapshots if snapshot.spec is not None})
            dump_json(coverage_output_path, _coverage_payload(coverage))
            snapshot_after = build_historical_price_snapshot(
                summary=coverage,
                total_markets=len(snapshots),
                backtest_metrics_path=REAL_HISTORY_METRICS_PATH,
            )

            batch_bronze = result.get("batch_bronze_price_history_requests", pd.DataFrame())
            status_counts = (
                {str(key): int(value) for key, value in batch_bronze["status"].astype(str).value_counts().to_dict().items()}
                if not batch_bronze.empty and "status" in batch_bronze.columns
                else {}
            )
            batch_request_count = int(len(batch_bronze))
            token_ok_before = int(snapshot_before.panel_token_counts.get("ok", 0) or 0)
            token_ok_after = int(snapshot_after.panel_token_counts.get("ok", 0) or 0)
            token_ok_delta = token_ok_after - token_ok_before
            decision_ready_delta = snapshot_after.decision_ready_rows - snapshot_before.decision_ready_rows
            decision_ready_delta_total += decision_ready_delta
            shards_attempted += 1
            markets_processed += shard.selected_markets
            outcome = classify_recovery_outcome(batch_request_count=batch_request_count, status_counts=status_counts)
            last_outcome = outcome
            next_offset = next_shard_start(shard)
            next_shard = describe_shard(snapshots, offset_start=next_offset, shard_size=shard_size)

            append_collection_log_entry(
                collection_log_path,
                HistoricalPriceLogEntry(
                    run_date=datetime.now(tz=SEOUL_TZ).date().isoformat(),
                    shard_text=shard.queue_text,
                    mode="daily price recovery agent",
                    outcome=outcome,
                    markets_processed=shard.selected_markets,
                    decision_ready_delta=decision_ready_delta,
                    notes=_build_log_note(
                        batch_request_count=batch_request_count,
                        status_counts=status_counts,
                        token_ok_delta=token_ok_delta,
                        decision_ready_delta=decision_ready_delta,
                        next_shard=next_shard,
                    ),
                ),
            )
            entries = parse_collection_log(collection_log_path)
            status_path.write_text(
                render_status_markdown(
                    snapshot=snapshot_after,
                    entries=entries,
                    next_shard=next_shard,
                    shard_size=shard_size,
                    markets_path=markets_path,
                    coverage_output_path=coverage_output_path,
                ),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "shard": shard.queue_text,
                        "outcome": outcome,
                        "markets_processed": shard.selected_markets,
                        "batch_request_count": batch_request_count,
                        "status_counts": status_counts,
                        "token_ok_delta": token_ok_delta,
                        "decision_ready_delta": decision_ready_delta,
                        "next_shard_start": next_offset,
                        "decision_ready_rows": snapshot_after.decision_ready_rows,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            snapshot_before = snapshot_after
            if outcome != "success":
                stop_reason = outcome
                break
            start_offset = next_offset
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    finally:
        pipeline.warehouse.close()
        http.close()

    return HistoricalPriceRecoverySummary(
        shard_start=shard_start,
        shard_end=last_shard.offset_end,
        shards_attempted=shards_attempted,
        markets_processed=markets_processed,
        decision_ready_delta=decision_ready_delta_total,
        stop_reason=stop_reason,
        last_outcome=last_outcome,
        next_shard_start=next_offset,
        decision_ready_rows=snapshot_before.decision_ready_rows,
        decision_total_rows=snapshot_before.decision_total_rows,
        latest_backtest_priced_decision_rows=snapshot_before.latest_backtest_priced_decision_rows,
    )
