from __future__ import annotations

from pathlib import Path

import pandas as pd

from pmtmax.backfill.price_recovery_agent import (
    HistoricalPriceLogEntry,
    HistoricalPriceSnapshot,
    MarketShard,
    append_collection_log_entry,
    classify_recovery_outcome,
    infer_next_shard_offset,
    parse_collection_log,
    render_status_markdown,
)


def test_infer_next_shard_offset_prefers_status_hint(tmp_path: Path) -> None:
    status_path = tmp_path / "historical_price_status.md"
    status_path.write_text(
        "## Next Recovery Queue\n"
        "1. Continue missing-price recovery from shard `25..49` (`25` markets, dates `2026-03-01..2026-03-25`, cities `Ankara`).\n",
        encoding="utf-8",
    )

    offset = infer_next_shard_offset(status_path, total_markets=1834)

    assert offset == 25


def test_append_collection_log_entry_round_trips(tmp_path: Path) -> None:
    log_path = tmp_path / "historical_price_collection_log.md"

    append_collection_log_entry(
        log_path,
        HistoricalPriceLogEntry(
            run_date="2026-04-24",
            shard_text="0..24 / 1834",
            mode="daily price recovery agent",
            outcome="success",
            markets_processed=25,
            decision_ready_delta=4,
            notes="`empty=58, ok=17`; token `ok` delta `+144`, decision-ready delta `+4`; next shard `25..49`.",
        ),
    )

    entries = parse_collection_log(log_path)

    assert len(entries) == 1
    assert entries[0].shard_text == "0..24 / 1834"
    assert entries[0].decision_ready_delta == 4


def test_render_status_markdown_reports_retention_limited_blocker() -> None:
    snapshot = HistoricalPriceSnapshot(
        total_markets=1834,
        request_count=14674,
        request_status_counts={"ok": 5213, "empty": 9461},
        empty_with_last_trade_count=9461,
        last_trade_probe_count=9461,
        panel_token_counts={"ok": 11940, "missing": 31368, "stale": 18},
        decision_ready_rows=1213,
        decision_total_rows=5478,
        latest_backtest_priced_decision_rows=1155,
        latest_backtest_pnl=1029.24,
        latest_backtest_path="artifacts/workspaces/historical_real/backtests/v2/backtest_metrics_real_history.json",
        latest_backtest_updated_at="2026-04-24T22:10:00+09:00",
    )
    entries = [
        HistoricalPriceLogEntry(
            run_date="2026-04-24",
            shard_text="0..24 / 1834",
            mode="daily price recovery agent",
            outcome="success",
            markets_processed=25,
            decision_ready_delta=4,
            notes="first shard",
        )
    ]
    next_shard = MarketShard(
        offset_start=25,
        offset_end=49,
        selected_markets=25,
        total_markets=1834,
        date_from=pd.Timestamp("2026-03-01").date(),
        date_to=pd.Timestamp("2026-03-25").date(),
        cities=("Ankara",),
    )

    markdown = render_status_markdown(
        snapshot=snapshot,
        entries=entries,
        next_shard=next_shard,
        shard_size=25,
        markets_path=Path("configs/market_inventory/full_training_set_snapshots.json"),
        coverage_output_path=Path("artifacts/workspaces/historical_real/coverage/latest_price_history_coverage.json"),
    )

    assert "panel-ready decision rows: `1,213` / `5,478`" in markdown
    assert "retention-limited history remains the main constraint" in markdown
    assert "Continue missing-price recovery from shard `25..49`" in markdown


def test_classify_recovery_outcome_treats_empty_as_completed() -> None:
    assert classify_recovery_outcome(batch_request_count=0, status_counts={}) == "success"
    assert classify_recovery_outcome(batch_request_count=10, status_counts={"empty": 10}) == "success"
    assert classify_recovery_outcome(batch_request_count=10, status_counts={"ok": 6, "http_error": 4}) == "partial"
    assert classify_recovery_outcome(batch_request_count=10, status_counts={"http_error": 10}) == "retry-only"
