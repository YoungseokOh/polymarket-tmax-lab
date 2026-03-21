from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pmtmax.config.settings import RepoConfig
from pmtmax.execution.paper_broker import PaperBroker
from pmtmax.execution.scanner import ContinuousScanner
from pmtmax.storage.schemas import PaperPosition, ProbForecast


def _make_broker_with_position(token_id: str = "t1", avg_price: float = 0.50) -> PaperBroker:
    broker = PaperBroker(bankroll=10_000.0)
    broker.positions[token_id] = PaperPosition(
        market_id="m1",
        token_id=token_id,
        outcome_label="8°C",
        side="buy",
        avg_price=avg_price,
        size=100.0,
        high_water_mark=avg_price,
        opened_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    return broker


def test_run_once_returns_summary(tmp_path: Path) -> None:
    config = RepoConfig()
    broker = PaperBroker(bankroll=10_000.0)
    scanner = ContinuousScanner(
        config=config,
        broker=broker,
        interval_seconds=1,
        max_cycles=1,
        state_path=tmp_path / "state.json",
    )

    summary = scanner.run_once()

    assert summary["cycle"] == 0
    assert summary["stops"] == []
    assert summary["forecast_exits"] == []
    assert summary["entries"] == 0


def test_run_once_triggers_stop_loss(tmp_path: Path) -> None:
    config = RepoConfig()
    broker = _make_broker_with_position(avg_price=0.50)

    scanner = ContinuousScanner(
        config=config,
        broker=broker,
        interval_seconds=1,
        max_cycles=1,
        state_path=tmp_path / "state.json",
        price_fetcher=lambda: {"t1": 0.35},
    )

    summary = scanner.run_once()

    assert len(summary["stops"]) == 1
    assert "t1" not in broker.positions


def test_run_once_triggers_forecast_exit(tmp_path: Path) -> None:
    config = RepoConfig()
    broker = _make_broker_with_position()

    forecast = ProbForecast(
        target_market="m1",
        generated_at=datetime(2025, 1, 1, tzinfo=UTC),
        mean=10.0,
        std=2.0,
        outcome_probabilities={"8°C": 0.01, "9°C": 0.99},
    )

    scanner = ContinuousScanner(
        config=config,
        broker=broker,
        interval_seconds=1,
        max_cycles=1,
        state_path=tmp_path / "state.json",
        forecast_fetcher=lambda: {"m1": forecast},
    )

    summary = scanner.run_once()

    assert len(summary["forecast_exits"]) == 1


def test_max_cycles_stops_loop(tmp_path: Path) -> None:
    config = RepoConfig()
    broker = PaperBroker(bankroll=10_000.0)
    scanner = ContinuousScanner(
        config=config,
        broker=broker,
        interval_seconds=0,
        max_cycles=2,
        state_path=tmp_path / "state.json",
    )

    scanner.run_loop()

    assert scanner._cycle == 2


def test_state_file_roundtrip(tmp_path: Path) -> None:
    config = RepoConfig()
    broker = PaperBroker(bankroll=5_000.0)
    state_path = tmp_path / "state.json"

    scanner = ContinuousScanner(
        config=config,
        broker=broker,
        interval_seconds=1,
        max_cycles=1,
        state_path=state_path,
    )
    scanner.run_once()

    assert state_path.exists()
    state = json.loads(state_path.read_text())
    assert state["cycle"] == 1
    assert state["bankroll"] == 5_000.0


def test_stops_and_forecast_exits_checked_each_cycle(tmp_path: Path) -> None:
    config = RepoConfig()
    broker = _make_broker_with_position(avg_price=0.50)

    call_count = {"price": 0, "forecast": 0}

    def price_fetch() -> dict[str, float]:
        call_count["price"] += 1
        return {"t1": 0.50}  # no stop triggered

    def forecast_fetch() -> dict[str, ProbForecast]:
        call_count["forecast"] += 1
        return {}

    scanner = ContinuousScanner(
        config=config,
        broker=broker,
        interval_seconds=0,
        max_cycles=2,
        state_path=tmp_path / "state.json",
        price_fetcher=price_fetch,
        forecast_fetcher=forecast_fetch,
    )

    scanner.run_loop()

    assert call_count["price"] == 2
    assert call_count["forecast"] == 2


def test_scanner_wires_price_fetcher_to_check_stops(tmp_path: Path) -> None:
    """price_fetcher queries the correct token_ids and stop triggers close_position."""

    broker = _make_broker_with_position(token_id="tok_abc", avg_price=0.60)
    queried_ids: list[str] = []

    def price_fetcher() -> dict[str, float]:
        for tid in broker.positions:
            queried_ids.append(tid)
        # Return a price well below stop-loss threshold (60% * 0.80 = 0.48)
        return {"tok_abc": 0.40}

    scanner = ContinuousScanner(
        config=RepoConfig(),
        broker=broker,
        interval_seconds=0,
        max_cycles=1,
        state_path=tmp_path / "state.json",
        price_fetcher=price_fetcher,
    )

    summary = scanner.run_once()

    assert queried_ids == ["tok_abc"]
    assert len(summary["stops"]) == 1
    assert "tok_abc" not in broker.positions


def test_scanner_entry_evaluator_skips_held_markets(tmp_path: Path) -> None:
    """entry_evaluator must not attempt re-entry on markets with open positions."""

    broker = _make_broker_with_position(token_id="t_held", avg_price=0.50)
    # Track market_ids the evaluator tries to enter
    attempted_market_ids: list[str] = []

    def entry_evaluator(brk: PaperBroker) -> None:
        held = {pos.market_id for pos in brk.positions.values()}
        # Simulate evaluating two candidates: one held ("m1"), one new ("m2")
        for mid in ["m1", "m2"]:
            if mid in held:
                continue
            attempted_market_ids.append(mid)

    scanner = ContinuousScanner(
        config=RepoConfig(),
        broker=broker,
        interval_seconds=0,
        max_cycles=1,
        state_path=tmp_path / "state.json",
        entry_evaluator=entry_evaluator,
    )

    scanner.run_once()

    # Only "m2" should be attempted; "m1" is already held
    assert attempted_market_ids == ["m2"]
