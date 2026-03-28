from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pmtmax.config.settings import RepoConfig
from pmtmax.markets.repository import bundled_market_snapshots
from pmtmax.monitoring.open_phase import (
    OpenPhaseShadowRunner,
    extract_open_phase_metadata,
    select_open_phase_candidates,
    summarize_open_phase_history,
)
from pmtmax.storage.schemas import OpenPhaseObservation


def _open_phase_snapshot(*, city: str = "Seoul", hours_ago: float = 1.0):
    snapshot = bundled_market_snapshots([city])[0].model_copy(deep=True)
    assert snapshot.spec is not None
    opened_at = datetime.now(tz=UTC) - timedelta(hours=hours_ago)
    snapshot.market["componentMarkets"] = [
        {
            "createdAt": (opened_at - timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
            "created_at": (opened_at - timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
            "deployingTimestamp": (opened_at - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
            "acceptingOrdersTimestamp": opened_at.isoformat().replace("+00:00", "Z"),
        }
    ]
    return snapshot, opened_at


def test_extract_open_phase_metadata_prefers_accepting_orders_timestamp() -> None:
    snapshot, opened_at = _open_phase_snapshot(hours_ago=3.0)
    observed_at = opened_at + timedelta(hours=3)

    metadata = extract_open_phase_metadata(snapshot, observed_at=observed_at)

    assert metadata["market_accepting_orders_at"] == opened_at
    assert metadata["market_opened_at"] == opened_at
    assert metadata["open_phase_age_hours"] == 3.0


def test_select_open_phase_candidates_filters_by_age_window() -> None:
    fresh_snapshot, _ = _open_phase_snapshot(hours_ago=2.0)
    stale_snapshot, _ = _open_phase_snapshot(hours_ago=30.0)
    observed_at = datetime.now(tz=UTC)

    candidates = select_open_phase_candidates(
        [fresh_snapshot, stale_snapshot],
        observed_at=observed_at,
        open_window_hours=24.0,
    )

    assert len(candidates) == 1
    assert candidates[0].snapshot.spec is not None
    assert candidates[0].snapshot.spec.city == "Seoul"
    assert candidates[0].open_phase_age_hours < 24.0


def test_open_phase_shadow_runner_writes_latest_history_summary_and_state(tmp_path: Path) -> None:
    config = RepoConfig()

    def _snapshot_fetcher():
        return ["m1", "m2"]

    def _evaluator(snapshots, observed_at):
        assert snapshots == ["m1", "m2"]
        return [
            OpenPhaseObservation(
                observed_at=observed_at,
                market_id="m1",
                city="Seoul",
                question="Highest temperature in Seoul on March 23?",
                target_local_date=datetime(2026, 3, 23, tzinfo=UTC).date(),
                decision_horizon="market_open",
                reason="tradable",
                market_opened_at=observed_at - timedelta(hours=1),
                open_phase_age_hours=1.0,
                outcome_label="11°C",
                fair_probability=0.55,
                best_bid=0.51,
                best_ask=0.53,
                spread=0.02,
                visible_liquidity=500.0,
                fee_estimate=0.0106,
                slippage_estimate=0.01,
                raw_gap=0.02,
                after_cost_edge=-0.0006,
            ),
            OpenPhaseObservation(
                observed_at=observed_at,
                market_id="m2",
                city="London",
                question="Highest temperature in London on March 23?",
                target_local_date=datetime(2026, 3, 23, tzinfo=UTC).date(),
                decision_horizon="market_open",
                reason="raw_gap_non_positive",
                market_opened_at=observed_at - timedelta(hours=2),
                open_phase_age_hours=2.0,
                outcome_label="14°C",
                fair_probability=0.31,
                best_bid=0.30,
                best_ask=0.34,
                spread=0.04,
                visible_liquidity=300.0,
                fee_estimate=0.0068,
                slippage_estimate=0.0201,
                raw_gap=-0.03,
                after_cost_edge=-0.0569,
            ),
        ]

    runner = OpenPhaseShadowRunner(
        config=config,
        interval_seconds=1,
        max_cycles=1,
        state_path=tmp_path / "state.json",
        latest_output_path=tmp_path / "latest.json",
        history_output_path=tmp_path / "history.jsonl",
        summary_output_path=tmp_path / "summary.json",
        snapshot_fetcher=_snapshot_fetcher,
        evaluator=_evaluator,
    )

    summary = runner.run_once()

    assert summary["markets_total"] == 2
    assert summary["markets_evaluated"] == 2
    assert summary["tradable_count"] == 1
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "history.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "state.json").exists()

    state = json.loads((tmp_path / "state.json").read_text())
    assert state["tradable_count"] == 1
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["tradable_count"] == 1
    assert payload["reason_counts"]["raw_gap_non_positive"] == 1
    assert payload["median_open_phase_age_hours"] == 1.5
    assert payload["by_horizon"]["market_open"]["markets_evaluated"] == 2
    assert payload["by_city_horizon"]["Seoul:market_open"]["tradable_count"] == 1
    assert payload["gate_decision"] == "INCONCLUSIVE"


def test_summarize_open_phase_history_counts_positive_raw_and_edge(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    rows = [
        OpenPhaseObservation(
            observed_at=datetime(2026, 3, 23, 3, 0, tzinfo=UTC),
            market_id="m1",
            city="Seoul",
            question="q1",
            target_local_date=datetime(2026, 3, 23, tzinfo=UTC).date(),
            decision_horizon="market_open",
            reason="raw_gap_non_positive",
            open_phase_age_hours=1.0,
            raw_gap=0.02,
            after_cost_edge=-0.01,
        ),
        OpenPhaseObservation(
            observed_at=datetime(2026, 3, 23, 3, 1, tzinfo=UTC),
            market_id="m2",
            city="London",
            question="q2",
            target_local_date=datetime(2026, 3, 23, tzinfo=UTC).date(),
            decision_horizon="market_open",
            reason="tradable",
            open_phase_age_hours=2.0,
            raw_gap=0.03,
            after_cost_edge=0.01,
            spread=0.02,
            visible_liquidity=500.0,
        ),
    ]
    with history_path.open("a") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")

    summary = summarize_open_phase_history(history_path)

    assert summary["cycles"] == 2
    assert summary["raw_gap_positive_count"] == 2
    assert summary["after_cost_edge_positive_count"] == 1
    assert summary["tradable_count"] == 1
    assert summary["median_open_phase_age_hours"] == 1.5
    assert summary["by_horizon"]["market_open"]["gate_decision"] == "INCONCLUSIVE"
    assert summary["gate_decision"] == "INCONCLUSIVE"
