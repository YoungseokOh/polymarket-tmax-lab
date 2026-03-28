from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pmtmax.config.settings import RepoConfig
from pmtmax.execution.opportunity_shadow import (
    OpportunityShadowRunner,
    select_shadow_horizon,
    summarize_opportunity_history,
)
from pmtmax.markets.repository import bundled_market_snapshots
from pmtmax.storage.schemas import OpportunityObservation


def test_select_shadow_horizon_supports_today_and_tomorrow() -> None:
    snapshot = bundled_market_snapshots(["Seoul"])[0]
    assert snapshot.spec is not None
    spec = snapshot.spec
    now_utc = datetime(2026, 3, 23, 3, 0, tzinfo=UTC)

    assert select_shadow_horizon(
        spec.model_copy(update={"target_local_date": datetime(2026, 3, 23, tzinfo=UTC).date()}),
        now_utc=now_utc,
    ) == "morning_of"
    assert select_shadow_horizon(
        spec.model_copy(update={"target_local_date": datetime(2026, 3, 24, tzinfo=UTC).date()}),
        now_utc=now_utc,
    ) == "previous_evening"
    assert select_shadow_horizon(
        spec.model_copy(update={"target_local_date": datetime(2026, 3, 25, tzinfo=UTC).date()}),
        now_utc=now_utc,
    ) is None


def test_opportunity_shadow_runner_writes_latest_history_summary_and_state(tmp_path: Path) -> None:
    config = RepoConfig()

    def _snapshot_fetcher():
        return ["m1", "m2"]

    def _evaluator(snapshots, observed_at):
        assert snapshots == ["m1", "m2"]
        return [
            OpportunityObservation(
                observed_at=observed_at,
                market_id="m1",
                city="Seoul",
                question="Highest temperature in Seoul on March 23?",
                target_local_date=datetime(2026, 3, 23, tzinfo=UTC).date(),
                decision_horizon="morning_of",
                reason="tradable",
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
            OpportunityObservation(
                observed_at=observed_at,
                market_id="m2",
                city="London",
                question="Highest temperature in London on March 23?",
                target_local_date=datetime(2026, 3, 23, tzinfo=UTC).date(),
                decision_horizon="morning_of",
                reason="no_positive_edge",
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

    runner = OpportunityShadowRunner(
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
    assert payload["reason_counts"]["no_positive_edge"] == 1
    assert payload["by_city"]["Seoul"]["tradable_count"] == 1
    assert payload["by_horizon"]["morning_of"]["markets_evaluated"] == 2
    assert payload["by_city_horizon"]["Seoul:morning_of"]["tradable_count"] == 1
    assert payload["gate_decision"] == "INCONCLUSIVE"


def test_summarize_opportunity_history_counts_positive_raw_and_edge(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    rows = [
        OpportunityObservation(
            observed_at=datetime(2026, 3, 23, 3, 0, tzinfo=UTC),
            market_id="m1",
            city="Seoul",
            question="q1",
            target_local_date=datetime(2026, 3, 23, tzinfo=UTC).date(),
            decision_horizon="morning_of",
            reason="no_positive_edge",
            raw_gap=0.02,
            after_cost_edge=-0.01,
        ),
        OpportunityObservation(
            observed_at=datetime(2026, 3, 23, 3, 1, tzinfo=UTC),
            market_id="m2",
            city="London",
            question="q2",
            target_local_date=datetime(2026, 3, 23, tzinfo=UTC).date(),
            decision_horizon="morning_of",
            reason="tradable",
            raw_gap=0.03,
            after_cost_edge=0.01,
            spread=0.02,
            visible_liquidity=500.0,
        ),
    ]
    with history_path.open("a") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")

    summary = summarize_opportunity_history(history_path)

    assert summary["cycles"] == 2
    assert summary["raw_gap_positive_count"] == 2
    assert summary["after_cost_edge_positive_count"] == 1
    assert summary["tradable_count"] == 1
    assert summary["by_horizon"]["morning_of"]["gate_decision"] == "INCONCLUSIVE"
    assert summary["gate_decision"] == "INCONCLUSIVE"
