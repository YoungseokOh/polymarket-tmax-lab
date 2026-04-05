from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pmtmax.config.settings import RepoConfig
from pmtmax.monitoring.observation_station import (
    ObservationShadowRunner,
    summarize_observation_history,
)
from pmtmax.storage.schemas import ObservationOpportunity


def test_observation_shadow_runner_writes_latest_history_summary_and_queue(tmp_path: Path) -> None:
    config = RepoConfig()

    def _snapshot_fetcher():
        return ["m1", "m2"]

    def _evaluator(snapshots, observed_at):
        assert snapshots == ["m1", "m2"]
        return [
            ObservationOpportunity(
                observed_at=observed_at,
                market_id="m1",
                city="Seoul",
                question="Highest temperature in Seoul on March 23?",
                target_local_date=datetime(2026, 3, 23, tzinfo=UTC).date(),
                decision_horizon="morning_of",
                reason="tradable",
                queue_state="tradable",
                source_family="official_intraday",
                observation_source="cwa_codis_report_month",
                truth_track="exact_public",
                candidate_tier="exact_public_live",
                observed_temp_c=11.2,
                observed_temp_market_unit=11.2,
                observation_observed_at=observed_at - timedelta(minutes=12),
                observation_freshness_minutes=12.0,
                observation_override_mass=0.22,
                impossible_outcome_count=2,
                impossible_price_mass=0.16,
                price_vs_observation_gap=0.16,
                outcome_label="11°C",
                token_id="tok-1",
                fair_probability=0.55,
                best_bid=0.51,
                best_ask=0.53,
                spread=0.02,
                visible_liquidity=500.0,
                fee_estimate=0.0106,
                slippage_estimate=0.01,
                raw_gap=0.02,
                after_cost_edge=0.004,
                approval_required=True,
                live_eligible=True,
                manual_approval_token="token-1",
                approval_expires_at=observed_at + timedelta(minutes=20),
            ),
            ObservationOpportunity(
                observed_at=observed_at,
                market_id="m2",
                city="London",
                question="Highest temperature in London on March 23?",
                target_local_date=datetime(2026, 3, 23, tzinfo=UTC).date(),
                decision_horizon="morning_of",
                reason="after_cost_positive_but_spread_too_wide",
                queue_state="manual_review",
                source_family="aviation",
                observation_source="aviationweather_metar",
                truth_track="research_public",
                candidate_tier="research_public_live",
                observed_temp_c=14.0,
                observed_temp_market_unit=14.0,
                observation_observed_at=observed_at - timedelta(minutes=25),
                observation_freshness_minutes=25.0,
                observation_override_mass=0.11,
                impossible_outcome_count=1,
                impossible_price_mass=0.07,
                price_vs_observation_gap=0.07,
                outcome_label="14°C",
                token_id="tok-2",
                fair_probability=0.31,
                best_bid=0.30,
                best_ask=0.34,
                spread=0.04,
                visible_liquidity=300.0,
                fee_estimate=0.0068,
                slippage_estimate=0.0201,
                raw_gap=0.01,
                after_cost_edge=0.001,
                approval_required=True,
                live_eligible=True,
                manual_approval_token="token-2",
                approval_expires_at=observed_at + timedelta(minutes=20),
                risk_flags=["research_public", "wide_spread"],
            ),
        ]

    runner = ObservationShadowRunner(
        config=config,
        interval_seconds=1,
        max_cycles=1,
        state_path=tmp_path / "state.json",
        latest_output_path=tmp_path / "latest.json",
        history_output_path=tmp_path / "history.jsonl",
        summary_output_path=tmp_path / "summary.json",
        alerts_output_path=tmp_path / "alerts.json",
        queue_output_path=tmp_path / "queue.json",
        snapshot_fetcher=_snapshot_fetcher,
        evaluator=_evaluator,
    )

    summary = runner.run_once()

    assert summary["markets_total"] == 2
    assert summary["markets_evaluated"] == 2
    assert summary["tradable_count"] == 1
    assert summary["manual_review_count"] == 1
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "history.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "state.json").exists()
    assert (tmp_path / "alerts.json").exists()
    assert (tmp_path / "queue.json").exists()

    queue_payload = json.loads((tmp_path / "queue.json").read_text())
    assert len(queue_payload) == 2
    assert queue_payload[0]["manual_approval_token"] == "token-1"
    summary_payload = json.loads((tmp_path / "summary.json").read_text())
    assert summary_payload["queue_state_counts"]["tradable"] == 1
    assert summary_payload["queue_state_counts"]["manual_review"] == 1
    assert summary_payload["by_candidate_tier"]["exact_public_live"]["tradable_count"] == 1
    assert summary_payload["by_truth_track"]["research_public"]["manual_review_count"] == 1
    assert summary_payload["by_source_family"]["official_intraday"]["tradable_count"] == 1
    assert summary_payload["by_observation_source"]["aviationweather_metar"]["manual_review_count"] == 1


def test_summarize_observation_history_counts_tiers_and_price_mass(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    rows = [
        ObservationOpportunity(
            observed_at=datetime(2026, 3, 23, 3, 0, tzinfo=UTC),
            market_id="m1",
            city="Seoul",
            question="q1",
            target_local_date=datetime(2026, 3, 23, tzinfo=UTC).date(),
            decision_horizon="morning_of",
            reason="tradable",
            queue_state="tradable",
            source_family="official_intraday",
            observation_source="cwa_codis_report_month",
            truth_track="exact_public",
            candidate_tier="exact_public_live",
            observation_freshness_minutes=10.0,
            impossible_price_mass=0.21,
            raw_gap=0.03,
            after_cost_edge=0.01,
        ),
        ObservationOpportunity(
            observed_at=datetime(2026, 3, 23, 3, 1, tzinfo=UTC),
            market_id="m2",
            city="London",
            question="q2",
            target_local_date=datetime(2026, 3, 23, tzinfo=UTC).date(),
            decision_horizon="market_open",
            reason="after_cost_positive_but_spread_too_wide",
            queue_state="manual_review",
            source_family="aviation",
            observation_source="aviationweather_metar",
            truth_track="research_public",
            candidate_tier="research_public_live",
            observation_freshness_minutes=32.0,
            impossible_price_mass=0.08,
            raw_gap=0.01,
            after_cost_edge=0.001,
        ),
    ]
    with history_path.open("a") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")

    summary = summarize_observation_history(history_path)

    assert summary["cycles"] == 2
    assert summary["tradable_count"] == 1
    assert summary["manual_review_count"] == 1
    assert summary["best_impossible_price_mass"] == 0.21
    assert summary["by_candidate_tier"]["research_public_live"]["manual_review_count"] == 1
    assert summary["by_horizon"]["market_open"]["manual_review_count"] == 1
    assert summary["by_source_family"]["official_intraday"]["after_cost_edge_positive_count"] == 1
    assert summary["by_observation_source"]["aviationweather_metar"]["manual_review_count"] == 1
    assert summary["top_after_cost_edges"][0]["observation_source"] == "cwa_codis_report_month"
    assert summary["gate_decision"] == "GO"
