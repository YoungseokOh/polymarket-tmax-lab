from __future__ import annotations

import json
from pathlib import Path

from pmtmax.config.settings import RepoConfig
from pmtmax.monitoring.station_dashboard import (
    StationDashboardRunner,
    build_station_dashboard,
    render_station_dashboard_html,
)


def test_build_station_dashboard_groups_panels_and_breakdowns() -> None:
    dashboard = build_station_dashboard(
        opportunity_rows=[
            {
                "market_id": "m-seoul-1",
                "city": "Seoul",
                "target_local_date": "2026-04-05",
                "decision_horizon": "morning_of",
                "outcome_label": "11°C",
                "edge": 0.023,
                "reason": "tradable",
                "best_ask": 0.89,
            }
        ],
        observation_rows=[
            {
                "city": "Taipei",
                "target_local_date": "2026-04-05",
                "decision_horizon": "morning_of",
                "queue_state": "tradable",
                "outcome_label": "21°C",
                "edge": 0.011,
                "source_family": "official_intraday",
                "observation_source": "cwa_codis_report_month",
            }
        ],
        queue_rows=[
            {
                "city": "Taipei",
                "target_local_date": "2026-04-05",
                "decision_horizon": "morning_of",
                "queue_state": "tradable",
                "outcome_label": "21°C",
                "edge": 0.011,
                "source_family": "official_intraday",
                "observation_source": "cwa_codis_report_month",
            }
        ],
        open_phase_rows=[
            {
                "city": "London",
                "target_local_date": "2026-04-05",
                "decision_horizon": "market_open",
                "outcome_label": "14°C",
                "edge": 0.018,
                "reason": "tradable",
                "open_phase_age_hours": 2.5,
            }
        ],
        revenue_gate_summary={
            "decision": "GO",
            "decision_reason": "benchmark_go_with_live_path_confirmation",
            "eligible_for_live_pilot": True,
            "required_model_alias": "champion",
            "observation_source_breakdown": {
                "official_intraday": {
                    "markets_evaluated": 1,
                    "tradable_count": 1,
                    "manual_review_count": 0,
                    "after_cost_edge_positive_count": 1,
                    "gate_decision": "GO",
                }
            },
        },
        observation_summary={
            "gate_decision": "GO",
            "gate_reason": "repeated_after_cost_positive_edges",
            "by_source_family": {
                "official_intraday": {
                    "markets_evaluated": 1,
                    "tradable_count": 1,
                    "manual_review_count": 0,
                    "after_cost_edge_positive_count": 1,
                    "gate_decision": "GO",
                }
            },
            "by_observation_source": {
                "cwa_codis_report_month": {
                    "markets_evaluated": 1,
                    "tradable_count": 1,
                    "manual_review_count": 0,
                    "after_cost_edge_positive_count": 1,
                    "gate_decision": "GO",
                }
            },
            "top_after_cost_edges": [
                {"city": "Taipei", "observation_source": "cwa_codis_report_month", "after_cost_edge": 0.011}
            ],
            "top_price_vs_observation_gaps": [
                {"city": "Taipei", "observation_source": "cwa_codis_report_month", "price_vs_observation_gap": 0.16}
            ],
        },
        open_phase_summary={"gate_decision": "INCONCLUSIVE", "gate_reason": "single_cycle_only"},
        watchlist_playbook={
            "playbook": [
                {
                    "tier": "A",
                    "name": "fee_sensitive_watchlist",
                    "cities": ["Seoul"],
                    "evidence": [
                        {
                            "city": "Seoul",
                            "market_id": "m-seoul-1",
                            "target_local_date": "2026-04-05",
                            "decision_horizon": "morning_of",
                            "outcome_label": "11°C",
                            "watch_rule_threshold_ask": 0.901,
                        }
                    ],
                }
            ],
            "next_actions": ["Keep fee-sensitive cities on the dashboard."],
        },
    )

    assert dashboard["overview"]["revenue_gate_decision"] == "GO"
    assert dashboard["overview"]["queue_size"] == 1
    assert dashboard["overview"]["watchlist_alert_count"] == 1
    assert dashboard["discovery_panel"]["top_markets"][0]["city"] == "London"
    assert dashboard["observation_panel"]["source_family_breakdown"][0]["name"] == "official_intraday"
    assert dashboard["execution_panel"]["queue_preview"][0]["observation_source"] == "cwa_codis_report_month"
    assert dashboard["watchlist_panel"]["triggered_alerts"][0]["city"] == "Seoul"


def test_station_dashboard_runner_writes_json_html_and_state(tmp_path: Path) -> None:
    runner = StationDashboardRunner(
        config=RepoConfig(),
        interval_seconds=1,
        max_cycles=1,
        state_path=tmp_path / "state.json",
        json_output_path=tmp_path / "dashboard.json",
        html_output_path=tmp_path / "dashboard.html",
        data_loader=lambda: {
            "opportunity_rows": [],
            "observation_rows": [],
            "queue_rows": [],
            "open_phase_rows": [],
            "revenue_gate_summary": {"decision": "INCONCLUSIVE", "decision_reason": "missing_summary"},
            "observation_summary": None,
            "open_phase_summary": None,
            "watchlist_playbook": None,
        },
    )

    summary = runner.run_once()

    assert summary["cycle"] == 1
    assert summary["revenue_gate_decision"] == "INCONCLUSIVE"
    assert (tmp_path / "dashboard.json").exists()
    assert (tmp_path / "dashboard.html").exists()
    assert (tmp_path / "state.json").exists()
    payload = json.loads((tmp_path / "dashboard.json").read_text())
    assert payload["overview"]["queue_size"] == 0
    assert payload["overview"]["watchlist_alert_count"] == 0
    html_output = (tmp_path / "dashboard.html").read_text()
    assert "PMTMAX Station Dashboard" in html_output
    assert "Discovery" in html_output


def test_render_station_dashboard_html_contains_panel_tables() -> None:
    html_output = render_station_dashboard_html(
        {
            "generated_at": "2026-04-05T00:00:00+00:00",
            "overview": {"revenue_gate_decision": "GO", "queue_size": 2, "observation_tradable_count": 1, "opportunity_tradable_count": 1, "manual_review_count": 1, "open_phase_count": 1, "revenue_gate_reason": "x"},
            "discovery_panel": {"gate_decision": "INCONCLUSIVE", "gate_reason": "x", "open_phase_count": 1, "top_markets": []},
            "observation_panel": {"gate_decision": "GO", "gate_reason": "x", "queue_counts": {"tradable": 1}, "source_family_breakdown": [], "observation_source_breakdown": [], "top_candidates": [], "top_after_cost_edges": [], "top_price_vs_observation_gaps": []},
            "execution_panel": {"required_model_alias": "champion", "eligible_for_live_pilot": False, "queue_preview": [], "top_opportunities": [], "observation_source_breakdown": []},
            "watchlist_panel": {"tier_a_cities": ["Taipei"], "tier_b_cities": [], "triggered_alert_count": 1, "triggered_alerts": [], "top_rules": []},
        }
    )

    assert "Observation Source Families" in html_output
    assert "Execution Opportunities" in html_output
    assert "Watchlist Alerts" in html_output
