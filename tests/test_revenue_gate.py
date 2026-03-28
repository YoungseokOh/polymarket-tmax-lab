from __future__ import annotations

from pmtmax.execution.revenue_gate import build_revenue_gate_report, classify_path_viability


def test_classify_path_viability_returns_go_for_repeated_positive_after_cost_edges() -> None:
    decision = classify_path_viability(
        {
            "cycles": 3,
            "markets_evaluated": 6,
            "raw_gap_positive_count": 4,
            "after_cost_edge_positive_count": 2,
        }
    )

    assert decision == {
        "decision": "GO",
        "decision_reason": "repeated_after_cost_positive_edges",
    }


def test_classify_path_viability_returns_no_go_when_raw_gap_never_positive() -> None:
    decision = classify_path_viability(
        {
            "cycles": 4,
            "markets_evaluated": 8,
            "raw_gap_positive_count": 0,
            "after_cost_edge_positive_count": 0,
        }
    )

    assert decision == {
        "decision": "NO_GO",
        "decision_reason": "raw_gap_never_positive",
    }


def test_build_revenue_gate_report_requires_benchmark_and_one_live_path() -> None:
    report = build_revenue_gate_report(
        benchmark_summary={
            "decision": "GO",
            "decision_reason": "positive_policy_pnl_in_real_and_proxy",
            "aggregate_policy_real_history_metrics": {"num_trades": 25.0, "pnl": 30.0},
            "aggregate_policy_quote_proxy_metrics": {"num_trades": 25.0, "pnl": 18.0},
            "aggregate_panel_coverage": {"rows": 60, "coverage": {"ok": 50, "missing": 10}},
        },
        opportunity_summary={
            "cycles": 3,
            "markets_evaluated": 5,
            "raw_gap_positive_count": 2,
            "after_cost_edge_positive_count": 0,
        },
        open_phase_summary={
            "cycles": 2,
            "markets_evaluated": 4,
            "raw_gap_positive_count": 3,
            "after_cost_edge_positive_count": 2,
        },
    )

    assert report["decision"] == "GO"
    assert report["decision_reason"] == "benchmark_go_with_live_path_confirmation"
    assert report["eligible_for_live_pilot"] is True
    assert report["required_model_alias"] == "trading_champion"
    assert report["pilot_constraints"]["bankroll"] == 500.0
