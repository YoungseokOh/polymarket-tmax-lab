from __future__ import annotations

import pandas as pd

from pmtmax.backtest.recent_core_diagnostics import summarize_recent_core_diagnostics


def test_summarize_recent_core_diagnostics_reports_coverage_and_negative_policy_slices() -> None:
    panel = pd.DataFrame(
        [
            {"city": "Seoul", "decision_horizon": "morning_of", "coverage_status": "ok"},
            {"city": "Seoul", "decision_horizon": "morning_of", "coverage_status": "missing"},
            {"city": "NYC", "decision_horizon": "previous_evening", "coverage_status": "missing"},
            {"city": "NYC", "decision_horizon": "previous_evening", "coverage_status": "missing"},
            {"city": "London", "decision_horizon": "previous_evening", "coverage_status": "ok"},
            {"city": "London", "decision_horizon": "previous_evening", "coverage_status": "missing"},
        ]
    )
    quote_proxy = pd.DataFrame(
        [
            {"city": "Seoul", "decision_horizon": "morning_of", "realized_pnl": 2.0},
            {"city": "NYC", "decision_horizon": "previous_evening", "realized_pnl": -1.0},
            {"city": "London", "decision_horizon": "previous_evening", "realized_pnl": -2.0},
            {"city": "London", "decision_horizon": "market_open", "realized_pnl": 10.0},
        ]
    )
    real_history = pd.DataFrame(
        [
            {"city": "Seoul", "decision_horizon": "morning_of", "realized_pnl": 3.0},
            {"city": "NYC", "decision_horizon": "previous_evening", "realized_pnl": 1.5},
            {"city": "London", "decision_horizon": "previous_evening", "realized_pnl": -0.5},
        ]
    )

    summary = summarize_recent_core_diagnostics(
        panel=panel,
        trade_logs_by_source={
            "quote_proxy": quote_proxy,
            "real_history": real_history,
        },
        horizon_policy={
            "Seoul": {"morning_of"},
            "NYC": {"previous_evening"},
            "London": {"previous_evening"},
        },
    )

    assert summary["coverage_bottlenecks"][0]["city"] == "NYC"
    assert summary["coverage_bottlenecks"][0]["ok_ratio"] == 0.0
    assert summary["sources"]["quote_proxy"]["policy_aggregate"]["trades"] == 3
    assert summary["sources"]["quote_proxy"]["negative_policy_cities"][0]["city"] == "London"
    assert summary["sources"]["real_history"]["policy_aggregate"]["pnl"] == 4.0
    assert any(row["type"] == "coverage" for row in summary["recommendations"])
    assert any(
        row["type"] == "policy_pnl" and row["city"] == "London"
        for row in summary["recommendations"]
    )
