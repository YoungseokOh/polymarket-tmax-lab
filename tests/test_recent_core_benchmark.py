from __future__ import annotations

from pmtmax.backtest.recent_core_benchmark import summarize_recent_core_profitability


def test_summarize_recent_core_profitability_returns_go_when_policy_pnl_is_positive_with_sample() -> None:
    summary = summarize_recent_core_profitability(
        {
            "Seoul": {
                "panel_summary": {"rows": 30, "coverage": {"ok": 20, "missing": 10}},
                "real_history_metrics": {
                    "num_trades": 12.0,
                    "pnl": 40.0,
                    "priced_decision_rows": 25.0,
                    "skipped_missing_price": 5.0,
                    "skipped_stale_price": 0.0,
                    "skipped_non_positive_edge": 1.0,
                    "hit_rate": 0.5,
                    "avg_edge": 0.2,
                    "avg_price_age_seconds": 1000.0,
                },
                "quote_proxy_metrics": {
                    "num_trades": 12.0,
                    "pnl": 25.0,
                    "priced_decision_rows": 25.0,
                    "skipped_missing_price": 5.0,
                    "skipped_stale_price": 0.0,
                    "skipped_non_positive_edge": 1.0,
                    "hit_rate": 0.5,
                    "avg_edge": 0.2,
                },
                "policy_real_history_metrics": {"num_trades": 12.0, "pnl": 30.0, "hit_rate": 0.5, "avg_edge": 0.2},
                "policy_quote_proxy_metrics": {"num_trades": 12.0, "pnl": 20.0, "hit_rate": 0.5, "avg_edge": 0.2},
            },
            "NYC": {
                "panel_summary": {"rows": 35, "coverage": {"ok": 25, "missing": 10}},
                "real_history_metrics": {
                    "num_trades": 10.0,
                    "pnl": 15.0,
                    "priced_decision_rows": 20.0,
                    "skipped_missing_price": 4.0,
                    "skipped_stale_price": 0.0,
                    "skipped_non_positive_edge": 2.0,
                    "hit_rate": 0.6,
                    "avg_edge": 0.3,
                    "avg_price_age_seconds": 2000.0,
                },
                "quote_proxy_metrics": {
                    "num_trades": 10.0,
                    "pnl": 18.0,
                    "priced_decision_rows": 20.0,
                    "skipped_missing_price": 4.0,
                    "skipped_stale_price": 0.0,
                    "skipped_non_positive_edge": 2.0,
                    "hit_rate": 0.6,
                    "avg_edge": 0.3,
                },
                "policy_real_history_metrics": {"num_trades": 10.0, "pnl": 12.0, "hit_rate": 0.6, "avg_edge": 0.3},
                "policy_quote_proxy_metrics": {"num_trades": 10.0, "pnl": 8.0, "hit_rate": 0.6, "avg_edge": 0.3},
            },
        }
    )

    assert summary["decision"] == "GO"
    assert summary["decision_reason"] == "positive_policy_pnl_in_real_and_proxy"
    assert summary["sample_adequacy"]["passes"] is True
    assert summary["aggregate_real_history_metrics"]["num_trades"] == 22.0
    assert summary["aggregate_real_history_metrics"]["priced_decision_rows"] == 45.0
    assert summary["aggregate_policy_real_history_metrics"]["pnl"] == 42.0
    assert summary["aggregate_panel_coverage"]["coverage"] == {"missing": 20, "ok": 45}


def test_summarize_recent_core_profitability_is_inconclusive_when_sample_is_too_small() -> None:
    summary = summarize_recent_core_profitability(
        {
            "London": {
                "panel_summary": {"rows": 20, "coverage": {"ok": 12, "missing": 8}},
                "real_history_metrics": {
                    "num_trades": 6.0,
                    "pnl": 10.0,
                    "priced_decision_rows": 18.0,
                    "skipped_missing_price": 2.0,
                    "skipped_stale_price": 0.0,
                    "skipped_non_positive_edge": 1.0,
                    "hit_rate": 0.5,
                    "avg_edge": 0.2,
                },
                "quote_proxy_metrics": {
                    "num_trades": 6.0,
                    "pnl": 11.0,
                    "priced_decision_rows": 18.0,
                    "skipped_missing_price": 2.0,
                    "skipped_stale_price": 0.0,
                    "skipped_non_positive_edge": 1.0,
                    "hit_rate": 0.5,
                    "avg_edge": 0.2,
                },
                "policy_real_history_metrics": {"num_trades": 6.0, "pnl": 7.0, "hit_rate": 0.5, "avg_edge": 0.2},
                "policy_quote_proxy_metrics": {"num_trades": 6.0, "pnl": 8.0, "hit_rate": 0.5, "avg_edge": 0.2},
            }
        }
    )

    assert summary["decision"] == "INCONCLUSIVE"
    assert summary["decision_reason"] == "insufficient_sample"
    assert summary["sample_adequacy"]["passes"] is False


def test_summarize_recent_core_profitability_is_inconclusive_when_pricing_paths_disagree() -> None:
    summary = summarize_recent_core_profitability(
        {
            "Seoul": {
                "panel_summary": {"rows": 50, "coverage": {"ok": 40, "missing": 10}},
                "real_history_metrics": {
                    "num_trades": 20.0,
                    "pnl": 5.0,
                    "priced_decision_rows": 50.0,
                    "skipped_missing_price": 0.0,
                    "skipped_stale_price": 0.0,
                    "skipped_non_positive_edge": 0.0,
                    "hit_rate": 0.55,
                    "avg_edge": 0.25,
                },
                "quote_proxy_metrics": {
                    "num_trades": 20.0,
                    "pnl": -3.0,
                    "priced_decision_rows": 50.0,
                    "skipped_missing_price": 0.0,
                    "skipped_stale_price": 0.0,
                    "skipped_non_positive_edge": 0.0,
                    "hit_rate": 0.45,
                    "avg_edge": 0.18,
                },
                "policy_real_history_metrics": {"num_trades": 20.0, "pnl": 6.0, "hit_rate": 0.55, "avg_edge": 0.25},
                "policy_quote_proxy_metrics": {"num_trades": 20.0, "pnl": -2.0, "hit_rate": 0.45, "avg_edge": 0.18},
            }
        }
    )

    assert summary["decision"] == "INCONCLUSIVE"
    assert summary["decision_reason"] == "pricing_paths_disagree"
    assert summary["sample_adequacy"]["passes"] is True
