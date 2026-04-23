from __future__ import annotations

from pmtmax.backtest.recent_core_benchmark import summarize_recent_core_profitability


def _city_row(*, panel_rows: int, ok_rows: int, real_trades: float, real_pnl: float, proxy_trades: float, proxy_pnl: float) -> dict[str, object]:
    return {
        "panel_summary": {"rows": panel_rows, "coverage": {"ok": ok_rows, "missing": panel_rows - ok_rows}},
        "real_history_metrics": {
            "num_trades": real_trades,
            "pnl": real_pnl,
            "priced_decision_rows": float(ok_rows),
            "skipped_missing_price": 0.0,
            "skipped_stale_price": 0.0,
            "skipped_non_positive_edge": 0.0,
            "hit_rate": 0.55,
            "avg_edge": 0.2,
            "avg_price_age_seconds": 900.0,
        },
        "quote_proxy_metrics": {
            "num_trades": proxy_trades,
            "pnl": proxy_pnl,
            "priced_decision_rows": float(ok_rows),
            "skipped_missing_price": 0.0,
            "skipped_stale_price": 0.0,
            "skipped_non_positive_edge": 0.0,
            "hit_rate": 0.52,
            "avg_edge": 0.18,
        },
        "policy_real_history_metrics": {
            "num_trades": real_trades,
            "pnl": real_pnl,
            "hit_rate": 0.55,
            "avg_edge": 0.2,
        },
        "policy_quote_proxy_metrics": {
            "num_trades": proxy_trades,
            "pnl": proxy_pnl,
            "hit_rate": 0.52,
            "avg_edge": 0.18,
        },
    }


def test_summarize_recent_core_profitability_returns_go_when_all_city_gates_pass() -> None:
    summary = summarize_recent_core_profitability(
        {
            "Seoul": _city_row(panel_rows=80, ok_rows=60, real_trades=48.0, real_pnl=14.0, proxy_trades=45.0, proxy_pnl=8.0),
            "NYC": _city_row(panel_rows=85, ok_rows=62, real_trades=44.0, real_pnl=9.0, proxy_trades=43.0, proxy_pnl=6.0),
            "London": _city_row(panel_rows=78, ok_rows=55, real_trades=41.0, real_pnl=7.0, proxy_trades=42.0, proxy_pnl=5.0),
        }
    )

    assert summary["decision"] == "GO"
    assert summary["decision_reason"] == "positive_policy_pnl_real_history_with_city_gates"
    assert summary["sample_adequacy"]["passes"] is True
    assert summary["aggregate_real_history_metrics"]["priced_decision_rows"] == 177.0
    assert summary["aggregate_panel_coverage"]["ok_ratio"] > 0.20
    assert summary["city_gate_details"]["Seoul"]["passes_panel_coverage"] is True
    assert summary["city_gate_details"]["Seoul"]["passes"] is True
    assert summary["city_gate_details"]["NYC"]["passes"] is True
    assert summary["city_gate_details"]["London"]["passes"] is True


def test_summarize_recent_core_profitability_is_inconclusive_when_city_trade_count_is_too_small() -> None:
    summary = summarize_recent_core_profitability(
        {
            "Seoul": _city_row(panel_rows=80, ok_rows=60, real_trades=48.0, real_pnl=14.0, proxy_trades=45.0, proxy_pnl=8.0),
            "NYC": _city_row(panel_rows=85, ok_rows=62, real_trades=39.0, real_pnl=9.0, proxy_trades=43.0, proxy_pnl=6.0),
            "London": _city_row(panel_rows=78, ok_rows=55, real_trades=41.0, real_pnl=7.0, proxy_trades=42.0, proxy_pnl=5.0),
        }
    )

    assert summary["decision"] == "INCONCLUSIVE"
    assert summary["decision_reason"] == "city_real_history_sample_inadequate"
    assert summary["sample_adequacy"]["passes"] is True
    assert summary["city_gate_details"]["NYC"]["passes_trade_count"] is False


def test_summarize_recent_core_profitability_is_inconclusive_when_one_city_panel_coverage_is_too_thin() -> None:
    summary = summarize_recent_core_profitability(
        {
            "Seoul": _city_row(panel_rows=100, ok_rows=80, real_trades=48.0, real_pnl=14.0, proxy_trades=45.0, proxy_pnl=8.0),
            "NYC": _city_row(panel_rows=100, ok_rows=75, real_trades=44.0, real_pnl=9.0, proxy_trades=43.0, proxy_pnl=6.0),
            "London": _city_row(panel_rows=100, ok_rows=10, real_trades=41.0, real_pnl=7.0, proxy_trades=42.0, proxy_pnl=5.0),
        }
    )

    assert summary["decision"] == "INCONCLUSIVE"
    assert summary["decision_reason"] == "city_panel_coverage_inadequate"
    assert summary["sample_adequacy"]["passes"] is True
    assert summary["city_gate_details"]["London"]["passes_panel_coverage"] is False
    assert summary["city_gate_details"]["London"]["passes"] is False
    assert summary["reduced_core_candidate"]["decision"] == "GO"
    assert summary["reduced_core_candidate"]["publish_eligible"] is False
    assert summary["reduced_core_candidate"]["coverage_eligible_cities"] == ["Seoul", "NYC"]
    assert summary["reduced_core_candidate"]["coverage_excluded_cities"]["London"]["reason"] == "panel_coverage_below_threshold"


def test_summarize_recent_core_profitability_returns_no_go_when_one_city_real_pnl_is_negative() -> None:
    summary = summarize_recent_core_profitability(
        {
            "Seoul": _city_row(panel_rows=80, ok_rows=60, real_trades=48.0, real_pnl=14.0, proxy_trades=45.0, proxy_pnl=8.0),
            "NYC": _city_row(panel_rows=85, ok_rows=62, real_trades=44.0, real_pnl=-1.0, proxy_trades=43.0, proxy_pnl=6.0),
            "London": _city_row(panel_rows=78, ok_rows=55, real_trades=41.0, real_pnl=7.0, proxy_trades=42.0, proxy_pnl=5.0),
        }
    )

    assert summary["decision"] == "NO_GO"
    assert summary["decision_reason"] == "city_real_history_negative_pnl"
    assert summary["city_gate_details"]["NYC"]["passes_non_negative_pnl"] is False


def test_summarize_recent_core_profitability_is_inconclusive_when_panel_coverage_is_too_low() -> None:
    summary = summarize_recent_core_profitability(
        {
            "Seoul": _city_row(panel_rows=80, ok_rows=10, real_trades=48.0, real_pnl=14.0, proxy_trades=45.0, proxy_pnl=8.0),
            "NYC": _city_row(panel_rows=85, ok_rows=9, real_trades=44.0, real_pnl=9.0, proxy_trades=43.0, proxy_pnl=6.0),
            "London": _city_row(panel_rows=78, ok_rows=8, real_trades=41.0, real_pnl=7.0, proxy_trades=42.0, proxy_pnl=5.0),
        }
    )

    assert summary["decision"] == "INCONCLUSIVE"
    assert summary["decision_reason"] == "insufficient_sample"
    assert summary["sample_adequacy"]["aggregate_panel_ok_ratio"] < 0.20
