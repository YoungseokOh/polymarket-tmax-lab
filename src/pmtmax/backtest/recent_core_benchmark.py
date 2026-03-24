"""Recent-core benchmark aggregation and profitability classification."""

from __future__ import annotations

from collections import Counter

DEFAULT_MIN_POLICY_TRADES = 20.0
DEFAULT_MIN_PRICED_DECISION_ROWS = 40.0


def _as_float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _weighted_average(rows: list[dict[str, object]], *, value_key: str, weight_key: str) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for row in rows:
        weight = _as_float(row.get(weight_key))
        if weight <= 0:
            continue
        weighted_sum += _as_float(row.get(value_key)) * weight
        total_weight += weight
    if total_weight <= 0:
        return 0.0
    return weighted_sum / total_weight


def aggregate_backtest_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    """Aggregate real-history or quote-proxy backtest metrics across cities."""

    aggregate = {
        "num_trades": sum(_as_float(row.get("num_trades")) for row in rows),
        "pnl": sum(_as_float(row.get("pnl")) for row in rows),
        "priced_decision_rows": sum(_as_float(row.get("priced_decision_rows")) for row in rows),
        "skipped_missing_price": sum(_as_float(row.get("skipped_missing_price")) for row in rows),
        "skipped_stale_price": sum(_as_float(row.get("skipped_stale_price")) for row in rows),
        "skipped_non_positive_edge": sum(_as_float(row.get("skipped_non_positive_edge")) for row in rows),
        "hit_rate": _weighted_average(rows, value_key="hit_rate", weight_key="num_trades"),
        "avg_edge": _weighted_average(rows, value_key="avg_edge", weight_key="num_trades"),
    }
    if any("avg_price_age_seconds" in row for row in rows):
        aggregate["avg_price_age_seconds"] = _weighted_average(
            rows,
            value_key="avg_price_age_seconds",
            weight_key="priced_decision_rows",
        )
    return aggregate


def aggregate_policy_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    """Aggregate policy-filtered trade metrics across cities."""

    return {
        "num_trades": sum(_as_float(row.get("num_trades")) for row in rows),
        "pnl": sum(_as_float(row.get("pnl")) for row in rows),
        "hit_rate": _weighted_average(rows, value_key="hit_rate", weight_key="num_trades"),
        "avg_edge": _weighted_average(rows, value_key="avg_edge", weight_key="num_trades"),
    }


def aggregate_panel_coverage(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate panel coverage counts across recent-core cities."""

    coverage = Counter()
    total_rows = 0
    for row in rows:
        total_rows += int(_as_float(row.get("rows")))
        for status, count in dict(row.get("coverage", {})).items():
            coverage[str(status)] += int(_as_float(count))
    ok_rows = coverage.get("ok", 0)
    return {
        "rows": total_rows,
        "coverage": dict(sorted(coverage.items())),
        "ok_ratio": float(ok_rows / total_rows) if total_rows > 0 else 0.0,
    }


def classify_profitability(
    *,
    aggregate_real_history_metrics: dict[str, object],
    aggregate_policy_real_history_metrics: dict[str, object],
    aggregate_policy_quote_proxy_metrics: dict[str, object],
    min_policy_trades: float = DEFAULT_MIN_POLICY_TRADES,
    min_priced_decision_rows: float = DEFAULT_MIN_PRICED_DECISION_ROWS,
) -> dict[str, object]:
    """Classify profitability using policy-filtered PnL and coverage gates."""

    policy_real_trades = _as_float(aggregate_policy_real_history_metrics.get("num_trades"))
    priced_decision_rows = _as_float(aggregate_real_history_metrics.get("priced_decision_rows"))
    sample_adequacy = {
        "min_policy_trades": float(min_policy_trades),
        "min_priced_decision_rows": float(min_priced_decision_rows),
        "policy_real_history_num_trades": policy_real_trades,
        "aggregate_real_history_priced_decision_rows": priced_decision_rows,
        "passes": bool(policy_real_trades >= min_policy_trades and priced_decision_rows >= min_priced_decision_rows),
    }
    if not sample_adequacy["passes"]:
        return {
            "decision": "INCONCLUSIVE",
            "decision_reason": "insufficient_sample",
            "sample_adequacy": sample_adequacy,
        }

    real_pnl = _as_float(aggregate_policy_real_history_metrics.get("pnl"))
    proxy_pnl = _as_float(aggregate_policy_quote_proxy_metrics.get("pnl"))
    if real_pnl > 0 and proxy_pnl > 0:
        decision = "GO"
        decision_reason = "positive_policy_pnl_in_real_and_proxy"
    elif real_pnl <= 0 and proxy_pnl <= 0:
        decision = "NO_GO"
        decision_reason = "non_positive_policy_pnl_in_real_and_proxy"
    else:
        decision = "INCONCLUSIVE"
        decision_reason = "pricing_paths_disagree"
    return {
        "decision": decision,
        "decision_reason": decision_reason,
        "sample_adequacy": sample_adequacy,
    }


def summarize_recent_core_profitability(
    cities: dict[str, dict[str, object]],
    *,
    min_policy_trades: float = DEFAULT_MIN_POLICY_TRADES,
    min_priced_decision_rows: float = DEFAULT_MIN_PRICED_DECISION_ROWS,
) -> dict[str, object]:
    """Build top-level aggregate profitability fields for the benchmark summary."""

    city_rows = list(cities.values())
    aggregate_real_history_metrics = aggregate_backtest_metrics(
        [dict(row.get("real_history_metrics", {})) for row in city_rows]
    )
    aggregate_quote_proxy_metrics = aggregate_backtest_metrics(
        [dict(row.get("quote_proxy_metrics", {})) for row in city_rows]
    )
    aggregate_policy_real_history_metrics = aggregate_policy_metrics(
        [dict(row.get("policy_real_history_metrics", {})) for row in city_rows]
    )
    aggregate_policy_quote_proxy_metrics = aggregate_policy_metrics(
        [dict(row.get("policy_quote_proxy_metrics", {})) for row in city_rows]
    )
    aggregate_panel = aggregate_panel_coverage([dict(row.get("panel_summary", {})) for row in city_rows])
    classification = classify_profitability(
        aggregate_real_history_metrics=aggregate_real_history_metrics,
        aggregate_policy_real_history_metrics=aggregate_policy_real_history_metrics,
        aggregate_policy_quote_proxy_metrics=aggregate_policy_quote_proxy_metrics,
        min_policy_trades=min_policy_trades,
        min_priced_decision_rows=min_priced_decision_rows,
    )
    return {
        "aggregate_real_history_metrics": aggregate_real_history_metrics,
        "aggregate_quote_proxy_metrics": aggregate_quote_proxy_metrics,
        "aggregate_policy_real_history_metrics": aggregate_policy_real_history_metrics,
        "aggregate_policy_quote_proxy_metrics": aggregate_policy_quote_proxy_metrics,
        "aggregate_panel_coverage": aggregate_panel,
        **classification,
    }
