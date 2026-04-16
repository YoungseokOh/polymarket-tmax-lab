"""Recent-core benchmark aggregation and profitability classification."""

from __future__ import annotations

from collections import Counter

DEFAULT_MIN_POLICY_TRADES = 20.0
DEFAULT_MIN_PRICED_DECISION_ROWS = 150.0
DEFAULT_MIN_PANEL_OK_RATIO = 0.20
DEFAULT_MIN_CITY_POLICY_TRADES = 40.0
DEFAULT_MIN_COVERAGE_ELIGIBLE_CITIES = 2
DEFAULT_REQUIRED_CITIES = ("Seoul", "NYC", "London")


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


def _panel_ok_ratio(panel_summary: dict[str, object]) -> float:
    total_rows = _as_float(panel_summary.get("rows"))
    ok_rows = _as_float(dict(panel_summary.get("coverage", {})).get("ok"))
    if total_rows <= 0:
        return 0.0
    return float(ok_rows / total_rows)


def _build_city_gate_details(
    cities: dict[str, dict[str, object]],
    *,
    required_cities: tuple[str, ...],
    min_city_policy_trades: float,
    min_panel_ok_ratio: float,
) -> dict[str, dict[str, object]]:
    city_gate_details: dict[str, dict[str, object]] = {}
    for city in required_cities:
        city_payload = dict(cities.get(city, {}))
        city_quote_proxy_metrics = dict(city_payload.get("policy_quote_proxy_metrics", {}))
        city_panel_summary = dict(city_payload.get("panel_summary", {}))
        city_num_trades = _as_float(city_quote_proxy_metrics.get("num_trades"))
        city_pnl = _as_float(city_quote_proxy_metrics.get("pnl"))
        city_panel_rows = _as_float(city_panel_summary.get("rows"))
        city_panel_ok_rows = _as_float(dict(city_panel_summary.get("coverage", {})).get("ok"))
        city_panel_ok_ratio = _panel_ok_ratio(city_panel_summary)
        city_gate_details[city] = {
            "required": True,
            "available": bool(city_payload),
            "min_quote_proxy_policy_trades": float(min_city_policy_trades),
            "min_panel_ok_ratio": float(min_panel_ok_ratio),
            "quote_proxy_policy_num_trades": city_num_trades,
            "quote_proxy_policy_pnl": city_pnl,
            "panel_rows": city_panel_rows,
            "panel_ok_rows": city_panel_ok_rows,
            "panel_ok_ratio": city_panel_ok_ratio,
            "passes_panel_coverage": bool(city_payload) and city_panel_ok_ratio >= min_panel_ok_ratio,
            "passes_trade_count": bool(city_payload) and city_num_trades >= min_city_policy_trades,
            "passes_non_negative_pnl": bool(city_payload) and city_pnl >= 0.0,
        }
        city_gate_details[city]["passes"] = bool(
            city_gate_details[city]["passes_panel_coverage"]
            and city_gate_details[city]["passes_trade_count"]
            and city_gate_details[city]["passes_non_negative_pnl"]
        )
    return city_gate_details


def _build_reduced_core_candidate(
    cities: dict[str, dict[str, object]],
    *,
    city_gate_details: dict[str, dict[str, object]],
    min_policy_trades: float,
    min_priced_decision_rows: float,
    min_panel_ok_ratio: float,
    min_city_policy_trades: float,
    min_coverage_eligible_cities: int,
    required_cities: tuple[str, ...],
) -> dict[str, object]:
    eligible_cities = [
        city for city in required_cities if bool(dict(city_gate_details.get(city, {})).get("passes_panel_coverage", False))
    ]
    eligible_payloads = {city: dict(cities.get(city, {})) for city in eligible_cities if city in cities}
    eligible_rows = list(eligible_payloads.values())
    aggregate_real_history_metrics = aggregate_backtest_metrics(
        [dict(row.get("real_history_metrics", {})) for row in eligible_rows]
    )
    aggregate_quote_proxy_metrics = aggregate_backtest_metrics(
        [dict(row.get("quote_proxy_metrics", {})) for row in eligible_rows]
    )
    aggregate_policy_real_history_metrics = aggregate_policy_metrics(
        [dict(row.get("policy_real_history_metrics", {})) for row in eligible_rows]
    )
    aggregate_policy_quote_proxy_metrics = aggregate_policy_metrics(
        [dict(row.get("policy_quote_proxy_metrics", {})) for row in eligible_rows]
    )
    aggregate_panel = aggregate_panel_coverage([dict(row.get("panel_summary", {})) for row in eligible_rows])
    eligible_city_gate_details = _build_city_gate_details(
        eligible_payloads,
        required_cities=tuple(eligible_cities),
        min_city_policy_trades=min_city_policy_trades,
        min_panel_ok_ratio=min_panel_ok_ratio,
    )
    classification = classify_profitability(
        aggregate_real_history_metrics=aggregate_real_history_metrics,
        aggregate_policy_real_history_metrics=aggregate_policy_real_history_metrics,
        aggregate_policy_quote_proxy_metrics=aggregate_policy_quote_proxy_metrics,
        aggregate_panel_coverage=aggregate_panel,
        city_gate_details=eligible_city_gate_details,
        min_policy_trades=min_policy_trades,
        min_priced_decision_rows=min_priced_decision_rows,
        min_panel_ok_ratio=min_panel_ok_ratio,
    )
    if len(eligible_cities) < min_coverage_eligible_cities:
        classification = {
            **classification,
            "decision": "INCONCLUSIVE",
            "decision_reason": "too_few_coverage_eligible_cities",
        }
    coverage_excluded_cities = {}
    for city in required_cities:
        if city in eligible_cities:
            continue
        details = dict(city_gate_details.get(city, {}))
        coverage_excluded_cities[city] = {
            "reason": "missing_city_payload" if not bool(details.get("available", False)) else "panel_coverage_below_threshold",
            "panel_ok_ratio": _as_float(details.get("panel_ok_ratio")),
            "min_panel_ok_ratio": float(min_panel_ok_ratio),
        }
    return {
        "publish_eligible": False,
        "min_coverage_eligible_cities": int(min_coverage_eligible_cities),
        "coverage_eligible_cities": eligible_cities,
        "coverage_excluded_cities": coverage_excluded_cities,
        "aggregate_real_history_metrics": aggregate_real_history_metrics,
        "aggregate_quote_proxy_metrics": aggregate_quote_proxy_metrics,
        "aggregate_policy_real_history_metrics": aggregate_policy_real_history_metrics,
        "aggregate_policy_quote_proxy_metrics": aggregate_policy_quote_proxy_metrics,
        "aggregate_panel_coverage": aggregate_panel,
        "city_gate_details": eligible_city_gate_details,
        **classification,
    }


def classify_profitability(
    *,
    aggregate_real_history_metrics: dict[str, object],
    aggregate_policy_real_history_metrics: dict[str, object],
    aggregate_policy_quote_proxy_metrics: dict[str, object],
    aggregate_panel_coverage: dict[str, object],
    city_gate_details: dict[str, dict[str, object]] | None = None,
    min_policy_trades: float = DEFAULT_MIN_POLICY_TRADES,
    min_priced_decision_rows: float = DEFAULT_MIN_PRICED_DECISION_ROWS,
    min_panel_ok_ratio: float = DEFAULT_MIN_PANEL_OK_RATIO,
) -> dict[str, object]:
    """Classify profitability using policy-filtered PnL and coverage gates."""

    policy_real_trades = _as_float(aggregate_policy_real_history_metrics.get("num_trades"))
    priced_decision_rows = _as_float(aggregate_real_history_metrics.get("priced_decision_rows"))
    panel_ok_ratio = _as_float(aggregate_panel_coverage.get("ok_ratio"))
    sample_adequacy = {
        "min_policy_trades": float(min_policy_trades),
        "min_priced_decision_rows": float(min_priced_decision_rows),
        "min_panel_ok_ratio": float(min_panel_ok_ratio),
        "policy_real_history_num_trades": policy_real_trades,
        "aggregate_real_history_priced_decision_rows": priced_decision_rows,
        "aggregate_panel_ok_ratio": panel_ok_ratio,
        "passes": bool(
            policy_real_trades >= min_policy_trades
            and priced_decision_rows >= min_priced_decision_rows
            and panel_ok_ratio >= min_panel_ok_ratio
        ),
    }
    if not sample_adequacy["passes"]:
        return {
            "decision": "INCONCLUSIVE",
            "decision_reason": "insufficient_sample",
            "sample_adequacy": sample_adequacy,
        }

    city_gate_details = city_gate_details or {}
    if city_gate_details and not all(bool(details.get("passes_panel_coverage")) for details in city_gate_details.values()):
        return {
            "decision": "INCONCLUSIVE",
            "decision_reason": "city_panel_coverage_inadequate",
            "sample_adequacy": sample_adequacy,
        }
    if city_gate_details and not all(bool(details.get("passes_trade_count")) for details in city_gate_details.values()):
        return {
            "decision": "INCONCLUSIVE",
            "decision_reason": "city_quote_proxy_sample_inadequate",
            "sample_adequacy": sample_adequacy,
        }
    if city_gate_details and not all(bool(details.get("passes_non_negative_pnl")) for details in city_gate_details.values()):
        return {
            "decision": "NO_GO",
            "decision_reason": "city_quote_proxy_negative_pnl",
            "sample_adequacy": sample_adequacy,
        }

    real_pnl = _as_float(aggregate_policy_real_history_metrics.get("pnl"))
    proxy_pnl = _as_float(aggregate_policy_quote_proxy_metrics.get("pnl"))
    if real_pnl > 0 and proxy_pnl > 0:
        decision = "GO"
        decision_reason = "positive_policy_pnl_in_real_and_proxy_with_city_gates"
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
    min_panel_ok_ratio: float = DEFAULT_MIN_PANEL_OK_RATIO,
    min_city_policy_trades: float = DEFAULT_MIN_CITY_POLICY_TRADES,
    min_coverage_eligible_cities: int = DEFAULT_MIN_COVERAGE_ELIGIBLE_CITIES,
    required_cities: tuple[str, ...] = DEFAULT_REQUIRED_CITIES,
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
    city_gate_details = _build_city_gate_details(
        cities,
        required_cities=required_cities,
        min_city_policy_trades=min_city_policy_trades,
        min_panel_ok_ratio=min_panel_ok_ratio,
    )
    classification = classify_profitability(
        aggregate_real_history_metrics=aggregate_real_history_metrics,
        aggregate_policy_real_history_metrics=aggregate_policy_real_history_metrics,
        aggregate_policy_quote_proxy_metrics=aggregate_policy_quote_proxy_metrics,
        aggregate_panel_coverage=aggregate_panel,
        city_gate_details=city_gate_details,
        min_policy_trades=min_policy_trades,
        min_priced_decision_rows=min_priced_decision_rows,
        min_panel_ok_ratio=min_panel_ok_ratio,
    )
    reduced_core_candidate = _build_reduced_core_candidate(
        cities,
        city_gate_details=city_gate_details,
        min_policy_trades=min_policy_trades,
        min_priced_decision_rows=min_priced_decision_rows,
        min_panel_ok_ratio=min_panel_ok_ratio,
        min_city_policy_trades=min_city_policy_trades,
        min_coverage_eligible_cities=min_coverage_eligible_cities,
        required_cities=required_cities,
    )
    return {
        "aggregate_real_history_metrics": aggregate_real_history_metrics,
        "aggregate_quote_proxy_metrics": aggregate_quote_proxy_metrics,
        "aggregate_policy_real_history_metrics": aggregate_policy_real_history_metrics,
        "aggregate_policy_quote_proxy_metrics": aggregate_policy_quote_proxy_metrics,
        "aggregate_panel_coverage": aggregate_panel,
        "city_gate_details": city_gate_details,
        "reduced_core_candidate": reduced_core_candidate,
        **classification,
    }
