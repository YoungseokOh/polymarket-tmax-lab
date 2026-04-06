"""Profitability gate helpers for recent-core revenue workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_PILOT_CONSTRAINTS = {
    "bankroll": 500.0,
    "max_city_exposure": 100.0,
    "global_max_exposure": 200.0,
    "approval_mode": "manual",
    "recent_core_cities": ["Seoul", "NYC", "London"],
}


def classify_path_viability(summary: Mapping[str, Any] | None) -> dict[str, object]:
    """Classify one shadow-style path into GO / INCONCLUSIVE / NO_GO."""

    if summary is None:
        return {
            "decision": "INCONCLUSIVE",
            "decision_reason": "missing_summary",
        }

    cycles = int(summary.get("cycles", 0) or 0)
    markets_evaluated = int(summary.get("markets_evaluated", 0) or 0)
    raw_gap_positive_count = int(summary.get("raw_gap_positive_count", 0) or 0)
    after_cost_edge_positive_count = int(summary.get("after_cost_edge_positive_count", 0) or 0)

    if markets_evaluated <= 0:
        decision = "INCONCLUSIVE"
        decision_reason = "no_observations"
    elif after_cost_edge_positive_count >= 2:
        decision = "GO"
        decision_reason = "repeated_after_cost_positive_edges"
    elif after_cost_edge_positive_count == 1:
        decision = "INCONCLUSIVE"
        decision_reason = "single_after_cost_positive_edge"
    elif cycles <= 1:
        decision = "INCONCLUSIVE"
        decision_reason = "single_cycle_only"
    elif raw_gap_positive_count <= 0:
        decision = "NO_GO"
        decision_reason = "raw_gap_never_positive"
    else:
        decision = "NO_GO"
        decision_reason = "after_cost_edge_never_positive"

    return {
        "decision": decision,
        "decision_reason": decision_reason,
    }


def build_revenue_gate_report(
    *,
    benchmark_summary: Mapping[str, Any] | None,
    opportunity_summary: Mapping[str, Any] | None,
    open_phase_summary: Mapping[str, Any] | None,
    observation_summary: Mapping[str, Any] | None = None,
    trading_alias_name: str = "trading_champion",
    pilot_constraints: Mapping[str, Any] | None = None,
    market_scope: str = "recent_core",
) -> dict[str, object]:
    """Build the combined revenue gate report used before live pilot promotion."""

    benchmark_decision = "INCONCLUSIVE"
    benchmark_reason = "missing_benchmark_summary"
    aggregate_policy_real_history_metrics: Mapping[str, Any] = {}
    aggregate_policy_quote_proxy_metrics: Mapping[str, Any] = {}
    aggregate_panel_coverage: Mapping[str, Any] = {}
    if benchmark_summary is not None:
        if "decision" in benchmark_summary or "decision_reason" in benchmark_summary:
            benchmark_decision = str(benchmark_summary.get("decision", benchmark_decision))
            benchmark_reason = str(benchmark_summary.get("decision_reason", benchmark_reason))
        elif bool(benchmark_summary.get("trading_champion_published")):
            benchmark_decision = "GO"
            benchmark_reason = "published_trading_champion_alias"
        elif bool(benchmark_summary.get("champion_published")):
            benchmark_decision = "INCONCLUSIVE"
            benchmark_reason = "published_research_champion_only"
        aggregate_policy_real_history_metrics = dict(
            benchmark_summary.get("aggregate_policy_real_history_metrics", {})
        )
        aggregate_policy_quote_proxy_metrics = dict(
            benchmark_summary.get("aggregate_policy_quote_proxy_metrics", {})
        )
        aggregate_panel_coverage = dict(benchmark_summary.get("aggregate_panel_coverage", {}))

    opportunity_gate = classify_path_viability(opportunity_summary)
    open_phase_gate = classify_path_viability(open_phase_summary)
    observation_gate = classify_path_viability(observation_summary)
    observation_source_breakdown = (
        dict(observation_summary.get("by_source_family", {}))
        if observation_summary is not None and isinstance(observation_summary.get("by_source_family", {}), Mapping)
        else {}
    )

    if benchmark_decision == "NO_GO":
        decision = "NO_GO"
        decision_reason = "benchmark_no_go"
    elif benchmark_decision != "GO":
        decision = "INCONCLUSIVE"
        decision_reason = "benchmark_not_go"
    elif (
        opportunity_gate["decision"] == "GO"
        or open_phase_gate["decision"] == "GO"
        or observation_gate["decision"] == "GO"
    ):
        decision = "GO"
        decision_reason = "benchmark_go_with_live_path_confirmation"
    elif (
        opportunity_gate["decision"] == "NO_GO"
        and open_phase_gate["decision"] == "NO_GO"
        and observation_gate["decision"] == "NO_GO"
    ):
        decision = "NO_GO"
        decision_reason = "benchmark_go_but_live_paths_no_go"
    else:
        decision = "INCONCLUSIVE"
        decision_reason = "benchmark_go_but_live_paths_inconclusive"

    constraints = dict(DEFAULT_PILOT_CONSTRAINTS)
    if pilot_constraints is not None:
        constraints.update(dict(pilot_constraints))

    return {
        "market_scope": market_scope,
        "decision": decision,
        "decision_reason": decision_reason,
        "benchmark_gate": {
            "decision": benchmark_decision,
            "decision_reason": benchmark_reason,
            "aggregate_policy_real_history_metrics": dict(aggregate_policy_real_history_metrics),
            "aggregate_policy_quote_proxy_metrics": dict(aggregate_policy_quote_proxy_metrics),
            "aggregate_panel_coverage": dict(aggregate_panel_coverage),
        },
        "opportunity_shadow_gate": opportunity_gate,
        "open_phase_shadow_gate": open_phase_gate,
        "observation_shadow_gate": observation_gate,
        "observation_source_breakdown": observation_source_breakdown,
        "required_model_alias": trading_alias_name,
        "eligible_for_live_pilot": decision == "GO",
        "pilot_constraints": constraints,
    }
