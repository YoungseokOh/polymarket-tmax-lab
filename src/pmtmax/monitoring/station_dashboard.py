"""Static station dashboard renderer for discovery, observation, and execution."""

from __future__ import annotations

import html
import json
import signal
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pmtmax.config.settings import RepoConfig
from pmtmax.logging_utils import get_logger
from pmtmax.utils import dump_json

LOGGER = get_logger(__name__)


def build_station_dashboard(
    *,
    opportunity_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
    queue_rows: list[dict[str, Any]],
    open_phase_rows: list[dict[str, Any]],
    revenue_gate_summary: Mapping[str, Any] | None,
    observation_summary: Mapping[str, Any] | None,
    open_phase_summary: Mapping[str, Any] | None,
    watchlist_playbook: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one combined station dashboard snapshot."""

    opportunity_rows = _sorted_rows(opportunity_rows, reason_key="reason")
    observation_rows = _sorted_rows(observation_rows, reason_key="queue_state")
    queue_rows = _sorted_rows(queue_rows, reason_key="queue_state")
    open_phase_rows = sorted(
        open_phase_rows,
        key=lambda row: (
            row.get("reason") != "tradable",
            -_float_value(row.get("edge")),
            _float_value(row.get("open_phase_age_hours"), default=10_000.0),
        ),
    )

    revenue_decision = str((revenue_gate_summary or {}).get("decision", "INCONCLUSIVE"))
    revenue_reason = str((revenue_gate_summary or {}).get("decision_reason", "missing_summary"))
    observation_source_breakdown = dict((observation_summary or {}).get("by_source_family", {}))
    observation_adapter_breakdown = dict((observation_summary or {}).get("by_observation_source", {}))
    watchlist_panel = _build_watchlist_panel(
        watchlist_playbook,
        opportunity_rows=opportunity_rows,
        observation_rows=observation_rows,
        queue_rows=queue_rows,
    )

    dashboard = {
        "generated_at": datetime.now(tz=UTC),
        "overview": {
            "revenue_gate_decision": revenue_decision,
            "revenue_gate_reason": revenue_reason,
            "opportunity_tradable_count": sum(1 for row in opportunity_rows if row.get("reason") == "tradable"),
            "observation_tradable_count": sum(1 for row in observation_rows if row.get("queue_state") == "tradable"),
            "queue_size": len(queue_rows),
            "open_phase_count": len(open_phase_rows),
            "manual_review_count": sum(1 for row in queue_rows if row.get("queue_state") == "manual_review"),
            "watchlist_alert_count": int(watchlist_panel.get("triggered_alert_count", 0) or 0),
        },
        "discovery_panel": {
            "gate_decision": str((open_phase_summary or {}).get("gate_decision", "INCONCLUSIVE")),
            "gate_reason": str((open_phase_summary or {}).get("gate_reason", "missing_summary")),
            "open_phase_count": len(open_phase_rows),
            "top_markets": [_compact_market_row(row, include_age=True) for row in open_phase_rows[:8]],
        },
        "observation_panel": {
            "gate_decision": str((observation_summary or {}).get("gate_decision", "INCONCLUSIVE")),
            "gate_reason": str((observation_summary or {}).get("gate_reason", "missing_summary")),
            "queue_counts": {
                "tradable": sum(1 for row in observation_rows if row.get("queue_state") == "tradable"),
                "manual_review": sum(1 for row in observation_rows if row.get("queue_state") == "manual_review"),
                "blocked": sum(1 for row in observation_rows if row.get("queue_state") == "blocked"),
            },
            "source_family_breakdown": _compact_breakdown(observation_source_breakdown),
            "observation_source_breakdown": _compact_breakdown(observation_adapter_breakdown),
            "top_candidates": [_compact_observation_row(row) for row in queue_rows[:8]],
            "top_after_cost_edges": list((observation_summary or {}).get("top_after_cost_edges", []))[:5],
            "top_price_vs_observation_gaps": list((observation_summary or {}).get("top_price_vs_observation_gaps", []))[
                :5
            ],
        },
        "execution_panel": {
            "required_model_alias": str((revenue_gate_summary or {}).get("required_model_alias", "trading_champion")),
            "eligible_for_live_pilot": bool((revenue_gate_summary or {}).get("eligible_for_live_pilot", False)),
            "queue_preview": [_compact_observation_row(row) for row in queue_rows[:5]],
            "top_opportunities": [_compact_market_row(row) for row in opportunity_rows[:8]],
            "observation_source_breakdown": _compact_breakdown(
                dict((revenue_gate_summary or {}).get("observation_source_breakdown", {}))
            ),
        },
        "watchlist_panel": watchlist_panel,
    }
    return dashboard


def render_station_dashboard_html(dashboard: Mapping[str, Any]) -> str:
    """Render a static HTML dashboard from one dashboard payload."""

    overview = dict(dashboard.get("overview", {}))
    discovery = dict(dashboard.get("discovery_panel", {}))
    observation = dict(dashboard.get("observation_panel", {}))
    execution = dict(dashboard.get("execution_panel", {}))
    watchlist = dict(dashboard.get("watchlist_panel", {}))

    def _kv_rows(payload: Mapping[str, Any]) -> str:
        return "".join(
            f"<tr><th>{html.escape(str(key).replace('_', ' ').title())}</th><td>{html.escape(str(value))}</td></tr>"
            for key, value in payload.items()
        )

    def _render_table(title: str, rows: list[Mapping[str, Any]], columns: list[str]) -> str:
        head = "".join(f"<th>{html.escape(column.replace('_', ' ').title())}</th>" for column in columns)
        body_rows = []
        for row in rows:
            body_cells = "".join(
                f"<td>{html.escape(_stringify_value(row.get(column)))}</td>"
                for column in columns
            )
            body_rows.append(f"<tr>{body_cells}</tr>")
        body = "".join(body_rows) or f"<tr><td colspan=\"{len(columns)}\">No rows</td></tr>"
        return (
            f"<section class=\"panel table-panel\"><h3>{html.escape(title)}</h3>"
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></section>"
        )

    def _render_breakdown(title: str, rows: list[Mapping[str, Any]]) -> str:
        return _render_table(
            title,
            rows,
            ["name", "markets_evaluated", "tradable_count", "manual_review_count", "after_cost_edge_positive_count", "gate_decision"],
        )

    generated_at = html.escape(str(dashboard.get("generated_at", "")))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PMTMAX Station Dashboard</title>
  <style>
    :root {{
      --bg: #07131a;
      --panel: #10222c;
      --panel-2: #16313d;
      --line: #2e5868;
      --ink: #eef7fb;
      --muted: #8eb6c6;
      --accent: #7cf0c8;
      --warn: #ffd166;
      --alert: #ff7b72;
      --shadow: 0 18px 44px rgba(0,0,0,.28);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(124,240,200,.08), transparent 28rem),
        linear-gradient(160deg, #07131a, #0d1820 45%, #07131a);
      color: var(--ink);
    }}
    header {{
      padding: 2rem 2rem 1rem 2rem;
      border-bottom: 1px solid rgba(142,182,198,.16);
    }}
    header h1 {{
      margin: 0 0 .35rem 0;
      font-size: 2rem;
      letter-spacing: .04em;
      text-transform: uppercase;
    }}
    header p {{
      margin: 0;
      color: var(--muted);
    }}
    .overview {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
      gap: 1rem;
      padding: 1.25rem 2rem 0 2rem;
    }}
    .stat {{
      background: linear-gradient(180deg, rgba(22,49,61,.92), rgba(16,34,44,.96));
      border: 1px solid rgba(142,182,198,.18);
      border-radius: 1.1rem;
      padding: 1rem 1.1rem;
      box-shadow: var(--shadow);
    }}
    .stat .label {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .08em;
      font-size: .72rem;
    }}
    .stat .value {{
      display: block;
      margin-top: .35rem;
      font-size: 1.55rem;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(22rem, 1fr));
      gap: 1.2rem;
      padding: 1.25rem 2rem 2rem 2rem;
    }}
    .panel {{
      background: linear-gradient(180deg, rgba(16,34,44,.96), rgba(9,24,31,.98));
      border: 1px solid rgba(142,182,198,.16);
      border-radius: 1.2rem;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .panel h2, .panel h3 {{
      margin: 0;
      padding: 1rem 1.1rem;
      border-bottom: 1px solid rgba(142,182,198,.12);
      letter-spacing: .06em;
      text-transform: uppercase;
      font-size: .92rem;
    }}
    .panel-body {{
      padding: 1rem 1.1rem 1.2rem 1.1rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: .88rem;
    }}
    th, td {{
      padding: .58rem .45rem;
      border-bottom: 1px solid rgba(142,182,198,.09);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .06em;
      font-size: .72rem;
    }}
    .tables {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 1rem;
      padding: 0 2rem 2rem 2rem;
    }}
    .table-panel {{
      background: linear-gradient(180deg, rgba(16,34,44,.96), rgba(9,24,31,.98));
      border: 1px solid rgba(142,182,198,.16);
      border-radius: 1.2rem;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .accent {{ color: var(--accent); }}
    .warn {{ color: var(--warn); }}
    .alert {{ color: var(--alert); }}
  </style>
</head>
<body>
  <header>
    <h1>PMTMAX Station Dashboard</h1>
    <p>Generated at {generated_at}</p>
  </header>
  <section class="overview">
    <article class="stat"><span class="label">Revenue Gate</span><span class="value">{html.escape(_stringify_value(overview.get("revenue_gate_decision", "INCONCLUSIVE")))}</span></article>
    <article class="stat"><span class="label">Opportunity Tradable</span><span class="value">{html.escape(_stringify_value(overview.get("opportunity_tradable_count", 0)))}</span></article>
    <article class="stat"><span class="label">Observation Tradable</span><span class="value">{html.escape(_stringify_value(overview.get("observation_tradable_count", 0)))}</span></article>
    <article class="stat"><span class="label">Manual Queue</span><span class="value">{html.escape(_stringify_value(overview.get("manual_review_count", 0)))}</span></article>
    <article class="stat"><span class="label">Queue Size</span><span class="value">{html.escape(_stringify_value(overview.get("queue_size", 0)))}</span></article>
    <article class="stat"><span class="label">Open-Phase Count</span><span class="value">{html.escape(_stringify_value(overview.get("open_phase_count", 0)))}</span></article>
    <article class="stat"><span class="label">Watchlist Alerts</span><span class="value">{html.escape(_stringify_value(overview.get("watchlist_alert_count", 0)))}</span></article>
  </section>
  <section class="grid">
    <section class="panel">
      <h2>Discovery</h2>
      <div class="panel-body">
        <table><tbody>{_kv_rows({"gate_decision": discovery.get("gate_decision"), "gate_reason": discovery.get("gate_reason"), "open_phase_count": discovery.get("open_phase_count")})}</tbody></table>
      </div>
    </section>
    <section class="panel">
      <h2>Observation</h2>
      <div class="panel-body">
        <table><tbody>{_kv_rows({"gate_decision": observation.get("gate_decision"), "gate_reason": observation.get("gate_reason"), "queue_counts": observation.get("queue_counts")})}</tbody></table>
      </div>
    </section>
    <section class="panel">
      <h2>Execution</h2>
      <div class="panel-body">
        <table><tbody>{_kv_rows({"required_model_alias": execution.get("required_model_alias"), "eligible_for_live_pilot": execution.get("eligible_for_live_pilot"), "revenue_gate_reason": overview.get("revenue_gate_reason")})}</tbody></table>
      </div>
    </section>
    <section class="panel">
      <h2>Watchlist</h2>
      <div class="panel-body">
        <table><tbody>{_kv_rows({"tier_a_cities": watchlist.get("tier_a_cities", []), "tier_b_cities": watchlist.get("tier_b_cities", []), "triggered_alert_count": watchlist.get("triggered_alert_count", 0)})}</tbody></table>
      </div>
    </section>
  </section>
  <section class="tables">
    {_render_table("Discovery Top Markets", list(discovery.get("top_markets", [])), ["city", "target_local_date", "decision_horizon", "outcome_label", "edge", "open_phase_age_hours", "reason"])}
    {_render_table("Observation Queue", list(observation.get("top_candidates", [])), ["city", "target_local_date", "decision_horizon", "queue_state", "outcome_label", "edge", "source_family", "observation_source"])}
    {_render_table("Execution Opportunities", list(execution.get("top_opportunities", [])), ["city", "target_local_date", "decision_horizon", "outcome_label", "edge", "reason"])}
    {_render_table("Watchlist Alerts", list(watchlist.get("triggered_alerts", [])), ["city", "target_local_date", "decision_horizon", "outcome_label", "current_best_ask", "threshold_ask", "gap_to_threshold"])}
    {_render_table("Watchlist Rules", list(watchlist.get("top_rules", [])), ["tier", "city", "target_local_date", "decision_horizon", "outcome_label", "current_best_ask", "threshold_ask"])}
    {_render_breakdown("Observation Source Families", list(observation.get("source_family_breakdown", [])))}
    {_render_breakdown("Observation Adapters", list(observation.get("observation_source_breakdown", [])))}
  </section>
</body>
</html>"""


@dataclass
class StationDashboardRunner:
    """Persist one combined dashboard from existing station artifacts."""

    config: RepoConfig
    interval_seconds: int = 60
    max_cycles: int = 0
    state_path: Path = Path("artifacts/signals/v2/station_dashboard_state.json")
    json_output_path: Path = Path("artifacts/signals/v2/station_dashboard.json")
    html_output_path: Path = Path("artifacts/signals/v2/station_dashboard.html")
    data_loader: Callable[[], dict[str, Any]] | None = None

    _running: bool = field(default=True, init=False, repr=False)
    _cycle: int = field(default=0, init=False, repr=False)

    def run_once(self) -> dict[str, Any]:
        """Render one dashboard snapshot."""

        raw_inputs = self.data_loader() if self.data_loader is not None else {}
        dashboard = build_station_dashboard(**raw_inputs)
        dump_json(self.json_output_path, dashboard)
        self.html_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.html_output_path.write_text(render_station_dashboard_html(dashboard))
        self._cycle += 1
        self._save_state(dashboard)
        overview = dict(dashboard.get("overview", {}))
        return {
            "cycle": self._cycle,
            "revenue_gate_decision": overview.get("revenue_gate_decision", "INCONCLUSIVE"),
            "queue_size": int(overview.get("queue_size", 0) or 0),
            "observation_tradable_count": int(overview.get("observation_tradable_count", 0) or 0),
            "opportunity_tradable_count": int(overview.get("opportunity_tradable_count", 0) or 0),
            "open_phase_count": int(overview.get("open_phase_count", 0) or 0),
            "watchlist_alert_count": int(overview.get("watchlist_alert_count", 0) or 0),
        }

    def run_loop(self) -> None:
        """Run the dashboard renderer until stopped."""

        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_shutdown(signum: int, frame: Any) -> None:
            LOGGER.info("Received signal %s — shutting down after current dashboard cycle", signum)
            self._running = False

        signal.signal(signal.SIGINT, _handle_shutdown)
        signal.signal(signal.SIGTERM, _handle_shutdown)

        try:
            while self._running:
                summary = self.run_once()
                LOGGER.info("Station dashboard cycle complete: %s", summary)
                if 0 < self.max_cycles <= self._cycle:
                    LOGGER.info("Reached max_cycles=%d — exiting dashboard renderer", self.max_cycles)
                    break
                if self._running:
                    time.sleep(self.interval_seconds)
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def _save_state(self, dashboard: Mapping[str, Any]) -> None:
        overview = dict(dashboard.get("overview", {}))
        state = {
            "cycle": self._cycle,
            "last_completed_at": datetime.now(tz=UTC),
            "json_output_path": str(self.json_output_path),
            "html_output_path": str(self.html_output_path),
            "revenue_gate_decision": overview.get("revenue_gate_decision", "INCONCLUSIVE"),
            "queue_size": int(overview.get("queue_size", 0) or 0),
            "observation_tradable_count": int(overview.get("observation_tradable_count", 0) or 0),
            "opportunity_tradable_count": int(overview.get("opportunity_tradable_count", 0) or 0),
            "open_phase_count": int(overview.get("open_phase_count", 0) or 0),
            "watchlist_alert_count": int(overview.get("watchlist_alert_count", 0) or 0),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, default=str))


def _sorted_rows(rows: list[dict[str, Any]], *, reason_key: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get(reason_key) not in {"tradable"},
            row.get(reason_key) not in {"manual_review"},
            -_float_value(row.get("edge")),
        ),
    )


def _compact_breakdown(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, value in sorted(payload.items()):
        if not isinstance(value, Mapping):
            continue
        rows.append(
            {
                "name": name,
                "markets_evaluated": int(value.get("markets_evaluated", 0) or 0),
                "tradable_count": int(value.get("tradable_count", 0) or 0),
                "manual_review_count": int(value.get("manual_review_count", 0) or 0),
                "after_cost_edge_positive_count": int(value.get("after_cost_edge_positive_count", 0) or 0),
                "gate_decision": str(value.get("gate_decision", "")),
            }
        )
    return rows


def _compact_market_row(row: Mapping[str, Any], *, include_age: bool = False) -> dict[str, Any]:
    payload = {
        "city": row.get("city", ""),
        "target_local_date": row.get("target_local_date", ""),
        "decision_horizon": row.get("decision_horizon", ""),
        "outcome_label": row.get("outcome_label", "—"),
        "edge": _rounded(row.get("edge")),
        "reason": row.get("reason", ""),
    }
    if include_age:
        payload["open_phase_age_hours"] = _rounded(row.get("open_phase_age_hours"))
    return payload


def _compact_observation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "market_id": row.get("market_id", ""),
        "city": row.get("city", ""),
        "target_local_date": row.get("target_local_date", ""),
        "decision_horizon": row.get("decision_horizon", ""),
        "queue_state": row.get("queue_state", ""),
        "outcome_label": row.get("outcome_label", "—"),
        "edge": _rounded(row.get("edge")),
        "source_family": row.get("source_family", ""),
        "observation_source": row.get("observation_source", ""),
    }


def _build_watchlist_panel(
    playbook: Mapping[str, Any] | None,
    *,
    opportunity_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
    queue_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tier_a = _playbook_entry(playbook, "fee_sensitive_watchlist")
    tier_b = _playbook_entry(playbook, "policy_blocked_watchlist")
    tier_c = _playbook_entry(playbook, "raw_edge_desert")
    tier_a_rules = _playbook_rules(tier_a)
    current_rows = _current_watchlist_rows(opportunity_rows, observation_rows, queue_rows)
    triggered_alerts = _detect_watchlist_alerts(current_rows, tier_a_rules)
    top_rules = [
        _compact_watchlist_rule(rule, current_rows=current_rows)
        for rule in tier_a_rules[:8]
    ]
    return {
        "tier_a_cities": list(tier_a.get("cities", [])) if isinstance(tier_a, Mapping) else [],
        "tier_b_cities": list(tier_b.get("cities", [])) if isinstance(tier_b, Mapping) else [],
        "tier_c_cities": list(tier_c.get("cities", [])) if isinstance(tier_c, Mapping) else [],
        "triggered_alert_count": len(triggered_alerts),
        "triggered_alerts": triggered_alerts[:8],
        "top_rules": top_rules,
        "next_actions": list(playbook.get("next_actions", [])) if isinstance(playbook, Mapping) else [],
    }


def _playbook_entry(playbook: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    entries = playbook.get("playbook", []) if isinstance(playbook, Mapping) else []
    if not isinstance(entries, list):
        return {}
    for entry in entries:
        if isinstance(entry, Mapping) and str(entry.get("name", "")) == name:
            return entry
    return {}


def _playbook_rules(entry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    evidence = entry.get("evidence", [])
    if not isinstance(evidence, list):
        return []
    return [item for item in evidence if isinstance(item, Mapping)]


def _current_watchlist_rows(*row_groups: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    current: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rows in row_groups:
        for row in rows:
            market_id = str(row.get("market_id") or "")
            outcome_label = str(row.get("outcome_label") or "")
            decision_horizon = str(row.get("decision_horizon") or "")
            if not market_id or not outcome_label or not decision_horizon:
                continue
            best_ask = _current_best_ask(row)
            if best_ask is None:
                continue
            key = (market_id, outcome_label, decision_horizon)
            existing = current.get(key)
            if existing is None or best_ask < _float_value(existing.get("current_best_ask"), default=999.0):
                current[key] = {
                    "market_id": market_id,
                    "city": row.get("city", ""),
                    "target_local_date": row.get("target_local_date", ""),
                    "decision_horizon": decision_horizon,
                    "outcome_label": outcome_label,
                    "current_best_ask": round(best_ask, 6),
                }
    return current


def _detect_watchlist_alerts(
    current_rows: Mapping[tuple[str, str, str], Mapping[str, Any]],
    rules: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for rule in rules:
        market_id = str(rule.get("market_id") or "")
        outcome_label = str(rule.get("outcome_label") or "")
        decision_horizon = str(rule.get("decision_horizon") or "")
        threshold = _float_value(rule.get("watch_rule_threshold_ask"), default=-1.0)
        if not market_id or not outcome_label or not decision_horizon or threshold < 0:
            continue
        matched = current_rows.get((market_id, outcome_label, decision_horizon))
        if matched is None:
            continue
        current_best_ask = _float_value(matched.get("current_best_ask"), default=999.0)
        if current_best_ask > threshold:
            continue
        alerts.append(
            {
                "city": matched.get("city", rule.get("city", "")),
                "target_local_date": matched.get("target_local_date", rule.get("target_local_date", "")),
                "decision_horizon": decision_horizon,
                "outcome_label": outcome_label,
                "current_best_ask": round(current_best_ask, 6),
                "threshold_ask": round(threshold, 6),
                "gap_to_threshold": round(threshold - current_best_ask, 6),
            }
        )
    alerts.sort(
        key=lambda row: (
            -_float_value(row.get("gap_to_threshold")),
            str(row.get("city", "")),
            str(row.get("target_local_date", "")),
        )
    )
    return alerts


def _compact_watchlist_rule(
    rule: Mapping[str, Any],
    *,
    current_rows: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    market_id = str(rule.get("market_id") or "")
    outcome_label = str(rule.get("outcome_label") or "")
    decision_horizon = str(rule.get("decision_horizon") or "")
    matched = current_rows.get((market_id, outcome_label, decision_horizon), {})
    return {
        "tier": "A",
        "city": rule.get("city", ""),
        "target_local_date": rule.get("target_local_date", ""),
        "decision_horizon": decision_horizon,
        "outcome_label": outcome_label,
        "current_best_ask": _rounded(matched.get("current_best_ask")),
        "threshold_ask": _rounded(rule.get("watch_rule_threshold_ask")),
    }


def _current_best_ask(row: Mapping[str, Any]) -> float | None:
    for key in ("best_ask", "executable_price"):
        value = row.get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _rounded(value: Any) -> float | str | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return str(value)


def _float_value(value: Any, *, default: float = -999.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _stringify_value(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key}={value[key]}" for key in sorted(value))
    if isinstance(value, list):
        return ", ".join(_stringify_value(item) for item in value)
    return str(value)
