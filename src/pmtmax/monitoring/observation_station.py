"""Observation-station monitoring and summary helpers."""

from __future__ import annotations

import json
import signal
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from pmtmax.config.settings import RepoConfig
from pmtmax.execution.revenue_gate import classify_path_viability
from pmtmax.logging_utils import get_logger
from pmtmax.storage.schemas import ObservationOpportunity
from pmtmax.utils import dump_json

LOGGER = get_logger(__name__)


def summarize_observation_history(history_path: Path) -> dict[str, Any]:
    """Summarize append-only observation-station observations."""

    rows: list[ObservationOpportunity] = []
    if history_path.exists():
        with history_path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(ObservationOpportunity.model_validate_json(line))

    def _nested_reason_counts(
        items: list[ObservationOpportunity],
        *,
        key_fn: Callable[[ObservationOpportunity], str],
    ) -> dict[str, dict[str, int]]:
        nested: dict[str, Counter[str]] = {}
        for item in items:
            key = key_fn(item)
            nested.setdefault(key, Counter())[item.reason] += 1
        return {
            group: dict(sorted(counter.items()))
            for group, counter in sorted(nested.items())
        }

    def _summary(items: list[ObservationOpportunity]) -> dict[str, Any]:
        reason_counts = Counter(item.reason for item in items)
        queue_counts = Counter(item.queue_state for item in items)
        risk_flag_counts = Counter(flag for item in items for flag in item.risk_flags)
        raw_values = [item.raw_gap for item in items if item.raw_gap is not None]
        edge_values = [item.after_cost_edge for item in items if item.after_cost_edge is not None]
        spread_values = [item.spread for item in items if item.spread is not None]
        liquidity_values = [item.visible_liquidity for item in items if item.visible_liquidity is not None]
        freshness_values = [
            item.observation_freshness_minutes
            for item in items
            if item.observation_freshness_minutes is not None
        ]
        impossible_price_values = [
            item.impossible_price_mass
            for item in items
            if item.impossible_price_mass is not None
        ]
        summary = {
            "cycles": len({item.observed_at.isoformat() for item in items}),
            "markets_evaluated": len(items),
            "tradable_count": queue_counts.get("tradable", 0),
            "manual_review_count": queue_counts.get("manual_review", 0),
            "reason_counts": dict(reason_counts),
            "queue_state_counts": dict(queue_counts),
            "risk_flag_counts": dict(risk_flag_counts),
            "raw_gap_positive_count": sum(1 for value in raw_values if value > 0),
            "after_cost_edge_positive_count": sum(1 for value in edge_values if value > 0),
            "best_raw_gap": max(raw_values) if raw_values else None,
            "best_after_cost_edge": max(edge_values) if edge_values else None,
            "median_spread": median(spread_values) if spread_values else None,
            "median_visible_liquidity": median(liquidity_values) if liquidity_values else None,
            "median_observation_freshness_minutes": median(freshness_values) if freshness_values else None,
            "best_impossible_price_mass": max(impossible_price_values) if impossible_price_values else None,
        }
        gate = classify_path_viability(summary)
        summary["gate_decision"] = gate["decision"]
        summary["gate_reason"] = gate["decision_reason"]
        return summary

    def _group_summary(items: list[ObservationOpportunity], key_fn: Callable[[ObservationOpportunity], str]) -> dict[str, Any]:
        groups: dict[str, list[ObservationOpportunity]] = {}
        for item in items:
            groups.setdefault(key_fn(item), []).append(item)
        return {
            group: _summary(group_rows)
            for group, group_rows in sorted(groups.items())
        }

    def _top_candidates(
        items: list[ObservationOpportunity],
        *,
        limit: int,
        metric: str,
        metric_key: Callable[[ObservationOpportunity], float | None],
        reasons: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        ranked = [
            item
            for item in items
            if metric_key(item) is not None and (reasons is None or item.reason in reasons)
        ]
        ranked.sort(
            key=lambda item: (
                -(metric_key(item) or 0.0),
                item.observed_at,
                item.market_id,
            )
        )
        rows: list[dict[str, Any]] = []
        for item in ranked[:limit]:
            rows.append(
                {
                    "market_id": item.market_id,
                    "city": item.city,
                    "target_local_date": item.target_local_date.isoformat(),
                    "decision_horizon": item.decision_horizon,
                    "queue_state": item.queue_state,
                    "truth_track": item.truth_track,
                    "candidate_tier": item.candidate_tier,
                    "source_family": item.source_family,
                    "observation_source": item.observation_source,
                    metric: metric_key(item),
                    "outcome_label": item.outcome_label,
                    "risk_flags": list(item.risk_flags),
                }
            )
        return rows

    summary = {
        "generated_at": datetime.now(tz=UTC),
        "cycles": len({row.observed_at.isoformat() for row in rows}),
        **_summary(rows),
        "by_city": _group_summary(rows, lambda item: item.city),
        "by_horizon": _group_summary(rows, lambda item: item.decision_horizon),
        "by_candidate_tier": _group_summary(rows, lambda item: item.candidate_tier or "unknown"),
        "by_truth_track": _group_summary(rows, lambda item: item.truth_track or "unknown"),
        "by_source_family": _group_summary(rows, lambda item: item.source_family or "unknown"),
        "by_observation_source": _group_summary(rows, lambda item: item.observation_source or "unknown"),
        "by_reason": _group_summary(rows, lambda item: item.reason),
        "by_city_reason": _nested_reason_counts(rows, key_fn=lambda item: item.city),
        "by_horizon_reason": _nested_reason_counts(rows, key_fn=lambda item: item.decision_horizon),
        "top_after_cost_edges": _top_candidates(
            rows,
            limit=5,
            metric="after_cost_edge",
            metric_key=lambda item: item.after_cost_edge,
        ),
        "top_price_vs_observation_gaps": _top_candidates(
            rows,
            limit=5,
            metric="price_vs_observation_gap",
            metric_key=lambda item: item.price_vs_observation_gap,
        ),
        "top_near_miss_markets": _top_candidates(
            rows,
            limit=10,
            metric="after_cost_edge",
            metric_key=lambda item: item.after_cost_edge,
            reasons={
                "fee_killed_edge",
                "slippage_killed_edge",
                "after_cost_positive_but_spread_too_wide",
                "after_cost_positive_but_liquidity_too_low",
                "after_cost_positive_but_below_threshold",
                "insufficient_depth",
            },
        ),
        "top_fee_killed_markets": _top_candidates(
            rows,
            limit=10,
            metric="price_vs_observation_gap",
            metric_key=lambda item: item.price_vs_observation_gap,
            reasons={"fee_killed_edge"},
        ),
        "top_spread_blocked_markets": _top_candidates(
            rows,
            limit=10,
            metric="spread",
            metric_key=lambda item: item.spread,
            reasons={"after_cost_positive_but_spread_too_wide", "slippage_killed_edge", "insufficient_depth"},
        ),
        "top_policy_filtered_markets": _top_candidates(
            rows,
            limit=10,
            metric="price_vs_observation_gap",
            metric_key=lambda item: item.price_vs_observation_gap if item.price_vs_observation_gap is not None else 0.0,
            reasons={"policy_filtered"},
        ),
    }
    gate = classify_path_viability(summary)
    summary["gate_decision"] = gate["decision"]
    summary["gate_reason"] = gate["decision_reason"]
    return summary


@dataclass
class ObservationShadowRunner:
    """Persist periodic observation-driven opportunities without trading."""

    config: RepoConfig
    interval_seconds: int = 300
    max_cycles: int = 0
    state_path: Path = Path("artifacts/signals/v2/observation_shadow_state.json")
    latest_output_path: Path = Path("artifacts/signals/v2/observation_shadow_latest.json")
    history_output_path: Path = Path("artifacts/signals/v2/observation_shadow.jsonl")
    summary_output_path: Path = Path("artifacts/signals/v2/observation_shadow_summary.json")
    alerts_output_path: Path = Path("artifacts/signals/v2/observation_alerts_latest.json")
    queue_output_path: Path = Path("artifacts/signals/v2/live_pilot_queue.json")

    snapshot_fetcher: Callable[[], list[Any]] | None = None
    evaluator: Callable[[list[Any], datetime], list[ObservationOpportunity]] | None = None

    _running: bool = field(default=True, init=False, repr=False)
    _cycle: int = field(default=0, init=False, repr=False)

    def run_once(self) -> dict[str, Any]:
        """Execute one observation-station cycle."""

        observed_at = datetime.now(tz=UTC)
        snapshots = self.snapshot_fetcher() if self.snapshot_fetcher is not None else []
        observations = self.evaluator(snapshots, observed_at) if self.evaluator is not None else []
        observations = sorted(
            observations,
            key=lambda row: (
                row.queue_state != "tradable",
                row.queue_state != "manual_review",
                -(row.after_cost_edge if row.after_cost_edge is not None else -999.0),
            ),
        )

        latest_rows = [observation.model_dump(mode="json") for observation in observations]
        queue_rows = [row for row in latest_rows if row.get("queue_state") in {"tradable", "manual_review"}]
        alert_rows = [row for row in queue_rows if row.get("price_vs_observation_gap") is not None]

        self.latest_output_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(self.latest_output_path, latest_rows)
        dump_json(self.queue_output_path, queue_rows)
        dump_json(self.alerts_output_path, alert_rows)

        self.history_output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_output_path.open("a") as handle:
            for observation in observations:
                handle.write(observation.model_dump_json() + "\n")

        summary = summarize_observation_history(self.history_output_path)
        self.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(self.summary_output_path, summary)

        self._cycle += 1
        self._save_state(observed_at, observations)
        return {
            "cycle": self._cycle,
            "markets_total": len(snapshots),
            "markets_evaluated": len(observations),
            "tradable_count": sum(1 for observation in observations if observation.queue_state == "tradable"),
            "manual_review_count": sum(1 for observation in observations if observation.queue_state == "manual_review"),
            "reason_counts": dict(Counter(observation.reason for observation in observations)),
        }

    def run_loop(self) -> None:
        """Run the observation loop until stopped."""

        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_shutdown(signum: int, frame: Any) -> None:
            LOGGER.info("Received signal %s — shutting down after current cycle", signum)
            self._running = False

        signal.signal(signal.SIGINT, _handle_shutdown)
        signal.signal(signal.SIGTERM, _handle_shutdown)

        try:
            while self._running:
                summary = self.run_once()
                LOGGER.info("Observation station cycle complete: %s", summary)

                if 0 < self.max_cycles <= self._cycle:
                    LOGGER.info("Reached max_cycles=%d — exiting", self.max_cycles)
                    break

                if self._running:
                    time.sleep(self.interval_seconds)
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def _save_state(self, observed_at: datetime, observations: list[ObservationOpportunity]) -> None:
        state = {
            "cycle": self._cycle,
            "last_completed_at": observed_at,
            "latest_output_path": str(self.latest_output_path),
            "history_output_path": str(self.history_output_path),
            "summary_output_path": str(self.summary_output_path),
            "alerts_output_path": str(self.alerts_output_path),
            "queue_output_path": str(self.queue_output_path),
            "markets_evaluated": len(observations),
            "tradable_count": sum(1 for observation in observations if observation.queue_state == "tradable"),
            "manual_review_count": sum(
                1 for observation in observations if observation.queue_state == "manual_review"
            ),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, default=str))
