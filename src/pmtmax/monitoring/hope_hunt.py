"""Open-phase hope-hunt monitoring helpers."""

from __future__ import annotations

import json
import os
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
from pmtmax.monitoring.open_phase import open_phase_age_bucket
from pmtmax.storage.schemas import HopeHuntObservation
from pmtmax.utils import dump_json

LOGGER = get_logger(__name__)


def _signal_path(filename: str) -> Path:
    artifacts_root = Path(os.environ.get("PMTMAX_ARTIFACTS_DIR", "artifacts"))
    return artifacts_root / "signals" / "v2" / filename


def summarize_hope_hunt_history(history_path: Path) -> dict[str, Any]:
    """Summarize append-only hope-hunt observations."""

    rows: list[HopeHuntObservation] = []
    if history_path.exists():
        with history_path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(HopeHuntObservation.model_validate_json(line))

    def _summary(items: list[HopeHuntObservation]) -> dict[str, Any]:
        reason_counts = Counter(item.reason for item in items)
        priority_bucket_counts = Counter(item.priority_bucket for item in items if item.priority_bucket)
        raw_values = [item.raw_gap for item in items if item.raw_gap is not None]
        edge_values = [item.after_cost_edge for item in items if item.after_cost_edge is not None]
        spread_values = [item.spread for item in items if item.spread is not None]
        age_values = [item.open_phase_age_hours for item in items if item.open_phase_age_hours is not None]
        volume_values = [item.market_volume for item in items if item.market_volume is not None]
        candidate_count = sum(1 for item in items if item.candidate_alert)
        mature_market_count = sum(
            1
            for item in items
            if item.priority_bucket in {"mature_core", "mature_expansion", "unknown_open_time"}
        )
        summary = {
            "cycles": len({item.observed_at.isoformat() for item in items}),
            "markets_evaluated": len(items),
            "candidate_count": candidate_count,
            "fresh_listing_count": priority_bucket_counts.get("fresh_listing", 0)
            + priority_bucket_counts.get("fresh_listing_spread_blocked", 0),
            "mature_market_count": mature_market_count,
            "tradable_count": reason_counts.get("tradable", 0),
            "reason_counts": dict(reason_counts),
            "priority_bucket_counts": dict(priority_bucket_counts),
            "raw_gap_positive_count": sum(1 for value in raw_values if value > 0),
            "after_cost_edge_positive_count": sum(1 for value in edge_values if value > 0),
            "best_raw_gap": max(raw_values) if raw_values else None,
            "best_after_cost_edge": max(edge_values) if edge_values else None,
            "median_spread": median(spread_values) if spread_values else None,
            "median_open_phase_age_hours": median(age_values) if age_values else None,
            "median_market_volume": median(volume_values) if volume_values else None,
        }
        gate = classify_path_viability(summary)
        summary["gate_decision"] = gate["decision"]
        summary["gate_reason"] = gate["decision_reason"]
        return summary

    by_city: dict[str, list[HopeHuntObservation]] = {}
    by_open_phase_age_bucket: dict[str, list[HopeHuntObservation]] = {}
    by_priority_bucket: dict[str, list[HopeHuntObservation]] = {}
    for row in rows:
        by_city.setdefault(row.city, []).append(row)
        by_open_phase_age_bucket.setdefault(row.open_phase_age_bucket or open_phase_age_bucket(row.open_phase_age_hours), []).append(row)
        if row.priority_bucket:
            by_priority_bucket.setdefault(row.priority_bucket, []).append(row)

    top_candidates = sorted(
        rows,
        key=lambda row: (
            not row.candidate_alert,
            -(row.priority_score if row.priority_score is not None else -999.0),
            -(row.after_cost_edge if row.after_cost_edge is not None else -999.0),
        ),
    )[:10]
    summary = {
        "generated_at": datetime.now(tz=UTC),
        "cycles": len({row.observed_at.isoformat() for row in rows}),
        **_summary(rows),
        "top_candidates": [
            {
                "city": row.city,
                "question": row.question,
                "target_local_date": row.target_local_date.isoformat(),
                "open_phase_age_hours": row.open_phase_age_hours,
                "open_phase_age_bucket": row.open_phase_age_bucket,
                "target_day_distance": row.target_day_distance,
                "market_volume": row.market_volume,
                "priority_bucket": row.priority_bucket,
                "priority_score": row.priority_score,
                "reason": row.reason,
                "after_cost_edge": row.after_cost_edge,
            }
            for row in top_candidates
        ],
        "by_city": {city: _summary(city_rows) for city, city_rows in sorted(by_city.items())},
        "by_open_phase_age_bucket": {
            bucket: _summary(bucket_rows)
            for bucket, bucket_rows in sorted(by_open_phase_age_bucket.items())
        },
        "by_priority_bucket": {
            bucket: _summary(bucket_rows)
            for bucket, bucket_rows in sorted(by_priority_bucket.items())
        },
    }
    gate = classify_path_viability(summary)
    summary["gate_decision"] = gate["decision"]
    summary["gate_reason"] = gate["decision_reason"]
    return summary


@dataclass
class HopeHuntRunner:
    """Persist periodic hope-hunt observations without trading."""

    config: RepoConfig
    interval_seconds: int = 300
    max_cycles: int = 0
    state_path: Path = field(default_factory=lambda: _signal_path("hope_hunt_state.json"))
    latest_output_path: Path = field(default_factory=lambda: _signal_path("hope_hunt_latest.json"))
    history_output_path: Path = field(default_factory=lambda: _signal_path("hope_hunt_history.jsonl"))
    summary_output_path: Path = field(default_factory=lambda: _signal_path("hope_hunt_summary.json"))

    snapshot_fetcher: Callable[[], list[Any]] | None = None
    evaluator: Callable[[list[Any], datetime], list[HopeHuntObservation]] | None = None

    _running: bool = field(default=True, init=False, repr=False)
    _cycle: int = field(default=0, init=False, repr=False)

    def run_once(self) -> dict[str, Any]:
        """Execute one hope-hunt cycle."""

        observed_at = datetime.now(tz=UTC)
        snapshots = self.snapshot_fetcher() if self.snapshot_fetcher is not None else []
        observations = self.evaluator(snapshots, observed_at) if self.evaluator is not None else []
        observations = sorted(
            observations,
            key=lambda row: (
                not row.candidate_alert,
                -(row.priority_score if row.priority_score is not None else -999.0),
                -(row.after_cost_edge if row.after_cost_edge is not None else -999.0),
            ),
        )

        self.latest_output_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(self.latest_output_path, [observation.model_dump(mode="json") for observation in observations])

        self.history_output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_output_path.open("a") as handle:
            for observation in observations:
                handle.write(observation.model_dump_json() + "\n")

        summary = summarize_hope_hunt_history(self.history_output_path)
        self.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(self.summary_output_path, summary)

        self._cycle += 1
        self._save_state(observed_at, observations)
        return {
            "cycle": self._cycle,
            "markets_total": len(snapshots),
            "markets_evaluated": len(observations),
            "candidate_count": sum(1 for observation in observations if observation.candidate_alert),
            "reason_counts": dict(Counter(observation.reason for observation in observations)),
        }

    def run_loop(self) -> None:
        """Run the hope-hunt loop until stopped."""

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
                LOGGER.info("Hope-hunt cycle complete: %s", summary)

                if 0 < self.max_cycles <= self._cycle:
                    LOGGER.info("Reached max_cycles=%d — exiting", self.max_cycles)
                    break

                if self._running:
                    time.sleep(self.interval_seconds)
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def _save_state(self, observed_at: datetime, observations: list[HopeHuntObservation]) -> None:
        state = {
            "cycle": self._cycle,
            "last_completed_at": observed_at,
            "latest_output_path": str(self.latest_output_path),
            "history_output_path": str(self.history_output_path),
            "summary_output_path": str(self.summary_output_path),
            "markets_evaluated": len(observations),
            "candidate_count": sum(1 for observation in observations if observation.candidate_alert),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, default=str))
