"""Continuous opportunity shadow validation."""

from __future__ import annotations

import json
import signal
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Callable
from zoneinfo import ZoneInfo

from pmtmax.config.settings import RepoConfig
from pmtmax.logging_utils import get_logger
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.storage.schemas import OpportunityObservation
from pmtmax.utils import dump_json

LOGGER = get_logger(__name__)


def select_shadow_horizon(
    spec: MarketSpec,
    *,
    now_utc: datetime,
    near_term_days: int = 1,
) -> str | None:
    """Return the dynamic horizon for the near-term shadow universe."""

    local_today = now_utc.astimezone(ZoneInfo(spec.timezone)).date()
    delta_days = (spec.target_local_date - local_today).days
    if delta_days == 0:
        return "morning_of"
    if 0 < delta_days <= near_term_days:
        return "previous_evening"
    return None


def summarize_opportunity_history(history_path: Path) -> dict[str, Any]:
    """Summarize append-only opportunity observations."""

    rows: list[OpportunityObservation] = []
    if history_path.exists():
        with history_path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(OpportunityObservation.model_validate_json(line))

    def _summary(items: list[OpportunityObservation]) -> dict[str, Any]:
        reason_counts = Counter(item.reason for item in items)
        raw_values = [item.raw_gap for item in items if item.raw_gap is not None]
        edge_values = [item.after_cost_edge for item in items if item.after_cost_edge is not None]
        spread_values = [item.spread for item in items if item.spread is not None]
        liquidity_values = [item.visible_liquidity for item in items if item.visible_liquidity is not None]
        return {
            "markets_evaluated": len(items),
            "tradable_count": reason_counts.get("tradable", 0),
            "reason_counts": dict(reason_counts),
            "raw_gap_positive_count": sum(1 for value in raw_values if value > 0),
            "after_cost_edge_positive_count": sum(1 for value in edge_values if value > 0),
            "best_raw_gap": max(raw_values) if raw_values else None,
            "best_after_cost_edge": max(edge_values) if edge_values else None,
            "median_spread": median(spread_values) if spread_values else None,
            "median_visible_liquidity": median(liquidity_values) if liquidity_values else None,
        }

    by_city: dict[str, list[OpportunityObservation]] = {}
    for row in rows:
        by_city.setdefault(row.city, []).append(row)

    return {
        "generated_at": datetime.now(tz=UTC),
        "cycles": len({row.observed_at.isoformat() for row in rows}),
        **_summary(rows),
        "by_city": {
            city: _summary(city_rows)
            for city, city_rows in sorted(by_city.items())
        },
    }


@dataclass
class OpportunityShadowRunner:
    """Persist periodic opportunity-validation snapshots without trading."""

    config: RepoConfig
    interval_seconds: int = 60
    max_cycles: int = 0
    state_path: Path = Path("artifacts/opportunity_shadow_state.json")
    latest_output_path: Path = Path("artifacts/opportunity_shadow_latest.json")
    history_output_path: Path = Path("artifacts/opportunity_shadow.jsonl")
    summary_output_path: Path = Path("artifacts/opportunity_shadow_summary.json")

    snapshot_fetcher: Callable[[], list[Any]] | None = None
    evaluator: Callable[[list[Any], datetime], list[OpportunityObservation]] | None = None

    _running: bool = field(default=True, init=False, repr=False)
    _cycle: int = field(default=0, init=False, repr=False)

    def run_once(self) -> dict[str, Any]:
        """Execute one shadow-validation cycle."""

        observed_at = datetime.now(tz=UTC)
        snapshots = self.snapshot_fetcher() if self.snapshot_fetcher is not None else []
        observations = self.evaluator(snapshots, observed_at) if self.evaluator is not None else []
        observations = sorted(
            observations,
            key=lambda row: (row.reason != "tradable", -(row.after_cost_edge if row.after_cost_edge is not None else -999.0)),
        )

        self.latest_output_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(
            self.latest_output_path,
            [observation.model_dump(mode="json") for observation in observations],
        )

        self.history_output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_output_path.open("a") as handle:
            for observation in observations:
                handle.write(observation.model_dump_json() + "\n")

        summary = summarize_opportunity_history(self.history_output_path)
        self.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(self.summary_output_path, summary)

        self._cycle += 1
        self._save_state(observed_at, observations)
        return {
            "cycle": self._cycle,
            "markets_total": len(snapshots),
            "markets_evaluated": len(observations),
            "tradable_count": sum(1 for observation in observations if observation.reason == "tradable"),
            "reason_counts": dict(Counter(observation.reason for observation in observations)),
        }

    def run_loop(self) -> None:
        """Run the shadow loop until stopped."""

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
                LOGGER.info("Opportunity shadow cycle complete: %s", summary)

                if 0 < self.max_cycles <= self._cycle:
                    LOGGER.info("Reached max_cycles=%d — exiting", self.max_cycles)
                    break

                if self._running:
                    time.sleep(self.interval_seconds)
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def _save_state(self, observed_at: datetime, observations: list[OpportunityObservation]) -> None:
        state = {
            "cycle": self._cycle,
            "last_completed_at": observed_at,
            "latest_output_path": str(self.latest_output_path),
            "history_output_path": str(self.history_output_path),
            "summary_output_path": str(self.summary_output_path),
            "markets_evaluated": len(observations),
            "tradable_count": sum(1 for observation in observations if observation.reason == "tradable"),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, default=str))
