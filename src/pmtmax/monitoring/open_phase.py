"""Open-phase market observation helpers."""

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
from pmtmax.logging_utils import get_logger
from pmtmax.storage.schemas import MarketSnapshot, OpenPhaseObservation
from pmtmax.utils import dump_json

LOGGER = get_logger(__name__)

_TIMESTAMP_KEYS = (
    "acceptingOrdersTimestamp",
    "createdAt",
    "created_at",
    "deployingTimestamp",
)


def _parse_market_timestamp(value: object) -> datetime | None:
    """Parse a Polymarket timestamp field when present."""

    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _collect_component_timestamps(snapshot: MarketSnapshot, key: str) -> list[datetime]:
    """Return all component-market timestamps for a given field."""

    market = snapshot.market
    values: list[datetime] = []
    for component in market.get("componentMarkets", []):
        if not isinstance(component, dict):
            continue
        parsed = _parse_market_timestamp(component.get(key))
        if parsed is not None:
            values.append(parsed)
    parsed_top_level = _parse_market_timestamp(market.get(key))
    if parsed_top_level is not None:
        values.append(parsed_top_level)
    return values


def extract_open_phase_metadata(snapshot: MarketSnapshot, *, observed_at: datetime) -> dict[str, object]:
    """Extract listing/open-phase timestamps and current market age."""

    created_candidates = _collect_component_timestamps(snapshot, "createdAt") + _collect_component_timestamps(
        snapshot, "created_at"
    )
    deploying_candidates = _collect_component_timestamps(snapshot, "deployingTimestamp")
    accepting_candidates = _collect_component_timestamps(snapshot, "acceptingOrdersTimestamp")

    created_at = min(created_candidates) if created_candidates else None
    deploying_at = min(deploying_candidates) if deploying_candidates else None
    accepting_orders_at = min(accepting_candidates) if accepting_candidates else None
    opened_at = accepting_orders_at or created_at or deploying_at
    age_hours = None
    if opened_at is not None:
        age_hours = (observed_at - opened_at).total_seconds() / 3600.0
    return {
        "market_created_at": created_at,
        "market_deploying_at": deploying_at,
        "market_accepting_orders_at": accepting_orders_at,
        "market_opened_at": opened_at,
        "open_phase_age_hours": age_hours,
    }


@dataclass(frozen=True)
class OpenPhaseCandidate:
    """One active market still inside the configured open window."""

    snapshot: MarketSnapshot
    market_created_at: datetime | None
    market_deploying_at: datetime | None
    market_accepting_orders_at: datetime | None
    market_opened_at: datetime | None
    open_phase_age_hours: float


def select_open_phase_candidates(
    snapshots: list[MarketSnapshot],
    *,
    observed_at: datetime,
    open_window_hours: float,
) -> list[OpenPhaseCandidate]:
    """Filter active snapshots down to markets still in the open phase."""

    selected: list[OpenPhaseCandidate] = []
    for snapshot in snapshots:
        spec = snapshot.spec
        if spec is None:
            continue
        metadata = extract_open_phase_metadata(snapshot, observed_at=observed_at)
        opened_at = metadata["market_opened_at"]
        age_hours = metadata["open_phase_age_hours"]
        if not isinstance(opened_at, datetime) or not isinstance(age_hours, float):
            continue
        if age_hours < 0 or age_hours > open_window_hours:
            continue
        selected.append(
            OpenPhaseCandidate(
                snapshot=snapshot,
                market_created_at=metadata["market_created_at"] if isinstance(metadata["market_created_at"], datetime) else None,
                market_deploying_at=(
                    metadata["market_deploying_at"] if isinstance(metadata["market_deploying_at"], datetime) else None
                ),
                market_accepting_orders_at=(
                    metadata["market_accepting_orders_at"]
                    if isinstance(metadata["market_accepting_orders_at"], datetime)
                    else None
                ),
                market_opened_at=opened_at,
                open_phase_age_hours=age_hours,
            )
        )
    selected.sort(key=lambda candidate: candidate.open_phase_age_hours)
    return selected


def summarize_open_phase_history(history_path: Path) -> dict[str, Any]:
    """Summarize append-only open-phase observations."""

    rows: list[OpenPhaseObservation] = []
    if history_path.exists():
        with history_path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(OpenPhaseObservation.model_validate_json(line))

    def _summary(items: list[OpenPhaseObservation]) -> dict[str, Any]:
        reason_counts = Counter(item.reason for item in items)
        raw_values = [item.raw_gap for item in items if item.raw_gap is not None]
        edge_values = [item.after_cost_edge for item in items if item.after_cost_edge is not None]
        spread_values = [item.spread for item in items if item.spread is not None]
        age_values = [item.open_phase_age_hours for item in items if item.open_phase_age_hours is not None]
        return {
            "markets_evaluated": len(items),
            "tradable_count": reason_counts.get("tradable", 0),
            "reason_counts": dict(reason_counts),
            "raw_gap_positive_count": sum(1 for value in raw_values if value > 0),
            "after_cost_edge_positive_count": sum(1 for value in edge_values if value > 0),
            "best_raw_gap": max(raw_values) if raw_values else None,
            "best_after_cost_edge": max(edge_values) if edge_values else None,
            "median_spread": median(spread_values) if spread_values else None,
            "median_open_phase_age_hours": median(age_values) if age_values else None,
            "min_open_phase_age_hours": min(age_values) if age_values else None,
            "max_open_phase_age_hours": max(age_values) if age_values else None,
        }

    by_city: dict[str, list[OpenPhaseObservation]] = {}
    for row in rows:
        by_city.setdefault(row.city, []).append(row)

    return {
        "generated_at": datetime.now(tz=UTC),
        "cycles": len({row.observed_at.isoformat() for row in rows}),
        **_summary(rows),
        "by_city": {city: _summary(city_rows) for city, city_rows in sorted(by_city.items())},
    }


@dataclass
class OpenPhaseShadowRunner:
    """Persist periodic open-phase observations without trading."""

    config: RepoConfig
    interval_seconds: int = 60
    max_cycles: int = 0
    state_path: Path = Path("artifacts/open_phase_shadow_state.json")
    latest_output_path: Path = Path("artifacts/open_phase_shadow_latest.json")
    history_output_path: Path = Path("artifacts/open_phase_shadow.jsonl")
    summary_output_path: Path = Path("artifacts/open_phase_shadow_summary.json")

    snapshot_fetcher: Callable[[], list[Any]] | None = None
    evaluator: Callable[[list[Any], datetime], list[OpenPhaseObservation]] | None = None

    _running: bool = field(default=True, init=False, repr=False)
    _cycle: int = field(default=0, init=False, repr=False)

    def run_once(self) -> dict[str, Any]:
        """Execute one open-phase observation cycle."""

        observed_at = datetime.now(tz=UTC)
        snapshots = self.snapshot_fetcher() if self.snapshot_fetcher is not None else []
        observations = self.evaluator(snapshots, observed_at) if self.evaluator is not None else []
        observations = sorted(
            observations,
            key=lambda row: (row.reason != "tradable", -(row.after_cost_edge if row.after_cost_edge is not None else -999.0)),
        )

        self.latest_output_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(self.latest_output_path, [observation.model_dump(mode="json") for observation in observations])

        self.history_output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_output_path.open("a") as handle:
            for observation in observations:
                handle.write(observation.model_dump_json() + "\n")

        summary = summarize_open_phase_history(self.history_output_path)
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
        """Run the observer loop until stopped."""

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
                LOGGER.info("Open-phase shadow cycle complete: %s", summary)

                if 0 < self.max_cycles <= self._cycle:
                    LOGGER.info("Reached max_cycles=%d — exiting", self.max_cycles)
                    break

                if self._running:
                    time.sleep(self.interval_seconds)
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def _save_state(self, observed_at: datetime, observations: list[OpenPhaseObservation]) -> None:
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
