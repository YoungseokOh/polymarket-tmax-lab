"""Continuous scanning daemon for market monitoring and automated trading."""

from __future__ import annotations

import json
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pmtmax.config.settings import RepoConfig
from pmtmax.execution.paper_broker import PaperBroker
from pmtmax.logging_utils import get_logger
from pmtmax.storage.schemas import ProbForecast

LOGGER = get_logger(__name__)


@dataclass
class ContinuousScanner:
    """Periodically scan markets, manage stops, and evaluate entry signals."""

    config: RepoConfig
    broker: PaperBroker
    interval_seconds: int = 60
    max_cycles: int = 0
    state_path: Path = Path("artifacts/scanner_state.json")

    _running: bool = field(default=True, init=False, repr=False)
    _cycle: int = field(default=0, init=False, repr=False)

    # Pluggable callbacks for fetching external data — set by the caller.
    price_fetcher: Any = None  # Callable[[], dict[str, float]]
    forecast_fetcher: Any = None  # Callable[[], dict[str, ProbForecast]]
    entry_evaluator: Any = None  # Callable[[PaperBroker], None]

    def run_once(self) -> dict[str, Any]:
        """Execute a single scan cycle.

        Returns a summary dict of actions taken during the cycle.
        """

        summary: dict[str, Any] = {"cycle": self._cycle, "stops": [], "forecast_exits": [], "entries": 0}

        # 1. Fetch current prices and check stops
        current_prices: dict[str, float] = {}
        if self.price_fetcher is not None:
            current_prices = self.price_fetcher()

        if current_prices:
            stop_fills = self.broker.check_stops(current_prices)
            summary["stops"] = [f.model_dump(mode="json") for f in stop_fills]

        # 2. Fetch latest forecasts and check forecast exits
        forecasts: dict[str, ProbForecast] = {}
        if self.forecast_fetcher is not None:
            forecasts = self.forecast_fetcher()

        if forecasts:
            forecast_fills = self.broker.check_forecast_exits(forecasts)
            summary["forecast_exits"] = [f.model_dump(mode="json") for f in forecast_fills]

        # 3. Evaluate new entry signals
        if self.entry_evaluator is not None:
            pre_count = len(self.broker.inventory)
            self.entry_evaluator(self.broker)
            summary["entries"] = len(self.broker.inventory) - pre_count

        self._cycle += 1
        self._save_state()
        return summary

    def run_loop(self) -> None:
        """Run scanning loop until stopped or *max_cycles* reached."""

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
                LOGGER.info("Cycle %d complete: %s", self._cycle, summary)

                if 0 < self.max_cycles <= self._cycle:
                    LOGGER.info("Reached max_cycles=%d — exiting", self.max_cycles)
                    break

                if self._running:
                    time.sleep(self.interval_seconds)
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def _save_state(self) -> None:
        """Persist scanner state to disk."""

        state = {
            "cycle": self._cycle,
            "bankroll": self.broker.bankroll,
            "positions": {k: v.model_dump(mode="json") for k, v in self.broker.positions.items()},
            "inventory_count": len(self.broker.inventory),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, default=str))

    @classmethod
    def load_state(cls, path: Path) -> dict[str, Any] | None:
        """Load previously saved scanner state."""

        if not path.exists():
            return None
        return json.loads(path.read_text())  # type: ignore[no-any-return]
