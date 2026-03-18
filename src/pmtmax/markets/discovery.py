"""Market discovery orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pmtmax.logging_utils import get_logger
from pmtmax.markets.gamma_client import GammaClient
from pmtmax.markets.market_filter import is_temperature_max_market
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.markets.repository import snapshot_from_market
from pmtmax.storage.schemas import MarketSnapshot

LOGGER = get_logger(__name__)


@dataclass
class DiscoveryResult:
    raw_markets: list[dict[str, Any]]
    snapshots: list[MarketSnapshot]

    @property
    def specs(self) -> list[MarketSpec]:
        """Return parsed specs for successfully parsed snapshots."""

        return [snapshot.spec for snapshot in self.snapshots if snapshot.spec is not None]


class MarketDiscoveryService:
    """Discover and parse temperature-max markets."""

    def __init__(self, gamma_client: GammaClient, max_pages: int = 20) -> None:
        self.gamma_client = gamma_client
        self.max_pages = max_pages

    def discover(self, *, active: bool | None = None, closed: bool | None = None) -> DiscoveryResult:
        """Fetch pages from Gamma and return parsed supported markets."""

        collected: list[dict[str, Any]] = []
        snapshots: list[MarketSnapshot] = []
        for page in range(self.max_pages):
            offset = page * 100
            batch = self.gamma_client.fetch_markets(active=active, closed=closed, limit=100, offset=offset)
            if not batch:
                break
            collected.extend(batch)
            for market in batch:
                if not is_temperature_max_market(market):
                    continue
                snapshot = snapshot_from_market(market)
                if snapshot.parse_error:
                    LOGGER.warning(
                        "market_parse_failed",
                        extra={"market_id": market.get("id"), "slug": market.get("slug"), "error": snapshot.parse_error},
                    )
                snapshots.append(snapshot)
            if len(batch) < 100:
                break
        return DiscoveryResult(raw_markets=collected, snapshots=snapshots)
