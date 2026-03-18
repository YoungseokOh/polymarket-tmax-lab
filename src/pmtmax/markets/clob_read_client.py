"""Public CLOB read client."""

from __future__ import annotations

from typing import Any, cast

from pmtmax.http import CachedHttpClient


class ClobReadClient:
    """Read-only wrapper for public CLOB endpoints."""

    def __init__(self, http: CachedHttpClient, base_url: str) -> None:
        self.http = http
        self.base_url = base_url.rstrip("/")

    def get_book(self, token_id: str) -> dict[str, Any]:
        """Fetch order book for a token."""

        payload = self.http.get_json(
            f"{self.base_url}/book",
            params={"token_id": token_id},
            use_cache=False,
        )
        return cast(dict[str, Any], payload)

    def get_price(self, token_id: str, side: str = "buy") -> dict[str, Any]:
        """Fetch price quote for a token and side."""

        payload = self.http.get_json(
            f"{self.base_url}/price",
            params={"token_id": token_id, "side": side},
            use_cache=False,
        )
        return cast(dict[str, Any], payload)

    def get_prices_history(
        self,
        market: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
        fidelity: int = 60,
    ) -> dict[str, Any]:
        """Fetch public price history."""

        params: dict[str, Any] = {"market": market, "fidelity": fidelity}
        if start_ts is not None:
            params["startTs"] = start_ts
        if end_ts is not None:
            params["endTs"] = end_ts
        payload = self.http.get_json(
            f"{self.base_url}/prices-history",
            params=params,
            use_cache=False,
        )
        return cast(dict[str, Any], payload)

    def get_last_trade_price(self, token_id: str) -> dict[str, Any]:
        """Fetch the last trade price."""

        payload = self.http.get_json(
            f"{self.base_url}/last-trade-price",
            params={"token_id": token_id},
            use_cache=False,
        )
        return cast(dict[str, Any], payload)
