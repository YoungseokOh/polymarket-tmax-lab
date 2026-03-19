"""Gamma API client."""

from __future__ import annotations

from typing import Any, cast

from pmtmax.http import CachedHttpClient


class GammaClient:
    """Client for Polymarket Gamma market discovery."""

    def __init__(self, http: CachedHttpClient, base_url: str) -> None:
        self.http = http
        self.base_url = base_url.rstrip("/")

    def fetch_markets(
        self,
        *,
        active: bool | None = None,
        closed: bool | None = None,
        limit: int = 100,
        offset: int = 0,
        archived: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch market metadata from Gamma."""

        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        if archived is not None:
            params["archived"] = str(archived).lower()
        payload = self.http.get_json(f"{self.base_url}/markets", params=params, use_cache=False)
        return cast(list[dict[str, Any]], payload)

    def fetch_events(
        self,
        *,
        active: bool | None = None,
        closed: bool | None = None,
        tag_slug: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Fetch grouped event metadata from Gamma."""

        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        if tag_slug:
            params["tag_slug"] = tag_slug
        payload = self.http.get_json(f"{self.base_url}/events", params=params, use_cache=False)
        return cast(list[dict[str, Any]], payload)
