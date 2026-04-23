from __future__ import annotations

import httpx
import pytest

from pmtmax.http import CachedHttpClient


def test_cached_http_client_retries_get_json_with_configured_limit(tmp_path, monkeypatch) -> None:
    http = CachedHttpClient(
        tmp_path / "cache",
        retries=2,
        retry_wait_min_seconds=0.0,
        retry_wait_max_seconds=0.0,
    )
    attempts = {"count": 0}

    def fake_get(url: str, params: dict[str, object] | None = None) -> httpx.Response:
        attempts["count"] += 1
        request = httpx.Request("GET", url, params=params)
        if attempts["count"] < 3:
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError("rate limited", request=request, response=response)
        return httpx.Response(200, request=request, json={"ok": True})

    monkeypatch.setattr(http.client, "get", fake_get)

    try:
        payload = http.get_json("https://example.test/data", use_cache=False)
    finally:
        http.close()

    assert payload == {"ok": True}
    assert attempts["count"] == 3


def test_cached_http_client_stops_after_configured_retries(tmp_path, monkeypatch) -> None:
    http = CachedHttpClient(
        tmp_path / "cache",
        retries=0,
        retry_wait_min_seconds=0.0,
        retry_wait_max_seconds=0.0,
    )
    attempts = {"count": 0}

    def fake_get(url: str, params: dict[str, object] | None = None) -> httpx.Response:
        attempts["count"] += 1
        request = httpx.Request("GET", url, params=params)
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    monkeypatch.setattr(http.client, "get", fake_get)

    try:
        with pytest.raises(httpx.HTTPStatusError):
            http.get_json("https://example.test/data", use_cache=False)
    finally:
        http.close()

    assert attempts["count"] == 1
