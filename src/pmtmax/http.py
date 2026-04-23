"""HTTP client with caching and retries."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import httpx

from pmtmax.logging_utils import get_logger
from pmtmax.utils import stable_hash

LOGGER = get_logger(__name__)
_T = TypeVar("_T")


def _is_retriable_http_error(exc: BaseException) -> bool:
    """Return whether an HTTP exception should be retried."""

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return isinstance(exc, httpx.HTTPError)


class CachedHttpClient:
    """Thin HTTP wrapper that caches JSON and text responses on disk."""

    def __init__(
        self,
        cache_dir: Path,
        timeout_seconds: float = 30.0,
        retries: int = 3,
        retry_wait_min_seconds: float = 4.0,
        retry_wait_max_seconds: float = 120.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds
        self.retries = max(0, int(retries))
        self.retry_wait_min_seconds = max(0.0, float(retry_wait_min_seconds))
        self.retry_wait_max_seconds = max(self.retry_wait_min_seconds, float(retry_wait_max_seconds))
        self.client = httpx.Client(
            timeout=timeout_seconds,
            headers={"User-Agent": "polymarket-tmax-lab/0.1.0"},
            follow_redirects=True,
        )

    def _cache_path(self, key: str, suffix: str) -> Path:
        return self.cache_dir / f"{stable_hash(key)}.{suffix}"

    @staticmethod
    def _cache_key(url: str, payload: dict[str, Any] | None, payload_key: str) -> str:
        return json.dumps({"url": url, payload_key: payload}, sort_keys=True, default=str)

    def _request_cache_path(self, url: str, suffix: str, *, payload: dict[str, Any] | None, payload_key: str) -> Path:
        return self._cache_path(self._cache_key(url, payload, payload_key), suffix)

    def _retry_delay_seconds(self, retry_index: int) -> float:
        if self.retry_wait_min_seconds <= 0:
            return 0.0
        return min(self.retry_wait_max_seconds, self.retry_wait_min_seconds * (2 ** (retry_index - 1)))

    def _request_with_retry(self, operation: Callable[[], _T], *, method: str, url: str) -> _T:
        max_attempts = self.retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                return operation()
            except Exception as exc:  # noqa: BLE001
                if not _is_retriable_http_error(exc) or attempt >= max_attempts:
                    raise
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                delay = self._retry_delay_seconds(attempt)
                LOGGER.warning(
                    "HTTP %s %s failed on attempt %s/%s%s; retrying in %.1fs",
                    method,
                    url,
                    attempt,
                    max_attempts,
                    f" (status={status_code})" if status_code is not None else "",
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
        msg = f"unreachable retry loop for {method} {url}"
        raise RuntimeError(msg)

    def get_json(self, url: str, params: dict[str, Any] | None = None, use_cache: bool = True, cache_ttl_seconds: float | None = None) -> Any:
        """Fetch JSON with optional disk cache and optional TTL expiry."""

        cache_path = self._request_cache_path(url, "json", payload=params, payload_key="params")
        if (
            use_cache
            and cache_path.exists()
            and (cache_ttl_seconds is None or (time.time() - cache_path.stat().st_mtime) < cache_ttl_seconds)
        ):
            return json.loads(cache_path.read_text())

        def _load_response() -> httpx.Response:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            return response

        response = self._request_with_retry(_load_response, method="GET", url=url)
        payload = response.json()
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return payload

    def post_json(self, url: str, data: dict[str, Any] | None = None, use_cache: bool = True) -> Any:
        """POST form data and cache the JSON response on disk."""

        cache_path = self._request_cache_path(url, "json", payload=data, payload_key="data")
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text())

        def _load_response() -> httpx.Response:
            response = self.client.post(url, data=data)
            response.raise_for_status()
            return response

        response = self._request_with_retry(_load_response, method="POST", url=url)
        payload = response.json()
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return payload

    def get_text(self, url: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> str:
        """Fetch text with optional disk cache."""

        cache_path = self._request_cache_path(url, "txt", payload=params, payload_key="params")
        if use_cache and cache_path.exists():
            return cache_path.read_text()

        def _load_response() -> httpx.Response:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            return response

        response = self._request_with_retry(_load_response, method="GET", url=url)
        text = response.text
        cache_path.write_text(text)
        return text

    def load_cached_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any | None:
        """Return cached JSON without issuing a network request."""

        cache_path = self._request_cache_path(url, "json", payload=params, payload_key="params")
        if not cache_path.exists():
            return None
        return json.loads(cache_path.read_text())

    def load_cached_text(self, url: str, *, params: dict[str, Any] | None = None) -> str | None:
        """Return cached text without issuing a network request."""

        cache_path = self._request_cache_path(url, "txt", payload=params, payload_key="params")
        if not cache_path.exists():
            return None
        return cache_path.read_text()

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self.client.close()
