"""HTTP client with caching and retries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from pmtmax.logging_utils import get_logger
from pmtmax.utils import stable_hash

LOGGER = get_logger(__name__)


def _is_retriable_http_error(exc: BaseException) -> bool:
    """Return whether an HTTP exception should be retried."""

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.HTTPError)


class CachedHttpClient:
    """Thin HTTP wrapper that caches JSON and text responses on disk."""

    def __init__(self, cache_dir: Path, timeout_seconds: float = 30.0, retries: int = 3) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds
        self.retries = retries
        self.client = httpx.Client(
            timeout=timeout_seconds,
            headers={"User-Agent": "polymarket-tmax-lab/0.1.0"},
            follow_redirects=True,
        )

    def _cache_path(self, key: str, suffix: str) -> Path:
        return self.cache_dir / f"{stable_hash(key)}.{suffix}"

    @retry(
        retry=retry_if_exception(_is_retriable_http_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def get_json(self, url: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> Any:
        """Fetch JSON with optional disk cache."""

        key = json.dumps({"url": url, "params": params}, sort_keys=True, default=str)
        cache_path = self._cache_path(key, "json")
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text())

        response = self.client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return payload

    @retry(
        retry=retry_if_exception(_is_retriable_http_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def get_text(self, url: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> str:
        """Fetch text with optional disk cache."""

        key = json.dumps({"url": url, "params": params}, sort_keys=True, default=str)
        cache_path = self._cache_path(key, "txt")
        if use_cache and cache_path.exists():
            return cache_path.read_text()

        response = self.client.get(url, params=params)
        response.raise_for_status()
        text = response.text
        cache_path.write_text(text)
        return text

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self.client.close()
