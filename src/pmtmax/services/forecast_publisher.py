"""Publish forecast summaries to Firebase Storage."""

from __future__ import annotations

import json
from typing import Any

from pmtmax.logging_utils import get_logger
from pmtmax.storage.schemas import ForecastSummary

LOGGER = get_logger(__name__)


class ForecastPublisher:
    """Upload ForecastSummary JSON to Firebase Storage."""

    def __init__(
        self,
        *,
        bucket_name: str,
        prefix: str = "pmtmax",
        credentials_json: str | None = None,
    ) -> None:
        self.bucket_name = bucket_name
        self.prefix = prefix.strip("/")
        self.credentials_json = credentials_json

    def publish(self, summaries: list[ForecastSummary], *, dry_run: bool = True) -> dict[str, Any]:
        """Upload forecast summaries to Firebase Storage."""

        uploaded: list[str] = []

        for summary in summaries:
            date_str = summary.target_local_date.isoformat()
            remote_path = f"{self.prefix}/forecasts/{date_str}/{summary.market_id}.json"
            payload = summary.model_dump(mode="json")

            if not dry_run:
                self._upload_json(remote_path, payload)

            uploaded.append(remote_path)
            LOGGER.info("Published forecast: %s", remote_path)

        return {
            "uploaded": uploaded,
            "count": len(uploaded),
            "dry_run": dry_run,
        }

    def _upload_json(self, remote_path: str, payload: dict) -> None:
        """Upload a JSON payload to Firebase Storage."""

        client = self._build_client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(remote_path)
        blob.upload_from_string(
            json.dumps(payload, indent=2, default=str),
            content_type="application/json",
        )

    def _build_client(self) -> Any:
        try:
            from google.cloud import storage  # type: ignore[import-not-found]
        except ImportError as exc:
            msg = "google-cloud-storage is required for Firebase forecast publishing."
            raise RuntimeError(msg) from exc
        if self.credentials_json:
            return storage.Client.from_service_account_json(self.credentials_json)
        return storage.Client()
