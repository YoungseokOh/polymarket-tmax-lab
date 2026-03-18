"""Firebase Storage mirror helpers for warehouse backups."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from pmtmax.storage.schemas import RemoteSyncManifest


class FirebaseMirror:
    """Incrementally mirror parquet/raw/manifest files into a Firebase bucket."""

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

    def sync(
        self,
        *,
        parquet_root: Path,
        raw_root: Path,
        manifest_root: Path,
        dry_run: bool = True,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Build or execute a Firebase Storage sync manifest."""

        if not self.bucket_name:
            msg = "Missing Firebase bucket name. Set PMTMAX_FIREBASE_BUCKET_NAME."
            raise RuntimeError(msg)
        files = self._collect_files(parquet_root=parquet_root, raw_root=raw_root, manifest_root=manifest_root)
        if limit is not None:
            files = files[:limit]
        uploaded: list[str] = []
        skipped: list[str] = []
        if not dry_run:
            client = self._build_client()
            bucket = client.bucket(self.bucket_name)
            for local_path, remote_path in files:
                blob = bucket.blob(remote_path)
                if blob.exists(client) and blob.size == local_path.stat().st_size:
                    skipped.append(remote_path)
                    continue
                blob.upload_from_filename(str(local_path))
                uploaded.append(remote_path)
        else:
            uploaded = [remote_path for _, remote_path in files]
        manifest = RemoteSyncManifest(
            generated_at=dt.datetime.now(tz=dt.UTC),
            backend="firebase",
            bucket_name=self.bucket_name,
            prefix=self.prefix,
            uploaded_files=uploaded,
            skipped_files=skipped,
            dry_run=dry_run,
        )
        return manifest.model_dump(mode="json")

    def _collect_files(
        self,
        *,
        parquet_root: Path,
        raw_root: Path,
        manifest_root: Path,
    ) -> list[tuple[Path, str]]:
        pairs: list[tuple[Path, str]] = []
        parquet_targets = [parquet_root / name for name in ("bronze", "silver", "gold")]
        for root in parquet_targets:
            if not root.exists():
                continue
            for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
                remote_path = f"{self.prefix}/parquet/{path.relative_to(parquet_root).as_posix()}"
                pairs.append((path, remote_path))
        for root, category in ((raw_root, "raw"), (manifest_root, "manifests")):
            if not root.exists():
                continue
            for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
                remote_path = f"{self.prefix}/{category}/{path.relative_to(root).as_posix()}"
                pairs.append((path, remote_path))
        return pairs

    def _build_client(self) -> Any:
        try:
            from google.cloud import storage  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised by runtime path
            msg = "google-cloud-storage is required for non-dry-run Firebase sync."
            raise RuntimeError(msg) from exc
        if self.credentials_json:
            return storage.Client.from_service_account_json(self.credentials_json)
        return storage.Client()
