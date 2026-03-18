"""Raw artifact archive helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pmtmax.storage.schemas import RawArtifactRef
from pmtmax.utils import stable_hash_bytes


class RawStore:
    """Persist raw JSON/HTML/text payloads under a deterministic local layout."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, relative_path: str, payload: Any) -> RawArtifactRef:
        """Write JSON payload to the raw archive."""

        content = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
        return self._write_bytes(relative_path, content, "application/json")

    def write_text(self, relative_path: str, payload: str, media_type: str = "text/plain") -> RawArtifactRef:
        """Write text payload to the raw archive."""

        return self._write_bytes(relative_path, payload.encode("utf-8"), media_type)

    def read_json(self, relative_path: str) -> Any:
        """Load a JSON artifact from the raw archive."""

        return json.loads((self.root / relative_path).read_text())

    def read_text(self, relative_path: str) -> str:
        """Load a text artifact from the raw archive."""

        return (self.root / relative_path).read_text()

    def _write_bytes(self, relative_path: str, payload: bytes, media_type: str) -> RawArtifactRef:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return RawArtifactRef(
            relative_path=relative_path,
            content_hash=stable_hash_bytes(payload),
            media_type=media_type,
        )
