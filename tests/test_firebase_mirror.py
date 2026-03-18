from __future__ import annotations

from pathlib import Path

import pytest

from pmtmax.storage.firebase_mirror import FirebaseMirror


def test_firebase_mirror_collects_expected_paths(tmp_path: Path) -> None:
    parquet_root = tmp_path / "parquet"
    raw_root = tmp_path / "raw"
    manifest_root = tmp_path / "manifests"
    (parquet_root / "gold").mkdir(parents=True)
    (raw_root / "bronze").mkdir(parents=True)
    manifest_root.mkdir(parents=True)
    (parquet_root / "gold" / "dataset.parquet").write_text("parquet")
    (raw_root / "bronze" / "snapshot.json").write_text("{}")
    (manifest_root / "warehouse_manifest.json").write_text("{}")

    mirror = FirebaseMirror(bucket_name="demo-bucket", prefix="pmtmax")
    payload = mirror.sync(
        parquet_root=parquet_root,
        raw_root=raw_root,
        manifest_root=manifest_root,
        dry_run=True,
    )

    assert payload["bucket_name"] == "demo-bucket"
    assert len(payload["uploaded_files"]) == 3
    assert "pmtmax/parquet/gold/dataset.parquet" in payload["uploaded_files"]
    assert "pmtmax/raw/bronze/snapshot.json" in payload["uploaded_files"]
    assert "pmtmax/manifests/warehouse_manifest.json" in payload["uploaded_files"]


def test_firebase_mirror_requires_bucket_name(tmp_path: Path) -> None:
    mirror = FirebaseMirror(bucket_name="")
    with pytest.raises(RuntimeError):
        mirror.sync(
            parquet_root=tmp_path / "parquet",
            raw_root=tmp_path / "raw",
            manifest_root=tmp_path / "manifests",
            dry_run=True,
        )
