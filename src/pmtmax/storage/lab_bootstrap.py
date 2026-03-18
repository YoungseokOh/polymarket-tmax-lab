"""Helpers for one-shot lab bootstrapping, seed bundles, and legacy-run cleanup."""

from __future__ import annotations

import datetime as dt
import shutil
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd

from pmtmax.storage.schemas import (
    BootstrapManifest,
    LegacyRunEntry,
    LegacyRunInventory,
    SeedManifest,
)
from pmtmax.storage.warehouse import TABLE_KEYS, DataWarehouse
from pmtmax.utils import dump_json, load_json, stable_hash

CANONICAL_RAW_CHILDREN = {"bronze"}
CANONICAL_PARQUET_CHILDREN = {"bronze", "silver", "gold"}
STANDARD_ACTIVE_PARQUETS = {
    "gold/historical_training_set.parquet",
    "gold/historical_training_sequence.parquet",
}


def collect_seed_files(
    *,
    data_root: Path,
    raw_root: Path,
    parquet_root: Path,
    manifest_root: Path,
) -> list[Path]:
    """Collect canonical raw/parquet/manifest files for a portable seed bundle."""

    candidates: list[Path] = []
    roots = [
        raw_root / "bronze",
        parquet_root / "bronze",
        parquet_root / "silver",
        parquet_root / "gold",
        manifest_root,
    ]
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(path for path in root.rglob("*") if path.is_file())
    unique = sorted({path.resolve() for path in candidates})
    return [path for path in unique if data_root.resolve() in path.parents or path == data_root.resolve()]


def export_seed_bundle(
    *,
    data_root: Path,
    raw_root: Path,
    parquet_root: Path,
    manifest_root: Path,
    seed_path: Path,
) -> SeedManifest:
    """Create a tarball containing canonical raw/parquet/manifests for fast bootstrap."""

    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_manifest_path = manifest_root / "seed_manifest.json"
    files = collect_seed_files(
        data_root=data_root,
        raw_root=raw_root,
        parquet_root=parquet_root,
        manifest_root=manifest_root,
    )
    provisional_manifest = SeedManifest(
        generated_at=dt.datetime.now(tz=dt.UTC),
        archive_path=str(seed_path),
        data_root=str(data_root),
        included_files=[path.relative_to(data_root).as_posix() for path in files],
        file_count=len(files),
    )
    dump_json(seed_manifest_path, provisional_manifest.model_dump(mode="json"))
    files = collect_seed_files(
        data_root=data_root,
        raw_root=raw_root,
        parquet_root=parquet_root,
        manifest_root=manifest_root,
    )
    manifest = SeedManifest(
        generated_at=provisional_manifest.generated_at,
        archive_path=str(seed_path),
        data_root=str(data_root),
        included_files=[path.relative_to(data_root).as_posix() for path in files],
        file_count=len(files),
    )
    dump_json(seed_manifest_path, manifest.model_dump(mode="json"))
    files = collect_seed_files(
        data_root=data_root,
        raw_root=raw_root,
        parquet_root=parquet_root,
        manifest_root=manifest_root,
    )
    with tarfile.open(seed_path, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(data_root).as_posix())
    return manifest


def restore_seed_bundle(
    *,
    seed_path: Path,
    data_root: Path,
) -> SeedManifest:
    """Extract a seed tarball into the local data directory."""

    if not seed_path.exists():
        msg = f"Seed archive does not exist: {seed_path}"
        raise FileNotFoundError(msg)
    data_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(seed_path, "r:gz") as archive:
        for member in archive.getmembers():
            destination = (data_root / member.name).resolve()
            try:
                destination.relative_to(data_root.resolve())
            except ValueError as exc:
                msg = f"Unsafe seed archive member: {member.name}"
                raise RuntimeError(msg) from exc
        for member in archive.getmembers():
            archive.extract(member, data_root, filter="data")
    manifest_path = data_root / "manifests" / "seed_manifest.json"
    if manifest_path.exists():
        return SeedManifest.model_validate(load_json(manifest_path))
    return SeedManifest(
        generated_at=dt.datetime.now(tz=dt.UTC),
        archive_path=str(seed_path),
        data_root=str(data_root),
        included_files=[],
        file_count=0,
    )


def restore_warehouse_from_seed(
    *,
    warehouse: DataWarehouse,
    parquet_root: Path,
    manifest_root: Path,
) -> dict[str, int]:
    """Rebuild the local warehouse from canonical parquet mirrors and manifest metadata."""

    for table_name in warehouse.list_tables():
        if table_name in TABLE_KEYS:
            warehouse.duckdb_store.execute(f"drop table if exists {table_name}")  # noqa: S608

    restored: dict[str, int] = {}
    for table_name, parquet_path in _restore_table_map(parquet_root=parquet_root, manifest_root=manifest_root).items():
        frame = pd.read_parquet(parquet_path)
        warehouse.duckdb_store.write_frame(table_name, frame)
        restored[table_name] = len(frame)
    warehouse.compact()
    return restored


def inventory_legacy_runs(
    *,
    raw_root: Path,
    parquet_root: Path,
    manifest_root: Path,
) -> LegacyRunInventory:
    """Inventory non-canonical raw/parquet run artifacts that can be archived."""

    entries: list[LegacyRunEntry] = []
    manifest_path = manifest_root / "warehouse_manifest.json"
    active_roots = [
        str(raw_root / "bronze"),
        str(parquet_root / "bronze"),
        str(parquet_root / "silver"),
        str(parquet_root / "gold"),
    ]

    if raw_root.exists():
        for path in sorted(raw_root.iterdir()):
            if path.name in CANONICAL_RAW_CHILDREN:
                continue
            entries.append(_legacy_entry(path, category="raw", reason="noncanonical_raw_root"))

    if parquet_root.exists():
        for path in sorted(parquet_root.iterdir()):
            if path.is_dir() and path.name in CANONICAL_PARQUET_CHILDREN:
                continue
            reason = "legacy_parquet_dir" if path.is_dir() else "legacy_parquet_file"
            entries.append(_legacy_entry(path, category="parquet", reason=reason))

    return LegacyRunInventory(
        generated_at=dt.datetime.now(tz=dt.UTC),
        manifest_path=str(manifest_path) if manifest_path.exists() else None,
        active_roots=active_roots,
        entries=entries,
    )


def archive_legacy_runs(
    *,
    inventory: LegacyRunInventory,
    archive_root: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Move inventoried legacy raw/parquet runs into the archive tree."""

    archived_paths: list[str] = []
    for entry in inventory.entries:
        source = Path(entry.path)
        if not source.exists():
            continue
        destination_root = archive_root / "legacy-runs" / entry.category
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / source.name
        if destination.exists():
            suffix = stable_hash(f"{source.name}|{dt.datetime.now(tz=dt.UTC).isoformat()}")[:8]
            destination = destination_root / f"{source.stem}_{suffix}{source.suffix}"
        archived_paths.append(str(destination))
        if not dry_run:
            shutil.move(str(source), str(destination))
    return {
        "generated_at": dt.datetime.now(tz=dt.UTC).isoformat(),
        "dry_run": dry_run,
        "inventory_size": len(inventory.entries),
        "archived_paths": archived_paths,
    }


def build_bootstrap_manifest(
    *,
    seed_path: Path | None,
    seed_restored: bool,
    archived_legacy_paths: list[str],
    steps: list[str],
    output_paths: dict[str, str],
    warehouse_counts: dict[str, int],
) -> BootstrapManifest:
    """Create a serializable one-shot bootstrap summary."""

    return BootstrapManifest(
        generated_at=dt.datetime.now(tz=dt.UTC),
        seed_path=str(seed_path) if seed_path is not None else None,
        seed_restored=seed_restored,
        archived_legacy_paths=archived_legacy_paths,
        steps=steps,
        output_paths=output_paths,
        warehouse_counts=warehouse_counts,
    )


def _legacy_entry(path: Path, *, category: str, reason: str) -> LegacyRunEntry:
    file_count, size_bytes = _path_metrics(path)
    return LegacyRunEntry(
        category=category,  # type: ignore[arg-type]
        path=str(path),
        kind="dir" if path.is_dir() else "file",
        reason=reason,
        file_count=file_count,
        size_bytes=size_bytes,
    )


def _path_metrics(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return 1, path.stat().st_size
    file_count = 0
    size_bytes = 0
    for child in path.rglob("*"):
        if child.is_file():
            file_count += 1
            size_bytes += child.stat().st_size
    return file_count, size_bytes


def _restore_table_map(*, parquet_root: Path, manifest_root: Path) -> dict[str, Path]:
    manifest_path = manifest_root / "warehouse_manifest.json"
    mapping: dict[str, Path] = {}
    if manifest_path.exists():
        payload = load_json(manifest_path)
        for table_name, metadata in payload.get("tables", {}).items():
            relative_path = metadata.get("parquet_path")
            if not relative_path:
                continue
            candidate = parquet_root / str(relative_path)
            if candidate.exists():
                mapping[table_name] = candidate
    if mapping:
        return mapping
    for path in sorted(parquet_root.rglob("*.parquet")):
        relative = path.relative_to(parquet_root).as_posix()
        if relative in STANDARD_ACTIVE_PARQUETS:
            continue
        table_name = path.stem
        if table_name in TABLE_KEYS:
            mapping[table_name] = path
    return mapping
