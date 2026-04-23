#!/usr/bin/env python3
"""Report workspace pipeline status from logs and local artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
WORKSPACES = {"ops_daily", "historical_real", "recent_core_eval"}


def workspace_paths(workspace: str) -> dict[str, Path]:
    artifacts_root = REPO / "artifacts" / "workspaces" / workspace
    data_root = REPO / "data" / "workspaces" / workspace
    return {
        "artifacts_root": artifacts_root,
        "data_root": data_root,
        "log_dir": artifacts_root / "batch_logs",
        "status_path": artifacts_root / "batch_logs" / ".pipeline_status.json",
        "duckdb_path": data_root / "duckdb" / "warehouse.duckdb",
        "dataset_path": data_root / "parquet" / "gold" / "historical_training_set.parquet",
        "v2_dataset_path": data_root / "parquet" / "gold" / "v2" / "historical_training_set.parquet",
    }


def count_snapshot_file(path: Path) -> int | None:
    if not path.exists():
        return None
    marker = b'"captured_at"'
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(marker)
    return count


def parquet_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(path).metadata
        return int(metadata.num_rows) if metadata is not None else None
    except Exception:
        return None


def running_pids(pattern: str) -> list[int]:
    result = subprocess.run(  # noqa: S603
        ["/usr/bin/pgrep", "-f", pattern],
        capture_output=True,
        check=False,
        text=True,
    )
    return [int(pid) for pid in result.stdout.split() if pid.strip().isdigit()]


def pid_workspace(pid: int) -> str | None:
    environ = Path("/proc") / str(pid) / "environ"
    try:
        raw = environ.read_bytes()
    except OSError:
        return None
    for item in raw.split(b"\0"):
        if item.startswith(b"PMTMAX_WORKSPACE_NAME="):
            return item.split(b"=", 1)[1].decode("utf-8", errors="replace")
    return None


def backfill_pids_for_workspace(workspace: str) -> tuple[list[int], list[int]]:
    all_pids = running_pids("pmtmax backfill-forecasts")
    workspace_pids: list[int] = []
    unscoped_pids: list[int] = []
    for pid in all_pids:
        proc_workspace = pid_workspace(pid)
        if proc_workspace == workspace:
            workspace_pids.append(pid)
        elif proc_workspace is None:
            unscoped_pids.append(pid)
    return workspace_pids, unscoped_pids


def read_jsonl_tail(path: Path, *, max_lines: int = 1000) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(errors="replace").splitlines()[-max_lines:]
    except OSError:
        return []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def latest_backfill_log(log_dir: Path) -> Path | None:
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def backfill_progress(log_dir: Path, *, inventory_path: Path | None) -> dict[str, Any]:
    log_path = latest_backfill_log(log_dir)
    total = count_snapshot_file(inventory_path) if inventory_path is not None else None
    progress: dict[str, Any] = {
        "status": "missing_log",
        "log": None,
        "processed": 0,
        "total": total,
        "pct": 0.0,
        "single_run_ok": 0,
        "single_run_err": 0,
        "last_error": None,
        "updated_at": None,
    }
    if log_path is None:
        return progress

    progress["status"] = "log_found"
    progress["log"] = display_path(log_path)
    progress["updated_at"] = datetime.fromtimestamp(log_path.stat().st_mtime, tz=UTC).isoformat()
    rows = read_jsonl_tail(log_path)
    for row in rows:
        message = str(row.get("message", ""))
        level = str(row.get("level", ""))
        if level in {"ERROR", "CRITICAL"} or "error" in message.lower():
            progress["last_error"] = row.get("error") or row.get("exception") or message
        if message == "backfill_forecasts checkpoint":
            processed = int(row.get("processed", progress["processed"]) or 0)
            progress["processed"] = max(int(progress["processed"]), processed)
            progress["total"] = int(row.get("total", progress["total"]) or progress["total"] or 0) or progress["total"]
            progress["pct"] = float(row.get("pct", progress["pct"]) or 0.0)
            progress["single_run_ok"] = max(
                int(progress["single_run_ok"]),
                int(row.get("single_run_ok", 0) or 0),
            )
            progress["single_run_err"] = max(
                int(progress["single_run_err"]),
                int(row.get("single_run_err", 0) or 0),
            )
        elif message == "single_run market done":
            progress["single_run_ok"] = max(
                int(progress["single_run_ok"]),
                int(row.get("cumulative_ok", 0) or 0),
            )
            progress["single_run_err"] = max(
                int(progress["single_run_err"]),
                int(row.get("cumulative_err", 0) or 0),
            )
    if progress["total"] and progress["processed"]:
        progress["pct"] = round(100.0 * int(progress["processed"]) / int(progress["total"]), 2)
    return progress


def load_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"stage": "unknown", "pct": None, "updated_at": None}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"stage": "unreadable", "pct": None, "updated_at": None}
    return payload if isinstance(payload, dict) else {"stage": "invalid", "pct": None, "updated_at": None}


def build_status(*, workspace: str, inventory_path: Path | None) -> dict[str, Any]:
    paths = workspace_paths(workspace)
    backfill_pids, unscoped_backfill_pids = backfill_pids_for_workspace(workspace)
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "workspace": workspace,
        "status_file": str(paths["status_path"].relative_to(REPO)),
        "pipeline_status": load_status(paths["status_path"]),
        "backfill_running": bool(backfill_pids),
        "backfill_pids": backfill_pids,
        "unscoped_backfill_pids": unscoped_backfill_pids,
        "backfill": backfill_progress(paths["log_dir"], inventory_path=inventory_path),
        "warehouse_exists": paths["duckdb_path"].exists(),
        "dataset_rows": parquet_rows(paths["dataset_path"]),
        "v2_dataset_rows": parquet_rows(paths["v2_dataset_path"]),
    }


def print_table(status: dict[str, Any]) -> None:
    backfill = status["backfill"]
    pipeline = status["pipeline_status"]
    print("=" * 72)
    print(f"  PMTMAX pipeline status [{status['generated_at']}]")
    print("=" * 72)
    print(f"workspace:        {status['workspace']}")
    print(f"pipeline stage:   {pipeline.get('stage')} ({pipeline.get('pct')}%)")
    print(f"backfill running: {status['backfill_running']}  pids={status['backfill_pids']}")
    if status.get("unscoped_backfill_pids"):
        print(f"unscoped pids:    {status['unscoped_backfill_pids']} (check workspace safety)")
    print(f"backfill log:     {backfill.get('log')}")
    print(
        "backfill rows:    "
        f"{int(backfill.get('processed') or 0):,}/{int(backfill.get('total') or 0):,} "
        f"({float(backfill.get('pct') or 0.0):.2f}%)"
    )
    print(
        "single-run:       "
        f"ok={int(backfill.get('single_run_ok') or 0):,} "
        f"err={int(backfill.get('single_run_err') or 0):,}"
    )
    print(f"warehouse exists: {status['warehouse_exists']}")
    print(f"dataset rows:     {status['dataset_rows']}")
    print(f"v2 dataset rows:  {status['v2_dataset_rows']}")
    if backfill.get("last_error"):
        print(f"last error:       {backfill['last_error']}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="historical_real", choices=sorted(WORKSPACES))
    parser.add_argument(
        "--inventory-path",
        type=Path,
        default=Path("configs/market_inventory/historical_temperature_snapshots.json"),
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory_path = args.inventory_path if args.inventory_path else None
    status = build_status(workspace=args.workspace, inventory_path=inventory_path)
    if args.as_json:
        print(json.dumps(status, indent=2, sort_keys=True, default=str))
    else:
        print_table(status)


if __name__ == "__main__":
    main()
