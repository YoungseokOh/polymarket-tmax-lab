#!/usr/bin/env python3
"""Durable overnight TMAX price-history recovery supervisor.

Runs one city at a time, records progress in a JSON state file, and can be
restarted safely after agent/session/gateway interruptions. It deliberately
waits for any already-running pmtmax writer before starting the next step.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = "historical_real"
MARKETS = "configs/market_inventory/full_training_set_snapshots.json"
OUT_DIR = ROOT / "artifacts/workspaces/historical_real/quality/overnight_20260513"
LOG_DIR = ROOT / "artifacts/workspaces/historical_real/batch_logs"
STATE_PATH = OUT_DIR / "price_recovery_supervisor_state.json"
LOCK_PATH = OUT_DIR / "price_recovery_supervisor.lock"
CITIES = [
    "NYC",
    "London",
    "Dallas",
    "Atlanta",
    "Seattle",
    "Toronto",
    "Chicago",
    "Miami",
    "Buenos Aires",
]
WRITER_PATTERNS = (
    "pmtmax backfill-price-history",
    "pmtmax materialize-backtest-panel",
    "pmtmax compact-warehouse",
)
SELF_NAME = "tmax_price_recovery_supervisor.py"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: UP017 - keep Python 3.8-compatible when run outside uv


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"started_at": now(), "cities": {}, "events": []}


def save_state(state: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now()
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(STATE_PATH)


def event(state: dict[str, Any], message: str, **extra: Any) -> None:
    item = {"at": now(), "message": message, **extra}
    state.setdefault("events", []).append(item)
    print(f"[{item['at']}] {message} {extra if extra else ''}", flush=True)
    save_state(state)


def acquire_lock() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    return fd


def release_lock(fd: int) -> None:
    try:
        os.close(fd)
    finally:
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass


def process_table() -> list[tuple[int, int, str]]:
    out = subprocess.check_output(["/bin/ps", "-axo", "pid,ppid,command"], text=True)  # noqa: S603 - fixed local command, no user input
    rows: list[tuple[int, int, str]] = []
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) == 3:
            try:
                rows.append((int(parts[0]), int(parts[1]), parts[2]))
            except ValueError:
                continue
    return rows


def active_external_writers() -> list[tuple[int, str]]:
    me = os.getpid()
    rows = process_table()
    # include direct children from this supervisor as internal; all other writers are external
    children = {pid for pid, ppid, _ in rows if ppid == me}
    active: list[tuple[int, str]] = []
    for pid, _ppid, cmd in rows:
        if pid == me or pid in children or SELF_NAME in cmd:
            continue
        if any(pattern in cmd for pattern in WRITER_PATTERNS):
            active.append((pid, cmd))
    return active


def wait_for_external_writers(state: dict[str, Any], *, poll_seconds: int = 120) -> None:
    last_report = 0.0
    while True:
        active = active_external_writers()
        if not active:
            return
        if time.monotonic() - last_report > 600:
            event(
                state,
                "waiting_for_external_writer",
                active=[{"pid": p, "cmd": c[:180]} for p, c in active],
            )
            last_report = time.monotonic()
        time.sleep(poll_seconds)


def run_step(state: dict[str, Any], name: str, command: list[str], log_path: Path) -> None:
    event(state, "step_start", step=name, log=str(log_path))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        log.write((f"\n=== {name} {now()} ===\n").encode())
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)  # noqa: S603 - commands are fixed internal argv lists
        state["current_step"] = name
        state["current_pid"] = proc.pid
        save_state(state)
        rc = proc.wait()
    state.pop("current_pid", None)
    if rc != 0:
        event(state, "step_failed", step=name, returncode=rc, log=str(log_path))
        raise SystemExit(rc)
    event(state, "step_done", step=name, log=str(log_path))


def recover_city(state: dict[str, Any], city: str) -> None:
    city_state = state.setdefault("cities", {}).setdefault(city, {})
    if city_state.get("status") == "done":
        return
    wait_for_external_writers(state)
    safe = city.replace(" ", "_")
    log = LOG_DIR / f"price_history_{safe}_overnight_20260513.log"
    cmd = [
        "scripts/pmtmax-workspace",
        WORKSPACE,
        "uv",
        "run",
        "pmtmax",
        "backfill-price-history",
        "--markets-path",
        MARKETS,
        "--city",
        city,
        "--interval",
        "max",
        "--fidelity",
        "60",
        "--only-missing",
    ]
    city_state["status"] = "running"
    city_state["started_at"] = now()
    save_state(state)
    run_step(state, f"price_history:{city}", cmd, log)
    city_state["status"] = "done"
    city_state["done_at"] = now()
    save_state(state)


def post_process(state: dict[str, Any]) -> None:
    wait_for_external_writers(state)
    if not state.get("materialized"):
        run_step(
            state,
            "materialize_backtest_panel",
            [
                "scripts/pmtmax-workspace",
                WORKSPACE,
                "uv",
                "run",
                "pmtmax",
                "materialize-backtest-panel",
                "--dataset-path",
                "data/workspaces/historical_real/parquet/gold/historical_training_set.parquet",
                "--markets-path",
                MARKETS,
                "--output-name",
                "historical_backtest_panel",
                "--allow-canonical-overwrite",
            ],
            LOG_DIR / "materialize_panel_overnight_20260513.log",
        )
        state["materialized"] = True
        save_state(state)
    if not state.get("coverage_summarized"):
        run_step(
            state,
            "summarize_price_coverage",
            [
                "scripts/pmtmax-workspace",
                WORKSPACE,
                "uv",
                "run",
                "pmtmax",
                "summarize-price-history-coverage",
                "--markets-path",
                MARKETS,
                "--output",
                str(OUT_DIR / "price_coverage_after.json"),
            ],
            LOG_DIR / "price_coverage_overnight_20260513.log",
        )
        state["coverage_summarized"] = True
        save_state(state)
    if not state.get("compacted"):
        run_step(
            state,
            "compact_warehouse",
            ["scripts/pmtmax-workspace", WORKSPACE, "uv", "run", "pmtmax", "compact-warehouse"],
            LOG_DIR / "compact_overnight_20260513.log",
        )
        state["compacted"] = True
        save_state(state)


def main() -> int:
    state = load_state()
    try:
        fd = acquire_lock()
    except FileExistsError:
        print(f"Supervisor lock exists: {LOCK_PATH}", file=sys.stderr)
        return 2
    try:
        event(state, "supervisor_start", pid=os.getpid())
        for city in CITIES:
            recover_city(state, city)
        post_process(state)
        event(state, "supervisor_done")
        return 0
    finally:
        release_lock(fd)


if __name__ == "__main__":
    raise SystemExit(main())
