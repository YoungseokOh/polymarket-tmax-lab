#!/usr/bin/env python3
"""Check Phase 15 pipeline progress (called by user whenever they ask)."""

import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent
STATUS_FILE = REPO_ROOT / "artifacts/batch_logs/.phase15_status.json"
LOG_DIR = REPO_ROOT / "artifacts/batch_logs"

def get_process_status():
    """Check if pipeline processes are still running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "phase15_pipeline|build-dataset|train-advanced|autoresearch"],
            capture_output=True, text=True
        )
        return len(result.stdout.strip().split("\n")) > 0 if result.stdout.strip() else False
    except:
        return False

def get_log_progress(log_file):
    """Extract progress from log file (heuristic)."""
    if not log_file.exists():
        return "0%"

    try:
        with open(log_file, "r") as f:
            lines = f.readlines()

        # build-dataset: look for "Materialized X rows"
        if "build_dataset" in log_file.name:
            for line in reversed(lines[-100:]):  # last 100 lines
                if "Materialized" in line or "rows" in line:
                    return "80-100%"

        # train-advanced: look for epoch progress
        if "train_advanced" in log_file.name:
            for line in reversed(lines[-100:]):
                if "epoch" in line.lower() or "CRPS" in line:
                    return "70-100%"

        # autoresearch: look for candidate count
        if "autoresearch" in log_file.name:
            for line in reversed(lines[-50:]):
                if "candidate" in line.lower():
                    return "50-100%"

        # Generic: file size heuristic
        size = log_file.stat().st_size
        if size > 10_000_000:
            return "90%+"
        elif size > 1_000_000:
            return "60%+"
        elif size > 100_000:
            return "30%+"
        elif size > 10_000:
            return "10%+"
        else:
            return "5%"
    except:
        return "unknown"

def main():
    if not STATUS_FILE.exists():
        print("❌ Status file not found. Pipeline hasn't started yet.")
        sys.exit(0)

    with open(STATUS_FILE) as f:
        status = json.load(f)

    is_running = get_process_status()

    print("=" * 70)
    print("📊 PHASE 15 PIPELINE STATUS")
    print("=" * 70)

    print(f"\n현재 상태: {status.get('stage', 'unknown').upper()}")
    print(f"프로세스: {'🟢 진행 중' if is_running else '🔴 정지됨'}")

    if 'updated_at' in status:
        print(f"마지막 업데이트: {status['updated_at'][:19]}")

    print("\n" + "=" * 70)
    print("📋 단계별 진행:")
    print("=" * 70)

    stages = [
        ("build_dataset", "1️⃣ build-dataset (NWP 다양성)"),
        ("train_advanced", "2️⃣ train-advanced (기준선)"),
        ("autoresearch_init", "3️⃣ autoresearch-init (Phase 15)"),
    ]

    for key, label in stages:
        stage_status = status.get(key, {})
        s = stage_status.get('status', 'pending').upper()

        # Symbol
        if s == 'COMPLETED':
            symbol = "✅"
        elif s == 'RUNNING':
            symbol = "⏳"
        else:
            symbol = "⏸"

        print(f"\n{symbol} {label}")
        print(f"   상태: {s}")

        # Log file progress
        log_pattern = None
        if "build_dataset" in key:
            log_pattern = "phase15_01_build_dataset.log"
        elif "train_advanced" in key:
            log_pattern = "phase15_02_train_advanced.log"
        elif "autoresearch" in key:
            log_pattern = "phase15_03_autoresearch_init.log"

        if log_pattern:
            log_file = LOG_DIR / log_pattern
            progress = get_log_progress(log_file) if log_file.exists() else "0%"
            print(f"   진행률: {progress}")

            # File size
            if log_file.exists():
                size_mb = log_file.stat().st_size / (1024*1024)
                print(f"   로그 크기: {size_mb:.1f} MB")

    print("\n" + "=" * 70)
    print("📁 로그 파일:")
    print("=" * 70)
    for i, (_, label) in enumerate(stages, 1):
        log_name = f"phase15_0{i}_*.log"
        logs = sorted(LOG_DIR.glob(log_name.replace("*", "*")))
        if logs:
            for log in logs:
                print(f"  tail -f {log.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 70)
    if is_running:
        print("💡 진행 중입니다. 'check_phase15_status.py'를 다시 실행하면 최신 상태를 봅니다.")
    else:
        print("✅ 파이프라인이 완료됐습니다!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
