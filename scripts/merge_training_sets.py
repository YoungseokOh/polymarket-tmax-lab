"""Merge existing historical training set with the synthetic historical training set.

Creates:
  data/parquet/gold/v2/expanded_training_set.parquet
  data/parquet/gold/v2/expanded_backtest_panel.parquet

Does NOT overwrite canonical files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = REPO_ROOT / "data/parquet/gold/v2"


def merge_frames(existing_path: Path, synthetic_path: Path, output_path: Path) -> int:
    """Merge two parquet files, dedup on (market_id, target_date, decision_horizon), save output."""
    if not existing_path.exists():
        print(f"ERROR: existing file not found: {existing_path}", file=sys.stderr)
        return -1
    if not synthetic_path.exists():
        print(f"ERROR: synthetic file not found: {synthetic_path}", file=sys.stderr)
        return -1

    print(f"Loading existing: {existing_path}", flush=True)
    existing = pd.read_parquet(existing_path)
    print(f"  rows: {len(existing)}", flush=True)

    print(f"Loading synthetic: {synthetic_path}", flush=True)
    synthetic = pd.read_parquet(synthetic_path)
    print(f"  rows: {len(synthetic)}", flush=True)

    merged = pd.concat([existing, synthetic], ignore_index=True)
    print(f"  combined rows before dedup: {len(merged)}", flush=True)

    # Dedup on key columns if they exist
    key_cols = ["market_id", "target_date", "decision_horizon"]
    available = [c for c in key_cols if c in merged.columns]
    if available:
        before = len(merged)
        merged = merged.drop_duplicates(subset=available, keep="last")
        after = len(merged)
        if before != after:
            print(f"  dropped {before - after} duplicates on {available}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    print(f"Saved: {output_path} ({len(merged)} rows)", flush=True)
    return len(merged)


def main() -> None:
    # Merge training sets
    training_count = merge_frames(
        existing_path=PARQUET_DIR / "historical_training_set.parquet",
        synthetic_path=PARQUET_DIR / "synthetic_historical_training_set.parquet",
        output_path=PARQUET_DIR / "expanded_training_set.parquet",
    )

    # Merge backtest panels
    panel_count = merge_frames(
        existing_path=PARQUET_DIR / "historical_backtest_panel.parquet",
        synthetic_path=PARQUET_DIR / "synthetic_historical_backtest_panel.parquet",
        output_path=PARQUET_DIR / "expanded_backtest_panel.parquet",
    )

    print(f"\nExpanded training set: {training_count} rows")
    print(f"Expanded backtest panel: {panel_count} rows")


if __name__ == "__main__":
    main()
