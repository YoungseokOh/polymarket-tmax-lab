"""Compare leaderboards from multiple benchmark runs to pick champion.

Usage:
    uv run python scripts/compare_leaderboards.py \
      --main artifacts/workspaces/historical_real/benchmarks/leaderboard.json \
      --extra artifacts/workspaces/historical_real/benchmarks/lgbm_emos_leaderboard.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pmtmax.modeling.champion import score_execution_candidate_leaderboard, score_leaderboard


def load_leaderboard(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def merge_leaderboards(paths: list[Path]) -> pd.DataFrame:
    records = []
    seen_models: set[str] = set()
    for path in paths:
        data = load_leaderboard(path)
        for row in data:
            name = row["model_name"]
            if name not in seen_models:
                records.append(row)
                seen_models.add(name)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark leaderboards")
    parser.add_argument("--main", type=Path, required=True, help="Main leaderboard JSON path")
    parser.add_argument("--extra", type=Path, nargs="*", default=[], help="Additional leaderboard JSONs")
    args = parser.parse_args()

    paths = [args.main] + (args.extra or [])
    paths = [p for p in paths if p.exists()]

    if not paths:
        print("No leaderboard files found")
        return

    df = merge_leaderboards(paths)

    display_cols = [
        "model_name",
        "avg_crps_mean",
        "mae_mean",
        "calibration_gap_mean",
        "real_history_pnl_mean",
        "real_history_hit_rate_mean",
        "avg_brier_mean",
        "nll_mean",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    print("\n=== ALL MODELS ===")
    print(df[display_cols].to_string(index=False, float_format="{:.4f}".format))

    print("\n=== RESEARCH CHAMPION RANKING ===")
    scored = score_leaderboard(df)
    for _, row in scored.iterrows():
        print(
            f"  {row['model_name']:25s} champion_score={row['champion_score']:.3f}"
            f"  CRPS={row['avg_crps_mean']:.4f} MAE={row['mae_mean']:.4f}"
            f"  PnL=${row['real_history_pnl_mean']:.1f}"
        )

    champion = scored.iloc[0]["model_name"]
    print(f"\n  --> RESEARCH CHAMPION: {champion}")

    if "real_history_pnl_mean" in df.columns:
        print("\n=== EXECUTION CANDIDATE RANKING ===")
        try:
            tscored = score_execution_candidate_leaderboard(df)
            for _, row in tscored.iterrows():
                print(
                    f"  {row['model_name']:25s} execution_score={row['execution_candidate_score']:.3f}"
                    f"  PnL=${row['real_history_pnl_mean']:.1f}"
                    f"  hit={row['real_history_hit_rate_mean']:.3f}"
                )
            execution_candidate = tscored.iloc[0]["model_name"]
            print(f"\n  --> EXECUTION CANDIDATE: {execution_candidate}")
        except Exception as e:
            print(f"  (execution candidate ranking failed: {e})")


if __name__ == "__main__":
    main()
