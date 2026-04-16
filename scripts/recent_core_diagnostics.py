"""Summarize recent-core coverage and policy-slice PnL from existing backtest artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pmtmax.backtest.recent_core_diagnostics import (
    load_recent_core_horizon_policy,
    summarize_recent_core_diagnostics,
)
from pmtmax.utils import dump_json, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=Path("data/parquet/gold/historical_backtest_panel.parquet"),
    )
    parser.add_argument(
        "--quote-proxy-trades-path",
        type=Path,
        default=Path("artifacts/workspaces/historical_real/backtests/v2/backtest_trades_quote_proxy.json"),
    )
    parser.add_argument(
        "--real-history-trades-path",
        type=Path,
        default=Path("artifacts/workspaces/historical_real/backtests/v2/backtest_trades_real_history.json"),
    )
    parser.add_argument(
        "--horizon-policy-path",
        type=Path,
        default=Path("configs/recent-core-horizon-policy.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/workspaces/historical_real/backtests/v2/recent_core_diagnostic.json"),
    )
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel_path)
    diagnostics = summarize_recent_core_diagnostics(
        panel=panel,
        trade_logs_by_source={
            "quote_proxy": pd.DataFrame(load_json(args.quote_proxy_trades_path)),
            "real_history": pd.DataFrame(load_json(args.real_history_trades_path)),
        },
        horizon_policy=load_recent_core_horizon_policy(args.horizon_policy_path),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump_json(args.output, diagnostics)
    print(f"Wrote recent-core diagnostics -> {args.output}")


if __name__ == "__main__":
    main()
