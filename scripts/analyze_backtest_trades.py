"""Analyze PMTMAX trade-level backtest results and suggest execution filters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_TRADES = Path(
    "artifacts/workspaces/historical_real/backtests/v2/backtest_trades_real_history.json"
)
DEFAULT_METRICS = Path(
    "artifacts/workspaces/historical_real/backtests/v2/backtest_metrics_real_history.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/workspaces/historical_real/quality/backtest_trade_diagnostics.json"
)
DEFAULT_MD = Path("artifacts/workspaces/historical_real/quality/backtest_trade_diagnostics.md")


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"trades": 0, "pnl": 0.0, "hit_rate": None, "avg_price": None, "avg_edge": None}
    pnl = pd.to_numeric(frame["realized_pnl"], errors="coerce").fillna(0.0)
    return {
        "trades": int(len(frame)),
        "pnl": _round(pnl.sum()),
        "pnl_per_trade": _round(pnl.mean()),
        "hit_rate": _round((pnl > 0).mean()),
        "avg_price": _round(pd.to_numeric(frame["price"], errors="coerce").mean()),
        "avg_edge": _round(pd.to_numeric(frame["edge"], errors="coerce").mean()),
        "median_edge": _round(pd.to_numeric(frame["edge"], errors="coerce").median()),
        "avg_price_age_min": _round(
            pd.to_numeric(frame.get("price_age_seconds"), errors="coerce").mean() / 60.0
        ),
    }


def _group_summary(frame: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(by, dropna=False):
        rows.append({by: str(key), **_summary(group)})
    rows.sort(key=lambda row: (float(row.get("pnl") or 0), -int(row.get("trades") or 0)))
    return rows


def _bucket_summary(
    frame: pd.DataFrame, column: str, bins: list[float], labels: list[str]
) -> list[dict[str, Any]]:
    work = frame.copy()
    work[f"{column}_bucket"] = pd.cut(
        pd.to_numeric(work[column], errors="coerce"), bins=bins, labels=labels, include_lowest=True
    )
    return _group_summary(work, f"{column}_bucket")


def _filter_grid(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for min_edge in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]:
        for min_price in [0.0, 0.03, 0.05, 0.10, 0.15, 0.20]:
            for max_price in [0.60, 0.75, 0.90, 1.0]:
                sub = frame.loc[
                    (pd.to_numeric(frame["edge"], errors="coerce") >= min_edge)
                    & (pd.to_numeric(frame["price"], errors="coerce") >= min_price)
                    & (pd.to_numeric(frame["price"], errors="coerce") <= max_price)
                ]
                if len(sub) < 5:
                    continue
                s = _summary(sub)
                rows.append(
                    {"min_edge": min_edge, "min_price": min_price, "max_price": max_price, **s}
                )
    rows.sort(
        key=lambda row: (float(row.get("pnl") or 0), float(row.get("pnl_per_trade") or 0)),
        reverse=True,
    )
    return rows[:30]


def _format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    trades = pd.DataFrame(json.loads(args.trades.read_text()))
    metrics = json.loads(args.metrics.read_text()) if args.metrics.exists() else {}
    if not trades.empty:
        trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
        trades["edge"] = pd.to_numeric(trades["edge"], errors="coerce")
        trades["realized_pnl"] = pd.to_numeric(trades["realized_pnl"], errors="coerce").fillna(0.0)
        trades["price_age_seconds"] = pd.to_numeric(
            trades.get("price_age_seconds"), errors="coerce"
        )

    report = {
        "metrics": metrics,
        "overall": _summary(trades),
        "by_horizon": _group_summary(trades, "decision_horizon") if not trades.empty else [],
        "by_city": _group_summary(trades, "city") if not trades.empty else [],
        "by_price_bucket": _bucket_summary(
            trades,
            "price",
            [0, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.01],
            ["0-3c", "3-5c", "5-10c", "10-20c", "20-40c", "40-60c", "60-80c", "80-100c"],
        )
        if not trades.empty
        else [],
        "by_edge_bucket": _bucket_summary(
            trades,
            "edge",
            [0, 0.05, 0.1, 0.2, 0.4, 0.75, 1.5, 10],
            ["0-5c", "5-10c", "10-20c", "20-40c", "40-75c", "75-150c", "150c+"],
        )
        if not trades.empty
        else [],
        "top_filter_candidates": _filter_grid(trades) if not trades.empty else [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")

    md = f"""# Backtest Trade Diagnostics

## Overall

- trades: **{report["overall"]["trades"]}**
- pnl: **{report["overall"]["pnl"]}**
- hit rate: **{report["overall"]["hit_rate"]}**
- avg price: **{report["overall"]["avg_price"]}**
- avg edge: **{report["overall"]["avg_edge"]}**

## By Horizon

{_format_table(report["by_horizon"], ["decision_horizon", "trades", "pnl", "pnl_per_trade", "hit_rate", "avg_price", "avg_edge"])}

## Worst/Best Cities

{_format_table(report["by_city"][:10] + list(reversed(report["by_city"][-10:])), ["city", "trades", "pnl", "pnl_per_trade", "hit_rate", "avg_price", "avg_edge"])}

## By Price Bucket

{_format_table(report["by_price_bucket"], ["price_bucket", "trades", "pnl", "pnl_per_trade", "hit_rate", "avg_price", "avg_edge"])}

## By Edge Bucket

{_format_table(report["by_edge_bucket"], ["edge_bucket", "trades", "pnl", "pnl_per_trade", "hit_rate", "avg_price", "avg_edge"])}

## Top Filter Candidates

{_format_table(report["top_filter_candidates"][:15], ["min_edge", "min_price", "max_price", "trades", "pnl", "pnl_per_trade", "hit_rate", "avg_price", "avg_edge"])}
"""
    args.markdown.write_text(md)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "markdown": str(args.markdown),
                "overall": report["overall"],
                "top_filter": report["top_filter_candidates"][:3],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
