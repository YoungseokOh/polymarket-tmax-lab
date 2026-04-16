"""Recent-core backtest diagnostics and coverage summaries."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from pmtmax.utils import load_yaml_with_extends

RECENT_CORE_CITIES = ("Seoul", "NYC", "London")


def load_recent_core_horizon_policy(path: Path) -> dict[str, set[str]]:
    """Load city-specific allowed horizons from the checked-in policy file."""

    payload = load_yaml_with_extends(path)
    cities = payload.get("cities", {}) if isinstance(payload, Mapping) else {}
    policy: dict[str, set[str]] = {}
    for city, city_payload in cities.items():
        if not isinstance(city_payload, Mapping):
            continue
        policy[str(city)] = {str(horizon) for horizon in city_payload.get("allowed_horizons", [])}
    return policy


def _trade_summary(frame: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    grouped = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            trades=("realized_pnl", "size"),
            pnl=("realized_pnl", "sum"),
            hit_rate=("realized_pnl", lambda series: float((pd.to_numeric(series, errors="coerce").fillna(0.0) > 0).mean())),
        )
        .reset_index()
    )
    rows = grouped.to_dict(orient="records")
    for row in rows:
        row["trades"] = int(row["trades"])
        row["pnl"] = float(row["pnl"])
        row["hit_rate"] = float(row["hit_rate"])
    return rows


def summarize_recent_core_diagnostics(
    *,
    panel: pd.DataFrame,
    trade_logs_by_source: Mapping[str, pd.DataFrame],
    horizon_policy: Mapping[str, set[str]],
    cities: tuple[str, ...] = RECENT_CORE_CITIES,
) -> dict[str, Any]:
    """Summarize recent-core coverage and policy-slice PnL from existing artifacts."""

    core_panel = panel.loc[panel["city"].astype(str).isin(cities)].copy()
    core_panel["coverage_status"] = core_panel["coverage_status"].astype(str)
    coverage = (
        core_panel.groupby(["city", "decision_horizon", "coverage_status"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    coverage_rows = coverage.to_dict(orient="records")
    city_horizon_rows: list[dict[str, Any]] = []
    for (city, horizon), group in coverage.groupby(["city", "decision_horizon"], dropna=False):
        counts = {str(row["coverage_status"]): int(row["count"]) for row in group.to_dict(orient="records")}
        total_rows = int(sum(counts.values()))
        ok_rows = int(counts.get("ok", 0))
        city_horizon_rows.append(
            {
                "city": str(city),
                "decision_horizon": str(horizon),
                "rows": total_rows,
                "ok": ok_rows,
                "stale": int(counts.get("stale", 0)),
                "missing": int(counts.get("missing", 0)),
                "ok_ratio": (float(ok_rows) / float(total_rows)) if total_rows > 0 else 0.0,
            }
        )
    city_horizon_rows.sort(key=lambda row: (row["ok_ratio"], row["city"], row["decision_horizon"]))

    diagnostics: dict[str, Any] = {
        "recent_core_cities": list(cities),
        "coverage_rows": coverage_rows,
        "coverage_by_city_horizon": city_horizon_rows,
        "coverage_bottlenecks": city_horizon_rows[:6],
        "sources": {},
    }

    for source_name, trades in trade_logs_by_source.items():
        frame = trades.loc[trades["city"].astype(str).isin(cities)].copy()
        if frame.empty:
            diagnostics["sources"][source_name] = {
                "policy_trades_by_city": [],
                "policy_trades_by_city_horizon": [],
                "policy_aggregate": {"trades": 0, "pnl": 0.0, "hit_rate": 0.0},
                "negative_policy_cities": [],
                "negative_policy_city_horizons": [],
            }
            continue

        frame["city"] = frame["city"].astype(str)
        frame["decision_horizon"] = frame["decision_horizon"].astype(str)
        frame["policy_allowed"] = frame.apply(
            lambda row: row["decision_horizon"] in horizon_policy.get(row["city"], set()),
            axis=1,
        )
        policy_trades = frame.loc[frame["policy_allowed"]].copy()
        by_city = _trade_summary(policy_trades, ["city"])
        by_city_horizon = _trade_summary(policy_trades, ["city", "decision_horizon"])
        policy_aggregate = {
            "trades": int(len(policy_trades)),
            "pnl": float(pd.to_numeric(policy_trades["realized_pnl"], errors="coerce").fillna(0.0).sum()),
            "hit_rate": float((pd.to_numeric(policy_trades["realized_pnl"], errors="coerce").fillna(0.0) > 0).mean())
            if not policy_trades.empty
            else 0.0,
        }
        diagnostics["sources"][source_name] = {
            "policy_trades_by_city": by_city,
            "policy_trades_by_city_horizon": by_city_horizon,
            "policy_aggregate": policy_aggregate,
            "negative_policy_cities": [row for row in by_city if row["pnl"] < 0],
            "negative_policy_city_horizons": [row for row in by_city_horizon if row["pnl"] < 0],
        }

    recommendations: list[dict[str, str]] = []
    for row in diagnostics["coverage_bottlenecks"]:
        if row["ok_ratio"] < 0.10:
            recommendations.append(
                {
                    "type": "coverage",
                    "city": row["city"],
                    "decision_horizon": row["decision_horizon"],
                    "message": "price_history coverage is too thin for reliable recent-core evaluation",
                }
            )
    for source_name, source_payload in diagnostics["sources"].items():
        for row in source_payload["negative_policy_cities"]:
            recommendations.append(
                {
                    "type": "policy_pnl",
                    "source": source_name,
                    "city": row["city"],
                    "message": "policy-allowed recent-core slice is losing money",
                }
            )
    diagnostics["recommendations"] = recommendations
    return diagnostics
