"""Run the recent Seoul/NYC/London benchmark end-to-end."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from pmtmax.backtest.metrics import summarize_trade_log
from pmtmax.backtest.recent_core_benchmark import classify_profitability, summarize_recent_core_profitability
from pmtmax.utils import dump_json, load_json, load_yaml_with_extends

DEFAULT_MARKETS = Path("configs/market_inventory/recent_core_temperature_snapshots.json")
DEFAULT_CONFIG = Path("configs/recent-core-benchmark.yaml")
DEFAULT_HORIZON_POLICY = Path("configs/recent-core-horizon-policy.yaml")
DEFAULT_OUTPUT_ROOT = Path("artifacts/recent_core_benchmark")
DEFAULT_CITIES = ["Seoul", "NYC", "London"]


def _city_slug(city: str) -> str:
    return city.lower().replace(" ", "_")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)  # noqa: S603


def _city_env(city_root: Path, config_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PMTMAX_CONFIG"] = str(config_path.resolve())
    env["PMTMAX_DATA_DIR"] = str((city_root / "data").resolve())
    env["PMTMAX_CACHE_DIR"] = str((city_root / "cache").resolve())
    env["PMTMAX_RAW_DIR"] = str((city_root / "raw").resolve())
    env["PMTMAX_RUN_DIR"] = str((city_root / "runs").resolve())
    env["PMTMAX_ARCHIVE_DIR"] = str((city_root / "archive").resolve())
    env["PMTMAX_MANIFEST_DIR"] = str((city_root / "manifests").resolve())
    env["PMTMAX_DUCKDB_PATH"] = str((city_root / "duckdb" / "warehouse.duckdb").resolve())
    env["PMTMAX_PARQUET_DIR"] = str((city_root / "parquet").resolve())
    return env


def _panel_summary(panel_path: Path) -> dict[str, object]:
    panel = pd.read_parquet(panel_path)
    return {
        "rows": int(len(panel)),
        "coverage": {str(key): int(value) for key, value in panel["coverage_status"].value_counts().to_dict().items()},
        "coverage_by_horizon": {
            f"{horizon}:{status}": int(count)
            for (horizon, status), count in panel.groupby(["decision_horizon", "coverage_status"]).size().to_dict().items()
        },
    }


def _copy_metrics(city_root: Path, *, pricing_source: str, city_slug: str) -> tuple[Path, Path]:
    artifacts_dir = city_root / "artifacts"
    if pricing_source == "real_history":
        metrics_src = artifacts_dir / "backtest_metrics_real_history.json"
        trades_src = artifacts_dir / "backtest_trades_real_history.json"
        metrics_dst = artifacts_dir / f"{city_slug}_backtest_metrics_real_history.json"
        trades_dst = artifacts_dir / f"{city_slug}_backtest_trades_real_history.json"
    else:
        metrics_src = artifacts_dir / "backtest_metrics_quote_proxy.json"
        trades_src = artifacts_dir / "backtest_trades_quote_proxy.json"
        metrics_dst = artifacts_dir / f"{city_slug}_backtest_metrics_quote_proxy.json"
        trades_dst = artifacts_dir / f"{city_slug}_backtest_trades_quote_proxy.json"
    shutil.copyfile(metrics_src, metrics_dst)
    shutil.copyfile(trades_src, trades_dst)
    return metrics_dst, trades_dst


def _trade_summary(trades_path: Path) -> dict[str, dict[str, float]]:
    trades = pd.DataFrame(load_json(trades_path))
    if trades.empty:
        return {}
    summary: dict[str, dict[str, float]] = {}
    for horizon, group in trades.groupby("decision_horizon", dropna=False):
        realized = pd.to_numeric(group["realized_pnl"], errors="coerce").fillna(0.0)
        summary[str(horizon)] = {
            "trade_count": float(len(group)),
            "pnl": float(realized.sum()),
            "hit_rate": float((realized > 0).mean()),
            "avg_price": float(pd.to_numeric(group["price"], errors="coerce").mean()),
        }
    return summary


def _panel_summary_for_horizon(panel_summary: dict[str, object], horizon: str) -> dict[str, object]:
    """Extract one horizon's coverage summary from the panel-level summary."""

    coverage_counts: dict[str, int] = {}
    for key, count in dict(panel_summary.get("coverage_by_horizon", {})).items():
        key_str = str(key)
        if not key_str.startswith(f"{horizon}:"):
            continue
        _, status = key_str.split(":", 1)
        coverage_counts[status] = int(count)
    return {
        "rows": int(sum(coverage_counts.values())),
        "coverage": dict(sorted(coverage_counts.items())),
    }


def _trade_metrics_from_horizon_row(row: dict[str, object]) -> dict[str, float]:
    """Normalize one horizon trade summary into benchmark-style trade metrics."""

    return {
        "num_trades": float(row.get("trade_count", 0.0)),
        "pnl": float(row.get("pnl", 0.0)),
        "hit_rate": float(row.get("hit_rate", 0.0)),
        "avg_price": float(row.get("avg_price", 0.0)),
    }


def _load_horizon_policy(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    payload = load_yaml_with_extends(path.resolve())
    cities = payload.get("cities", {})
    result: dict[str, list[str]] = {}
    for city, city_payload in cities.items():
        if not isinstance(city_payload, dict):
            continue
        horizons = city_payload.get("allowed_horizons", [])
        result[str(city)] = [str(horizon) for horizon in horizons]
    return result


def _policy_trade_metrics(trades_path: Path, allowed_horizons: list[str]) -> dict[str, float]:
    trades = pd.DataFrame(load_json(trades_path))
    if trades.empty:
        return {"num_trades": 0.0, "pnl": 0.0, "hit_rate": 0.0, "avg_edge": 0.0}
    if allowed_horizons:
        trades = trades.loc[trades["decision_horizon"].astype(str).isin(allowed_horizons)].copy()
    return summarize_trade_log(trades)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets-path", type=Path, default=DEFAULT_MARKETS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--horizon-policy", type=Path, default=DEFAULT_HORIZON_POLICY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--city", action="append", default=None, help="Repeat to limit benchmark cities.")
    parser.add_argument("--model-name", default="gaussian_emos")
    parser.add_argument("--quote-proxy-half-spread", type=float, default=0.02)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing per-city dataset/panel/backtest artifacts in the output root when present.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    markets_path = args.markets_path.resolve()
    config_path = args.config.resolve()
    horizon_policy_path = args.horizon_policy.resolve() if args.horizon_policy is not None else None
    output_root = args.output_root.resolve()
    cities = args.city or DEFAULT_CITIES
    horizon_policy = _load_horizon_policy(horizon_policy_path)

    snapshots = load_json(markets_path)
    snapshot_counts: dict[str, int] = {}
    snapshot_ranges: dict[str, dict[str, str]] = {}
    for city in cities:
        city_rows = [row for row in snapshots if row.get("spec", {}).get("city") == city]
        dates = sorted(str(row.get("spec", {}).get("target_local_date")) for row in city_rows)
        snapshot_counts[city] = len(city_rows)
        snapshot_ranges[city] = {
            "start_date": dates[0] if dates else "",
            "end_date": dates[-1] if dates else "",
        }

    summary: dict[str, object] = {
        "config_path": str(config_path),
        "markets_path": str(markets_path),
        "horizon_policy_path": str(horizon_policy_path) if horizon_policy_path is not None else None,
        "quote_proxy_half_spread": args.quote_proxy_half_spread,
        "cities": {},
    }

    for city in cities:
        city_slug = _city_slug(city)
        city_root = output_root / city_slug
        city_root.mkdir(parents=True, exist_ok=True)
        env = _city_env(city_root, config_path)
        dataset_name = f"{city_slug}_recent_training_set"
        panel_name = f"{city_slug}_recent_backtest_panel"
        dataset_path = city_root / "parquet" / "gold" / f"{dataset_name}.parquet"
        if not (args.reuse_existing and dataset_path.exists()):
            _run(
                [
                    "uv",
                    "run",
                    "--project",
                    str(repo_root),
                    "pmtmax",
                    "build-dataset",
                    "--markets-path",
                    str(markets_path),
                    "--city",
                    city,
                    "--output-name",
                    dataset_name,
                ],
                cwd=city_root,
                env=env,
            )
        dataset_rows = int(len(pd.read_parquet(dataset_path)))

        panel_path = city_root / "parquet" / "gold" / f"{panel_name}.parquet"
        if not (args.reuse_existing and panel_path.exists()):
            _run(
                [
                    "uv",
                    "run",
                    "--project",
                    str(repo_root),
                    "pmtmax",
                    "backfill-price-history",
                    "--markets-path",
                    str(markets_path),
                    "--city",
                    city,
                ],
                cwd=city_root,
                env=env,
            )
            _run(
                [
                    "uv",
                    "run",
                    "--project",
                    str(repo_root),
                    "pmtmax",
                    "materialize-backtest-panel",
                    "--dataset-path",
                    str(dataset_path),
                    "--markets-path",
                    str(markets_path),
                    "--city",
                    city,
                    "--output-name",
                    panel_name,
                ],
                cwd=city_root,
                env=env,
            )

        real_metrics_path = city_root / "artifacts" / f"{city_slug}_backtest_metrics_real_history.json"
        real_trades_path = city_root / "artifacts" / f"{city_slug}_backtest_trades_real_history.json"
        if not (args.reuse_existing and real_metrics_path.exists() and real_trades_path.exists()):
            _run(
                [
                    "uv",
                    "run",
                    "--project",
                    str(repo_root),
                    "pmtmax",
                    "backtest",
                    "--dataset-path",
                    str(dataset_path),
                    "--panel-path",
                    str(panel_path),
                    "--pricing-source",
                    "real_history",
                    "--model-name",
                    args.model_name,
                ],
                cwd=city_root,
                env=env,
            )
            real_metrics_path, real_trades_path = _copy_metrics(
                city_root,
                pricing_source="real_history",
                city_slug=city_slug,
            )

        proxy_metrics_path = city_root / "artifacts" / f"{city_slug}_backtest_metrics_quote_proxy.json"
        proxy_trades_path = city_root / "artifacts" / f"{city_slug}_backtest_trades_quote_proxy.json"
        if not (args.reuse_existing and proxy_metrics_path.exists() and proxy_trades_path.exists()):
            _run(
                [
                    "uv",
                    "run",
                    "--project",
                    str(repo_root),
                    "pmtmax",
                    "backtest",
                    "--dataset-path",
                    str(dataset_path),
                    "--panel-path",
                    str(panel_path),
                    "--pricing-source",
                    "quote_proxy",
                    "--quote-proxy-half-spread",
                    str(args.quote_proxy_half_spread),
                    "--model-name",
                    args.model_name,
                ],
                cwd=city_root,
                env=env,
            )
            proxy_metrics_path, proxy_trades_path = _copy_metrics(
                city_root,
                pricing_source="quote_proxy",
                city_slug=city_slug,
            )

        real_metrics = load_json(real_metrics_path)
        proxy_metrics = load_json(proxy_metrics_path)
        real_horizon = _trade_summary(real_trades_path)
        proxy_horizon = _trade_summary(proxy_trades_path)
        allowed_horizons = horizon_policy.get(city, [])
        panel_summary = _panel_summary(panel_path)
        horizon_delta: dict[str, dict[str, float]] = {}
        horizons: dict[str, dict[str, object]] = {}
        for horizon in sorted(set(real_horizon) | set(proxy_horizon)):
            real_row = real_horizon.get(horizon, {})
            proxy_row = proxy_horizon.get(horizon, {})
            real_trade_metrics = _trade_metrics_from_horizon_row(real_row)
            proxy_trade_metrics = _trade_metrics_from_horizon_row(proxy_row)
            horizon_panel_summary = _panel_summary_for_horizon(panel_summary, horizon)
            policy_allowed = not allowed_horizons or horizon in allowed_horizons
            horizon_delta[horizon] = {
                "real_trade_count": float(real_row.get("trade_count", 0.0)),
                "proxy_trade_count": float(proxy_row.get("trade_count", 0.0)),
                "real_pnl": float(real_row.get("pnl", 0.0)),
                "proxy_pnl": float(proxy_row.get("pnl", 0.0)),
                "pnl_delta": float(proxy_row.get("pnl", 0.0)) - float(real_row.get("pnl", 0.0)),
                "real_hit_rate": float(real_row.get("hit_rate", 0.0)),
                "proxy_hit_rate": float(proxy_row.get("hit_rate", 0.0)),
            }
            horizon_profitability = classify_profitability(
                aggregate_real_history_metrics={
                    "priced_decision_rows": float(horizon_panel_summary["rows"]),
                },
                aggregate_policy_real_history_metrics=real_trade_metrics if policy_allowed else {"num_trades": 0.0, "pnl": 0.0},
                aggregate_policy_quote_proxy_metrics=(
                    proxy_trade_metrics if policy_allowed else {"num_trades": 0.0, "pnl": 0.0}
                ),
            )
            horizons[horizon] = {
                "policy_allowed": policy_allowed,
                "panel_summary": horizon_panel_summary,
                "real_history_metrics": real_trade_metrics,
                "quote_proxy_metrics": proxy_trade_metrics,
                "decision": horizon_profitability["decision"],
                "decision_reason": horizon_profitability["decision_reason"],
                "sample_adequacy": horizon_profitability["sample_adequacy"],
            }
        policy_real_metrics = _policy_trade_metrics(real_trades_path, allowed_horizons)
        policy_proxy_metrics = _policy_trade_metrics(proxy_trades_path, allowed_horizons)
        summary["cities"][city] = {
            "snapshot_count": snapshot_counts[city],
            **snapshot_ranges[city],
            "dataset_rows": dataset_rows,
            "dataset_path": str(dataset_path),
            "panel_path": str(panel_path),
            "panel_summary": panel_summary,
            "allowed_horizons": allowed_horizons,
            "real_history_metrics": real_metrics,
            "quote_proxy_metrics": proxy_metrics,
            "policy_real_history_metrics": policy_real_metrics,
            "policy_quote_proxy_metrics": policy_proxy_metrics,
            "horizons": horizons,
            "real_history_by_horizon": real_horizon,
            "quote_proxy_by_horizon": proxy_horizon,
            "horizon_delta_quote_proxy": horizon_delta,
            "real_history_trades_path": str(real_trades_path),
            "quote_proxy_trades_path": str(proxy_trades_path),
            "pnl_delta_quote_proxy": float(proxy_metrics["pnl"]) - float(real_metrics["pnl"]),
        }

    summary.update(summarize_recent_core_profitability(dict(summary["cities"])))
    summary_path = output_root / "recent_core_benchmark_summary.json"
    dump_json(summary_path, summary)
    print(f"Wrote recent core benchmark summary -> {summary_path}")


if __name__ == "__main__":
    main()
