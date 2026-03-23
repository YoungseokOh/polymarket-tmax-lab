"""Run the recent Seoul/NYC/London benchmark end-to-end."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from pmtmax.utils import dump_json, load_json

DEFAULT_MARKETS = Path("configs/market_inventory/recent_core_temperature_snapshots.json")
DEFAULT_CONFIG = Path("configs/recent-core-benchmark.yaml")
DEFAULT_OUTPUT_ROOT = Path("artifacts/recent_core_benchmark")
DEFAULT_CITIES = ["Seoul", "NYC", "London"]


def _city_slug(city: str) -> str:
    return city.lower().replace(" ", "_")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets-path", type=Path, default=DEFAULT_MARKETS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--city", action="append", default=None, help="Repeat to limit benchmark cities.")
    parser.add_argument("--model-name", default="gaussian_emos")
    parser.add_argument("--quote-proxy-half-spread", type=float, default=0.02)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    markets_path = args.markets_path.resolve()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    cities = args.city or DEFAULT_CITIES

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
        dataset_path = city_root / "parquet" / "gold" / f"{dataset_name}.parquet"
        dataset_rows = int(len(pd.read_parquet(dataset_path)))

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
        panel_path = city_root / "parquet" / "gold" / f"{panel_name}.parquet"

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
        real_metrics_path, real_trades_path = _copy_metrics(city_root, pricing_source="real_history", city_slug=city_slug)

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
        proxy_metrics_path, proxy_trades_path = _copy_metrics(city_root, pricing_source="quote_proxy", city_slug=city_slug)

        real_metrics = load_json(real_metrics_path)
        proxy_metrics = load_json(proxy_metrics_path)
        summary["cities"][city] = {
            "snapshot_count": snapshot_counts[city],
            **snapshot_ranges[city],
            "dataset_rows": dataset_rows,
            "dataset_path": str(dataset_path),
            "panel_path": str(panel_path),
            "panel_summary": _panel_summary(panel_path),
            "real_history_metrics": real_metrics,
            "quote_proxy_metrics": proxy_metrics,
            "real_history_trades_path": str(real_trades_path),
            "quote_proxy_trades_path": str(proxy_trades_path),
            "pnl_delta_quote_proxy": float(proxy_metrics["pnl"]) - float(real_metrics["pnl"]),
        }

    summary_path = output_root / "recent_core_benchmark_summary.json"
    dump_json(summary_path, summary)
    print(f"Wrote recent core benchmark summary -> {summary_path}")


if __name__ == "__main__":
    main()
