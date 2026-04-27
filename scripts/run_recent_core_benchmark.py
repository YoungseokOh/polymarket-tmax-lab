"""Run the recent Seoul/NYC/London benchmark end-to-end."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from pmtmax.backtest.metrics import summarize_trade_log
from pmtmax.backtest.recent_core_benchmark import (
    classify_profitability,
    summarize_recent_core_profitability,
)
from pmtmax.markets.repository import load_market_snapshot_payloads
from pmtmax.utils import dump_json, load_json, load_yaml_with_extends

DEFAULT_MARKETS = Path("configs/market_inventory/recent_core_temperature_snapshots.json")
DEFAULT_CONFIG = Path("configs/recent-core-benchmark.yaml")
DEFAULT_HORIZON_POLICY = Path("configs/recent-core-horizon-policy.yaml")
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("PMTMAX_ARTIFACTS_DIR", "artifacts")) / "recent_core_benchmark"
DEFAULT_CITIES = ["Seoul", "NYC", "London"]


def _city_slug(city: str) -> str:
    return city.lower().replace(" ", "_")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)  # noqa: S603


def _city_env(city_root: Path, config_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PMTMAX_CONFIG"] = str(config_path.resolve())
    env["PMTMAX_WORKSPACE_NAME"] = env.get("PMTMAX_WORKSPACE_NAME", "recent_core_eval")
    env["PMTMAX_DATASET_PROFILE"] = env.get("PMTMAX_DATASET_PROFILE", "real_market")
    env["PMTMAX_DATA_DIR"] = str((city_root / "data").resolve())
    env["PMTMAX_CACHE_DIR"] = str((city_root / "cache").resolve())
    env["PMTMAX_RAW_DIR"] = str((city_root / "raw").resolve())
    env["PMTMAX_RUN_DIR"] = str((city_root / "runs").resolve())
    env["PMTMAX_ARCHIVE_DIR"] = str((city_root / "archive").resolve())
    env["PMTMAX_MANIFEST_DIR"] = str((city_root / "manifests").resolve())
    env["PMTMAX_DUCKDB_PATH"] = str((city_root / "duckdb" / "warehouse.duckdb").resolve())
    env["PMTMAX_PARQUET_DIR"] = str((city_root / "parquet").resolve())
    env["PMTMAX_ARTIFACTS_DIR"] = str((city_root / "artifacts").resolve())
    env["PMTMAX_PUBLIC_MODEL_DIR"] = str(
        Path(env.get("PMTMAX_PUBLIC_MODEL_DIR", "artifacts/public_models")).resolve()
    )
    return env


def _test_group_ids(dataset_path: Path, *, backtest_last_n: int) -> set[str] | None:
    """Return market-day group ids used as test groups when --last-n is active."""

    if backtest_last_n <= 0:
        return None
    frame = pd.read_parquet(dataset_path)
    if not {"market_id", "target_date"}.issubset(frame.columns):
        return None
    ordered = frame.sort_values(
        [column for column in ["target_date", "decision_time_utc", "market_id", "decision_horizon"] if column in frame.columns]
    ).reset_index(drop=True)
    key_frame = ordered[["market_id", "target_date"]].astype({"market_id": str}).copy()
    group_ids = key_frame.astype(str).agg("|".join, axis=1)
    unique_groups = group_ids.drop_duplicates().tolist()
    if not unique_groups:
        return set()
    return set(unique_groups[-backtest_last_n:])


def _panel_summary(
    panel_path: Path,
    *,
    dataset_path: Path | None = None,
    backtest_last_n: int = 0,
) -> dict[str, object]:
    panel = pd.read_parquet(panel_path)
    if dataset_path is not None and backtest_last_n > 0 and {"market_id", "target_date"}.issubset(panel.columns):
        test_group_ids = _test_group_ids(dataset_path, backtest_last_n=backtest_last_n)
        if test_group_ids is not None:
            key_frame = panel[["market_id", "target_date"]].astype({"market_id": str}).copy()
            group_ids = key_frame.astype(str).agg("|".join, axis=1)
            panel = panel.loc[group_ids.isin(test_group_ids)].copy()
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
    backtests_dir = artifacts_dir / "backtests" / "v2"
    if pricing_source == "real_history":
        metrics_src = backtests_dir / "backtest_metrics_real_history.json"
        trades_src = backtests_dir / "backtest_trades_real_history.json"
        metrics_dst = artifacts_dir / f"{city_slug}_backtest_metrics_real_history.json"
        trades_dst = artifacts_dir / f"{city_slug}_backtest_trades_real_history.json"
    else:
        metrics_src = backtests_dir / "backtest_metrics_quote_proxy.json"
        trades_src = backtests_dir / "backtest_trades_quote_proxy.json"
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


def _filter_prebuilt_city_frame(
    frame: pd.DataFrame,
    *,
    city: str,
    label: str,
    start_date: str | None = None,
    end_date: str | None = None,
    last_n_market_days: int = 0,
) -> pd.DataFrame:
    """Return one city slice from a trusted prebuilt recent-core frame."""

    if "city" not in frame.columns:
        msg = f"Prebuilt {label} frame is missing required city column."
        raise ValueError(msg)
    city_frame = frame.loc[frame["city"].astype(str) == city].copy()
    if city_frame.empty:
        msg = f"Prebuilt {label} frame has no rows for city={city}."
        raise ValueError(msg)
    if "target_date" in city_frame.columns:
        target_dates = pd.to_datetime(city_frame["target_date"], errors="coerce")
        if start_date:
            city_frame = city_frame.loc[target_dates >= pd.Timestamp(start_date)].copy()
            target_dates = pd.to_datetime(city_frame["target_date"], errors="coerce")
        if end_date:
            city_frame = city_frame.loc[target_dates <= pd.Timestamp(end_date)].copy()
            target_dates = pd.to_datetime(city_frame["target_date"], errors="coerce")
        if last_n_market_days > 0:
            unique_dates = sorted(target_dates.dropna().dt.normalize().unique())
            keep_dates = set(unique_dates[-last_n_market_days:])
            city_frame = city_frame.loc[target_dates.dt.normalize().isin(keep_dates)].copy()
    if city_frame.empty:
        msg = f"Prebuilt {label} frame has no rows for city={city} after date filters."
        raise ValueError(msg)
    return city_frame


def _write_prebuilt_city_frame(
    frame: pd.DataFrame,
    *,
    city: str,
    output_path: Path,
    label: str,
    start_date: str | None = None,
    end_date: str | None = None,
    last_n_market_days: int = 0,
) -> int:
    """Write one filtered city slice from a trusted prebuilt frame."""

    city_frame = _filter_prebuilt_city_frame(
        frame,
        city=city,
        label=label,
        start_date=start_date,
        end_date=end_date,
        last_n_market_days=last_n_market_days,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    city_frame.to_parquet(output_path, index=False)
    return int(len(city_frame))


def _city_dataset_range(frame: pd.DataFrame) -> tuple[int, str, str]:
    """Return market-day count and date bounds for one city dataset."""

    if "target_date" not in frame.columns or frame.empty:
        return int(len(frame)), "", ""
    target_dates = pd.to_datetime(frame["target_date"], errors="coerce").dropna().dt.normalize()
    if target_dates.empty:
        return int(len(frame)), "", ""
    unique_dates = sorted(target_dates.unique())
    return len(unique_dates), str(pd.Timestamp(unique_dates[0]).date()), str(pd.Timestamp(unique_dates[-1]).date())


def _backtest_split_count(frame: pd.DataFrame, *, backtest_last_n: int) -> int:
    """Estimate how many market-day test splits the downstream backtest will run."""

    if not {"market_id", "target_date"}.issubset(frame.columns):
        return 0
    key_frame = frame[["market_id", "target_date"]].astype({"market_id": str})
    num_groups = int(key_frame.astype(str).agg("|".join, axis=1).nunique())
    if num_groups <= 1:
        return 0
    if backtest_last_n > 0:
        return min(int(backtest_last_n), num_groups - 1)
    return num_groups - 1


def _validate_retrain_stride(*, retrain_stride: int, split_count: int, city: str) -> None:
    """Reject recent-core runs that would train once and reuse the model for every split."""

    if split_count > 1 and retrain_stride >= split_count:
        msg = (
            f"Invalid recent-core retrain stride for {city}: retrain_stride={retrain_stride} "
            f"but only {split_count} test splits. Use a smaller --retrain-stride or set "
            "--backtest-last-n with a wider prebuilt window."
        )
        raise ValueError(msg)


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
    parser.add_argument(
        "--prebuilt-dataset-path",
        type=Path,
        default=None,
        help="Trusted combined recent-core dataset parquet to slice by city instead of rebuilding.",
    )
    parser.add_argument(
        "--prebuilt-panel-path",
        type=Path,
        default=None,
        help="Trusted combined recent-core panel parquet to slice by city instead of rebuilding price history.",
    )
    parser.add_argument("--prebuilt-start-date", default=None, help="Optional inclusive start date for prebuilt parquet slices.")
    parser.add_argument("--prebuilt-end-date", default=None, help="Optional inclusive end date for prebuilt parquet slices.")
    parser.add_argument(
        "--prebuilt-last-n-market-days",
        type=int,
        default=0,
        help="Keep only the last N market days per city from prebuilt parquet.",
    )
    parser.add_argument("--city", action="append", default=None, help="Repeat to limit benchmark cities.")
    parser.add_argument("--model-name", default="gaussian_emos")
    parser.add_argument("--variant", default=None, help="Optional model variant to pass to pmtmax backtest.")
    parser.add_argument("--variant-spec", type=Path, default=None, help="Optional YAML-backed model variant spec.")
    parser.add_argument("--quote-proxy-half-spread", type=float, default=0.02)
    parser.add_argument(
        "--retrain-stride",
        type=int,
        default=1,
        help="Pass-through pmtmax backtest retrain stride.",
    )
    parser.add_argument(
        "--backtest-last-n",
        type=int,
        default=0,
        help="Pass-through pmtmax backtest --last-n value measured in market-day groups.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing per-city dataset/panel/backtest artifacts in the output root when present.",
    )
    args = parser.parse_args()
    if (args.prebuilt_dataset_path is None) != (args.prebuilt_panel_path is None):
        parser.error("--prebuilt-dataset-path and --prebuilt-panel-path must be provided together.")

    repo_root = Path(__file__).resolve().parents[1]
    markets_path = args.markets_path.resolve()
    config_path = args.config.resolve()
    horizon_policy_path = args.horizon_policy.resolve() if args.horizon_policy is not None else None
    output_root = args.output_root.resolve()
    prebuilt_dataset_path = args.prebuilt_dataset_path.resolve() if args.prebuilt_dataset_path is not None else None
    prebuilt_panel_path = args.prebuilt_panel_path.resolve() if args.prebuilt_panel_path is not None else None
    cities = args.city or DEFAULT_CITIES
    horizon_policy = _load_horizon_policy(horizon_policy_path)
    prebuilt_dataset = pd.read_parquet(prebuilt_dataset_path) if prebuilt_dataset_path is not None else None
    prebuilt_panel = pd.read_parquet(prebuilt_panel_path) if prebuilt_panel_path is not None else None

    snapshots = load_market_snapshot_payloads(markets_path)
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
        "prebuilt_dataset_path": str(prebuilt_dataset_path) if prebuilt_dataset_path is not None else None,
        "prebuilt_panel_path": str(prebuilt_panel_path) if prebuilt_panel_path is not None else None,
        "prebuilt_start_date": args.prebuilt_start_date,
        "prebuilt_end_date": args.prebuilt_end_date,
        "prebuilt_last_n_market_days": args.prebuilt_last_n_market_days,
        "model_name": args.model_name,
        "variant": args.variant or "",
        "variant_spec": str(args.variant_spec.resolve()) if args.variant_spec is not None else None,
        "retrain_stride": args.retrain_stride,
        "backtest_last_n": args.backtest_last_n,
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
        dataset_path = city_root / "parquet" / "gold" / "v2" / f"{dataset_name}.parquet"
        if prebuilt_dataset is not None:
            if not (args.reuse_existing and dataset_path.exists()):
                _write_prebuilt_city_frame(
                    prebuilt_dataset,
                    city=city,
                    output_path=dataset_path,
                    label="dataset",
                    start_date=args.prebuilt_start_date,
                    end_date=args.prebuilt_end_date,
                    last_n_market_days=args.prebuilt_last_n_market_days,
                )
        elif not (args.reuse_existing and dataset_path.exists()):
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
        city_dataset = pd.read_parquet(dataset_path)
        dataset_market_days, dataset_start_date, dataset_end_date = _city_dataset_range(city_dataset)
        split_count = _backtest_split_count(city_dataset, backtest_last_n=args.backtest_last_n)
        _validate_retrain_stride(
            retrain_stride=args.retrain_stride,
            split_count=split_count,
            city=city,
        )

        panel_path = city_root / "parquet" / "gold" / "v2" / f"{panel_name}.parquet"
        if prebuilt_panel is not None:
            if not (args.reuse_existing and panel_path.exists()):
                _write_prebuilt_city_frame(
                    prebuilt_panel,
                    city=city,
                    output_path=panel_path,
                    label="panel",
                    start_date=args.prebuilt_start_date,
                    end_date=args.prebuilt_end_date,
                    last_n_market_days=args.prebuilt_last_n_market_days,
                )
        elif not (args.reuse_existing and panel_path.exists()):
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
            real_backtest_cmd = [
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
                "--retrain-stride",
                str(args.retrain_stride),
            ]
            if args.backtest_last_n > 0:
                real_backtest_cmd.extend(["--last-n", str(args.backtest_last_n)])
            if args.variant:
                real_backtest_cmd.extend(["--variant", args.variant])
            if args.variant_spec is not None:
                real_backtest_cmd.extend(["--variant-spec", str(args.variant_spec.resolve())])
            _run(
                real_backtest_cmd,
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
            proxy_backtest_cmd = [
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
                "--retrain-stride",
                str(args.retrain_stride),
            ]
            if args.backtest_last_n > 0:
                proxy_backtest_cmd.extend(["--last-n", str(args.backtest_last_n)])
            if args.variant:
                proxy_backtest_cmd.extend(["--variant", args.variant])
            if args.variant_spec is not None:
                proxy_backtest_cmd.extend(["--variant-spec", str(args.variant_spec.resolve())])
            _run(
                proxy_backtest_cmd,
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
        panel_summary = _panel_summary(
            panel_path,
            dataset_path=dataset_path,
            backtest_last_n=args.backtest_last_n,
        )
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
                aggregate_panel_coverage={
                    "rows": float(horizon_panel_summary["rows"]),
                    "coverage": dict(horizon_panel_summary["coverage"]),
                    "ok_ratio": (
                        float(horizon_panel_summary["coverage"].get("ok", 0)) / float(horizon_panel_summary["rows"])
                        if float(horizon_panel_summary["rows"]) > 0
                        else 0.0
                    ),
                },
                min_priced_decision_rows=40.0,
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
            "snapshot_count": dataset_market_days if prebuilt_dataset is not None else snapshot_counts[city],
            "start_date": dataset_start_date if prebuilt_dataset is not None else snapshot_ranges[city]["start_date"],
            "end_date": dataset_end_date if prebuilt_dataset is not None else snapshot_ranges[city]["end_date"],
            "dataset_rows": dataset_rows,
            "dataset_path": str(dataset_path),
            "panel_path": str(panel_path),
            "panel_summary": panel_summary,
            "allowed_horizons": allowed_horizons,
            "split_count": split_count,
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
