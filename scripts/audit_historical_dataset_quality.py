"""Audit the historical-real gold dataset for leakage, feature quality, and split robustness.

This script is intentionally lightweight: it only reads local parquet/JSON artifacts and writes
machine-readable + markdown reports. It does not mutate the canonical dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DATASET = Path(
    "data/workspaces/historical_real/parquet/gold/historical_training_set.parquet"
)
DEFAULT_PANEL = Path(
    "data/workspaces/historical_real/parquet/gold/v2/historical_backtest_panel.parquet"
)
DEFAULT_READINESS = Path("artifacts/dataset_readiness.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/workspaces/historical_real/quality")
DEFAULT_DOC = Path("docs/research/historical_dataset_card.md")

IDENTITY_COLUMNS = {
    "market_id",
    "station_id",
    "city",
    "truth_track",
    "settlement_eligible",
    "target_date",
    "decision_horizon",
    "decision_time_utc",
    "issue_time_utc",
    "lead_hours",
    "market_spec_json",
    "market_prices_json",
    "winning_outcome",
    "available_models_json",
    "selected_models_json",
    "forecast_source_kind",
    "run_id",
    "data_version",
    "created_at",
    "source_priority",
    "contract_version",
    "group_id",
    "split_group",
    "feature_availability_json",
}
LEAKY_NAME_TOKENS = ("realized", "winning", "settlement", "truth")
BASELINE_FORECAST_COLUMNS = (
    "model_daily_max",
    "gfs_seamless_model_daily_max",
    "neighbor_mean_temp",
    "gfs_seamless_midday_temp",
)


@dataclass
class Check:
    name: str
    status: str
    detail: str
    value: Any | None = None


def _json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _status_from_bool(ok: bool, warn: bool = False) -> str:
    if ok:
        return "pass"
    return "warn" if warn else "fail"


def _round_float(value: float | int | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def _numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric_columns = set(frame.select_dtypes(include="number").columns)
    return sorted(
        column
        for column in numeric_columns
        if column not in IDENTITY_COLUMNS and column != "realized_daily_max"
    )


def _time_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce", utc=True)


def _baseline_error_summary(frame: pd.DataFrame, forecast_column: str) -> dict[str, Any]:
    if forecast_column not in frame.columns or "realized_daily_max" not in frame.columns:
        return {"available": False}
    forecast = pd.to_numeric(frame[forecast_column], errors="coerce")
    actual = pd.to_numeric(frame["realized_daily_max"], errors="coerce")
    valid = forecast.notna() & actual.notna()
    if not valid.any():
        return {"available": False}
    error = forecast[valid] - actual[valid]
    return {
        "available": True,
        "rows": int(valid.sum()),
        "bias": _round_float(error.mean()),
        "mae": _round_float(error.abs().mean()),
        "rmse": _round_float(math.sqrt(float((error**2).mean()))),
        "p95_abs_error": _round_float(error.abs().quantile(0.95)),
        "p99_abs_error": _round_float(error.abs().quantile(0.99)),
    }


def _city_baseline_summary(frame: pd.DataFrame, forecast_column: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for city, group in frame.groupby("city", dropna=False):
        summary = _baseline_error_summary(group, forecast_column)
        if summary.get("available"):
            rows.append({"city": str(city), **summary})
    rows.sort(key=lambda row: (-int(row["rows"]), str(row["city"])))
    maes = [float(row["mae"]) for row in rows if row.get("mae") is not None]
    return {
        "macro_mae": _round_float(sum(maes) / len(maes)) if maes else None,
        "city_count": len(rows),
        "worst_10_by_mae": sorted(rows, key=lambda row: float(row.get("mae") or 0), reverse=True)[
            :10
        ],
        "small_sample_cities": [row for row in rows if int(row["rows"]) < 90],
    }


def _time_forward_summary(
    frame: pd.DataFrame, forecast_column: str, folds: int = 4
) -> dict[str, Any]:
    if "target_date" not in frame.columns:
        return {"available": False}
    ordered_dates = sorted(
        pd.to_datetime(frame["target_date"], errors="coerce").dropna().dt.normalize().unique()
    )
    if len(ordered_dates) < folds + 1:
        return {"available": False, "reason": "too_few_dates"}
    fold_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        start = int(len(ordered_dates) * (fold + 1) / (folds + 1))
        end = (
            int(len(ordered_dates) * (fold + 2) / (folds + 1))
            if fold < folds - 1
            else len(ordered_dates)
        )
        test_dates = set(ordered_dates[start:end])
        test = frame.loc[
            pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize().isin(test_dates)
        ]
        summary = _baseline_error_summary(test, forecast_column)
        if summary.get("available"):
            fold_rows.append(
                {
                    "fold": fold + 1,
                    "start_date": pd.Timestamp(min(test_dates)).date().isoformat(),
                    "end_date": pd.Timestamp(max(test_dates)).date().isoformat(),
                    **summary,
                }
            )
    maes = [float(row["mae"]) for row in fold_rows if row.get("mae") is not None]
    return {
        "available": bool(fold_rows),
        "folds": fold_rows,
        "macro_fold_mae": _round_float(sum(maes) / len(maes)) if maes else None,
        "mae_range": [_round_float(min(maes)), _round_float(max(maes))] if maes else None,
    }


def _quality_score(checks: list[Check], payload: dict[str, Any]) -> dict[str, Any]:
    # Conservative 10-point rubric. A score near 9 requires not only clean rows but also
    # strong truth fidelity and representative/evaluated splits.
    score = 10.0
    fail_count = sum(1 for check in checks if check.status == "fail")
    warn_count = sum(1 for check in checks if check.status == "warn")
    score -= fail_count * 1.0
    score -= warn_count * 0.25

    truth_track = payload.get("truth_track_counts", {})
    total_truth = sum(int(v) for v in truth_track.values()) or 1
    exact_ratio = int(truth_track.get("exact_public", 0)) / total_truth
    if exact_ratio < 0.25:
        score -= 1.25
    elif exact_ratio < 0.60:
        score -= 0.75

    city_rows = payload.get("city_counts", {})
    if city_rows:
        max_city_ratio = max(int(v) for v in city_rows.values()) / sum(
            int(v) for v in city_rows.values()
        )
        min_city_rows = min(int(v) for v in city_rows.values())
        if max_city_ratio > 0.15:
            score -= 0.35
        if min_city_rows < 90:
            score -= 0.35

    readiness = payload.get("readiness_status_counts", {})
    non_ready = sum(int(v) for k, v in readiness.items() if k != "ready")
    if non_ready:
        score -= min(0.5, non_ready / 100)

    return {
        "score_10pt": max(0.0, _round_float(score, 2)),
        "grade": "research-grade" if score >= 7.0 else "needs-repair",
        "target_9pt_blockers": [
            "Increase high-confidence/exact truth-track coverage or explicitly tier training/evaluation sets.",
            "Run model-level city/time split backtests, not only deterministic forecast audits.",
            "Balance or weight city exposure; current small-sample cities should not dominate claims.",
            "Remove or repair constant/dead features before model training.",
        ],
    }


def build_report(dataset_path: Path, panel_path: Path, readiness_path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(dataset_path)
    readiness = _read_json(readiness_path)
    checks: list[Check] = []

    checks.append(Check("dataset_exists", "pass", str(dataset_path), dataset_path.exists()))
    checks.append(
        Check("non_empty", _status_from_bool(len(frame) > 0), f"rows={len(frame)}", int(len(frame)))
    )
    dup_count = int(frame.duplicated().sum())
    checks.append(
        Check(
            "duplicate_rows",
            _status_from_bool(dup_count == 0),
            f"duplicate_rows={dup_count}",
            dup_count,
        )
    )
    null_cells = int(frame.isna().sum().sum())
    checks.append(
        Check(
            "null_cells",
            _status_from_bool(null_cells == 0, warn=True),
            f"null_cells={null_cells}",
            null_cells,
        )
    )

    if {"market_id", "decision_horizon"}.issubset(frame.columns):
        rows_per_market = frame.groupby("market_id")["decision_horizon"].nunique()
        bad_markets = int((rows_per_market != 3).sum())
        checks.append(
            Check(
                "three_horizons_per_market",
                _status_from_bool(bad_markets == 0, warn=True),
                f"markets_without_3_horizons={bad_markets}",
                bad_markets,
            )
        )

    if {"issue_time_utc", "decision_time_utc"}.issubset(frame.columns):
        issue_time = _time_series(frame, "issue_time_utc")
        decision_time = _time_series(frame, "decision_time_utc")
        invalid_time_order = int(
            ((issue_time > decision_time) | issue_time.isna() | decision_time.isna()).sum()
        )
        checks.append(
            Check(
                "issue_time_not_after_decision_time",
                _status_from_bool(invalid_time_order == 0),
                f"invalid_rows={invalid_time_order}",
                invalid_time_order,
            )
        )

    feature_columns = _numeric_feature_columns(frame)
    leaky_named = [
        column
        for column in feature_columns
        if any(token in column.lower() for token in LEAKY_NAME_TOKENS)
    ]
    checks.append(
        Check(
            "no_obvious_leaky_feature_names",
            _status_from_bool(not leaky_named),
            f"leaky_named_features={leaky_named}",
            leaky_named,
        )
    )
    constant_features = [
        column for column in feature_columns if frame[column].nunique(dropna=True) <= 1
    ]
    checks.append(
        Check(
            "no_constant_numeric_features",
            _status_from_bool(not constant_features, warn=True),
            f"constant_features={constant_features}",
            constant_features,
        )
    )

    truth_track_counts = {
        str(k): int(v)
        for k, v in frame.get("truth_track", pd.Series(dtype=object))
        .value_counts(dropna=False)
        .to_dict()
        .items()
    }
    research_public_rows = int(truth_track_counts.get("research_public", 0))
    checks.append(
        Check(
            "truth_tier_explicit",
            _status_from_bool(bool(truth_track_counts), warn=True),
            f"truth_track_counts={truth_track_counts}",
            truth_track_counts,
        )
    )
    checks.append(
        Check(
            "exact_truth_coverage",
            "warn" if research_public_rows else "pass",
            "Most rows are research_public; keep as research-grade or tier exact/proxy datasets.",
            truth_track_counts,
        )
    )

    readiness_status_counts: dict[str, int] = {}
    if readiness.get("details"):
        readiness_status_counts = dict(
            Counter(str(row.get("readiness_status")) for row in readiness["details"])
        )
        non_ready = sum(
            count for status, count in readiness_status_counts.items() if status != "ready"
        )
        checks.append(
            Check(
                "readiness_non_ready_small",
                _status_from_bool(non_ready <= 30, warn=True),
                f"non_ready={non_ready}, status_counts={readiness_status_counts}",
                non_ready,
            )
        )

    panel_summary: dict[str, Any] = {"exists": panel_path.exists()}
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        panel_summary = {
            "exists": True,
            "rows": int(len(panel)),
            "columns": list(panel.columns),
            "coverage_status_counts": {
                str(k): int(v)
                for k, v in panel.get("coverage_status", pd.Series(dtype=object))
                .value_counts(dropna=False)
                .to_dict()
                .items()
            },
        }
        ok_rows = int(panel_summary["coverage_status_counts"].get("ok", 0))
        checks.append(
            Check(
                "panel_has_official_price_rows",
                _status_from_bool(ok_rows > 0, warn=True),
                f"ok_rows={ok_rows}",
                ok_rows,
            )
        )
    else:
        checks.append(Check("panel_exists", "warn", f"missing panel: {panel_path}", None))

    city_counts = {
        str(k): int(v)
        for k, v in frame.get("city", pd.Series(dtype=object)).value_counts().to_dict().items()
    }
    horizon_counts = {
        str(k): int(v)
        for k, v in frame.get("decision_horizon", pd.Series(dtype=object))
        .value_counts()
        .to_dict()
        .items()
    }
    target_dates = (
        pd.to_datetime(frame["target_date"], errors="coerce")
        if "target_date" in frame.columns
        else pd.Series(dtype="datetime64[ns]")
    )

    baseline = {
        column: _baseline_error_summary(frame, column) for column in BASELINE_FORECAST_COLUMNS
    }
    split_benchmarks = {
        "time_forward_model_daily_max": _time_forward_summary(frame, "model_daily_max"),
        "city_holdout_model_daily_max": _city_baseline_summary(frame, "model_daily_max"),
    }

    payload: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017 - keep Python 3.8-compatible when run outside uv
        "dataset_path": str(dataset_path),
        "panel_path": str(panel_path),
        "readiness_path": str(readiness_path),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "market_count": int(frame["market_id"].nunique()) if "market_id" in frame.columns else None,
        "city_count": int(frame["city"].nunique()) if "city" in frame.columns else None,
        "target_date_range": [
            target_dates.min().date().isoformat()
            if len(target_dates) and pd.notna(target_dates.min())
            else None,
            target_dates.max().date().isoformat()
            if len(target_dates) and pd.notna(target_dates.max())
            else None,
        ],
        "city_counts": city_counts,
        "horizon_counts": horizon_counts,
        "truth_track_counts": truth_track_counts,
        "settlement_eligible_counts": {
            str(k): int(v)
            for k, v in frame.get("settlement_eligible", pd.Series(dtype=object))
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "forecast_source_kind_counts": {
            str(k): int(v)
            for k, v in frame.get("forecast_source_kind", pd.Series(dtype=object))
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "numeric_feature_columns": feature_columns,
        "constant_numeric_features": constant_features,
        "baseline_forecast_quality": baseline,
        "split_benchmarks": split_benchmarks,
        "panel_summary": panel_summary,
        "readiness_status_counts": readiness_status_counts,
        "checks": [asdict(check) for check in checks],
    }
    payload["quality_score"] = _quality_score(checks, payload)
    return payload


def _markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    score = report["quality_score"]
    checks = report["checks"]
    status_counts = Counter(check["status"] for check in checks)
    baseline = report["baseline_forecast_quality"].get("model_daily_max", {})
    time_forward = report["split_benchmarks"]["time_forward_model_daily_max"]
    city_holdout = report["split_benchmarks"]["city_holdout_model_daily_max"]

    check_rows = [[check["status"], check["name"], check["detail"]] for check in checks]
    city_rows = [[city, rows] for city, rows in list(report["city_counts"].items())[:15]]
    worst_city_rows = [
        [row["city"], row["rows"], row["mae"], row["bias"]]
        for row in city_holdout.get("worst_10_by_mae", [])
    ]
    time_rows = [
        [row["fold"], row["start_date"], row["end_date"], row["rows"], row["mae"], row["bias"]]
        for row in time_forward.get("folds", [])
    ]

    return (
        f"""# Historical Real Dataset Card

Generated: `{report["generated_at"]}`

## Verdict

- Quality score: **{score["score_10pt"]}/10** ({score["grade"]})
- Check statuses: `{dict(status_counts)}`
- Rows: **{report["row_count"]}** / Markets: **{report["market_count"]}** / Cities: **{report["city_count"]}**
- Target date range: **{report["target_date_range"][0]} → {report["target_date_range"][1]}**

## Core Coverage

- Horizon counts: `{report["horizon_counts"]}`
- Truth tracks: `{report["truth_track_counts"]}`
- Settlement eligible: `{report["settlement_eligible_counts"]}`
- Forecast source kind: `{report["forecast_source_kind_counts"]}`
- Readiness statuses: `{report["readiness_status_counts"]}`

## Baseline Forecast Quality

`model_daily_max` vs `realized_daily_max`:

- rows: `{baseline.get("rows")}`
- bias: `{baseline.get("bias")}`
- MAE: `{baseline.get("mae")}`
- RMSE: `{baseline.get("rmse")}`
- p95 abs error: `{baseline.get("p95_abs_error")}`
- p99 abs error: `{baseline.get("p99_abs_error")}`

## Time-Forward Baseline Benchmark

Macro fold MAE: `{time_forward.get("macro_fold_mae")}`; MAE range: `{time_forward.get("mae_range")}`

{_markdown_table(time_rows, ["fold", "start", "end", "rows", "mae", "bias"]) if time_rows else "_No time-forward folds available._"}

## City Holdout Diagnostic

Macro city MAE: `{city_holdout.get("macro_mae")}` across `{city_holdout.get("city_count")}` cities.

Worst 10 cities by baseline MAE:

{_markdown_table(worst_city_rows, ["city", "rows", "mae", "bias"]) if worst_city_rows else "_No city diagnostics available._"}

Small-sample cities: `{[row["city"] for row in city_holdout.get("small_sample_cities", [])]}`

Top city row counts:

{_markdown_table(city_rows, ["city", "rows"]) if city_rows else "_No city counts available._"}

## Leakage / Feature Audit Checks

{_markdown_table(check_rows, ["status", "check", "detail"])}

## Current 9/10 Blockers

"""
        + "\n".join(f"- {item}" for item in score["target_9pt_blockers"])
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--readiness-path", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-output", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()

    report = build_report(args.dataset_path, args.panel_path, args.readiness_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.doc_output.parent.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "historical_dataset_quality.json"
    md_path = args.output_dir / "historical_dataset_quality.md"
    markdown = render_markdown(report)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n"
    )
    md_path.write_text(markdown)
    args.doc_output.write_text(markdown)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "doc": str(args.doc_output),
                "score": report["quality_score"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
