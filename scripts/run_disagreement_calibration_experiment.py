"""Run disagreement-driven calibration experiments for curated multi-source LGBM."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pmtmax.markets.market_spec import MarketSpec
from pmtmax.modeling.disagreement_calibration import (
    DisagreementCalibratedGaussianModel,
    DisagreementCalibrationConfig,
)
from pmtmax.modeling.quick_eval import (
    _gaussian_crps_vectorized,
    _market_units,
    _values_to_celsius,
    evaluate_saved_model,
    quick_eval_holdout,
)
from pmtmax.modeling.train import load_model, sanitize_model_frame

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY_MODEL = (
    REPO_ROOT
    / "artifacts/workspaces/historical_real/models/curated_multisource_sentinelfix_20260426"
    / "lgbm_emos__high_neighbor_oof.pkl"
)
DEFAULT_FALLBACK_MODEL = (
    REPO_ROOT / "artifacts/workspaces/historical_real/models/curated_20260425"
    / "lgbm_emos__high_neighbor_oof.pkl"
)
DEFAULT_CURATED_DATASET = (
    REPO_ROOT
    / "data/workspaces/historical_real/parquet/gold/v2"
    / "historical_training_set_curated_multisource_sentinelfix_20260426.parquet"
)
DEFAULT_BASELINE_DATASET = (
    REPO_ROOT / "data/workspaces/historical_real/parquet/gold/historical_training_set.parquet"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts/workspaces/historical_real/models/curated_multisource_disagreementcal_20260426"
)
DEFAULT_REPORT = REPO_ROOT / "artifacts/curated_multisource_disagreementcal_experiment_20260426.json"


def _load_split(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = sanitize_model_frame(pd.read_parquet(path))
    fit, holdout = quick_eval_holdout(frame)
    return fit.reset_index(drop=True), holdout.reset_index(drop=True)


def _predict_gaussian_c(model: Any, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    units = _market_units(frame)
    mean, std = model.predict(frame)
    return (
        units,
        _values_to_celsius(np.asarray(mean).reshape(-1).astype(float), units),
        np.maximum(
            _values_to_celsius(np.asarray(std).reshape(-1).astype(float), units, scale=True),
            0.1,
        ),
    )


def _crps_summary(model: Any, frame: pd.DataFrame) -> dict[str, float]:
    units, mean_c, std_c = _predict_gaussian_c(model, frame)
    truth_c = _values_to_celsius(frame["realized_daily_max"].to_numpy(dtype=float), units)
    valid = np.isfinite(truth_c) & np.isfinite(mean_c) & np.isfinite(std_c)
    crps = _gaussian_crps_vectorized(mean_c[valid], std_c[valid], truth_c[valid])
    return {
        "n": float(valid.sum()),
        "crps_celsius_normalized": float(np.mean(crps)) if len(crps) else float("nan"),
        "p90_crps_celsius": float(np.quantile(crps, 0.9)) if len(crps) else float("nan"),
        "mae_celsius_normalized": float(np.mean(np.abs(truth_c[valid] - mean_c[valid])))
        if valid.any()
        else float("nan"),
        "mean_std_c": float(np.mean(std_c[valid])) if valid.any() else float("nan"),
        "median_std_c": float(np.median(std_c[valid])) if valid.any() else float("nan"),
    }


def _adjustment_summary(model: DisagreementCalibratedGaussianModel, frame: pd.DataFrame) -> dict[str, float]:
    adjustments = model.predict_adjustments(frame)
    weights = adjustments["blend_weight"]
    std_c = adjustments["std_c"]
    source_range = adjustments["source_range_c"]
    diff = adjustments["diff_c"]
    return {
        "mean_blend_weight": float(np.mean(weights)) if len(weights) else float("nan"),
        "median_blend_weight": float(np.median(weights)) if len(weights) else float("nan"),
        "p90_blend_weight": float(np.quantile(weights, 0.9)) if len(weights) else float("nan"),
        "share_blend_ge_0_5": float(np.mean(weights >= 0.5)) if len(weights) else float("nan"),
        "mean_std_c": float(np.mean(std_c)) if len(std_c) else float("nan"),
        "p90_std_c": float(np.quantile(std_c, 0.9)) if len(std_c) else float("nan"),
        "mean_source_range_c": float(np.mean(source_range)) if len(source_range) else float("nan"),
        "mean_positive_diff_c": float(np.mean(np.maximum(diff, 0.0))) if len(diff) else float("nan"),
        "mean_negative_diff_c": float(np.mean(np.maximum(-diff, 0.0))) if len(diff) else float("nan"),
    }


def _country_for_row(row: pd.Series) -> str:
    if "market_spec_json" not in row:
        return "unknown"
    try:
        spec = MarketSpec.model_validate_json(str(row["market_spec_json"]))
        return spec.country or "unknown"
    except Exception:
        return "unknown"


def _city_country_diagnostics(
    *,
    frame: pd.DataFrame,
    primary: Any,
    fallback: Any,
    candidate: DisagreementCalibratedGaussianModel,
    min_rows: int = 15,
) -> list[dict[str, Any]]:
    units, primary_mean_c, primary_std_c = _predict_gaussian_c(primary, frame)
    _, fallback_mean_c, fallback_std_c = _predict_gaussian_c(fallback, frame)
    _, candidate_mean_c, candidate_std_c = _predict_gaussian_c(candidate, frame)
    truth_c = _values_to_celsius(frame["realized_daily_max"].to_numpy(dtype=float), units)
    diagnostics = pd.DataFrame(
        {
            "city": frame.get("city", pd.Series(["unknown"] * len(frame))).astype(str),
            "country": frame.apply(_country_for_row, axis=1),
            "primary_crps": _gaussian_crps_vectorized(primary_mean_c, primary_std_c, truth_c),
            "fallback_crps": _gaussian_crps_vectorized(fallback_mean_c, fallback_std_c, truth_c),
            "candidate_crps": _gaussian_crps_vectorized(candidate_mean_c, candidate_std_c, truth_c),
            "primary_minus_fallback_mean_c": primary_mean_c - fallback_mean_c,
            "candidate_std_c": candidate_std_c,
        }
    )
    rows: list[dict[str, Any]] = []
    for (city, country), group in diagnostics.groupby(["city", "country"]):
        if len(group) < min_rows:
            continue
        rows.append(
            {
                "city": city,
                "country": country,
                "n": int(len(group)),
                "primary_crps_c": float(group["primary_crps"].mean()),
                "fallback_crps_c": float(group["fallback_crps"].mean()),
                "candidate_crps_c": float(group["candidate_crps"].mean()),
                "primary_minus_fallback_mean_c": float(group["primary_minus_fallback_mean_c"].mean()),
                "candidate_std_c": float(group["candidate_std_c"].mean()),
            }
        )
    return sorted(
        rows,
        key=lambda row: row["primary_crps_c"] - row["candidate_crps_c"],
        reverse=True,
    )


def _save_wrapper(path: Path, model: DisagreementCalibratedGaussianModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-model", type=Path, default=DEFAULT_PRIMARY_MODEL)
    parser.add_argument("--fallback-model", type=Path, default=DEFAULT_FALLBACK_MODEL)
    parser.add_argument("--curated-dataset", type=Path, default=DEFAULT_CURATED_DATASET)
    parser.add_argument("--baseline-dataset", type=Path, default=DEFAULT_BASELINE_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    primary = load_model(args.primary_model)
    fallback = load_model(args.fallback_model)
    curated_fit, curated_holdout = _load_split(args.curated_dataset)
    _, baseline_holdout = _load_split(args.baseline_dataset)

    candidate_configs = {
        "variance_only_disagreement": DisagreementCalibrationConfig(
            name="variance_only_disagreement",
            disagreement_variance_weight=0.75,
        ),
        "positive_shrink_light": DisagreementCalibrationConfig(
            name="positive_shrink_light",
            disagreement_variance_weight=0.05,
            mean_blend_mode="positive",
            mean_blend_weight_per_c=0.05,
            max_mean_blend_weight=0.5,
        ),
        "positive_shrink_balanced": DisagreementCalibrationConfig(
            name="positive_shrink_balanced",
            disagreement_variance_weight=0.1,
            mean_blend_mode="positive",
            mean_blend_weight_per_c=0.1,
            max_mean_blend_weight=1.0,
        ),
        "positive_shrink_minvar": DisagreementCalibrationConfig(
            name="positive_shrink_minvar",
            disagreement_variance_weight=0.05,
            mean_blend_mode="positive",
            mean_blend_weight_per_c=0.1,
            max_mean_blend_weight=1.0,
        ),
    }
    candidates: dict[str, DisagreementCalibratedGaussianModel] = {}
    candidate_paths: dict[str, Path] = {}
    for name, config in candidate_configs.items():
        model = DisagreementCalibratedGaussianModel(
            primary_model=primary,
            fallback_model=fallback,
            config=config,
            feature_names=list(getattr(primary, "feature_names", [])),
        )
        path = args.output_dir / f"lgbm_emos__high_neighbor_oof_{name}.pkl"
        _save_wrapper(path, model)
        candidates[name] = model
        candidate_paths[name] = path

    eval_models = {
        "raw_primary_multisource_sentinelfix": args.primary_model,
        "raw_fallback_gfs": args.fallback_model,
        **candidate_paths,
    }
    eval_datasets = {
        "curated_fit": curated_fit,
        "curated_sentinelfix_holdout": curated_holdout,
        "baseline_holdout": baseline_holdout,
    }
    quick_eval_results: dict[str, dict[str, dict[str, float] | None]] = {}
    for dataset_name, frame in eval_datasets.items():
        quick_eval_results[dataset_name] = {}
        for model_name, model_path in eval_models.items():
            quick_eval_results[dataset_name][model_name] = evaluate_saved_model(
                Path(model_path),
                frame,
            )

    diagnostics: dict[str, dict[str, Any]] = {}
    for dataset_name, frame in eval_datasets.items():
        diagnostics[dataset_name] = {
            "primary": _crps_summary(primary, frame),
            "fallback": _crps_summary(fallback, frame),
            "candidates": {name: _crps_summary(model, frame) for name, model in candidates.items()},
            "adjustments": {
                name: _adjustment_summary(model, frame)
                for name, model in candidates.items()
            },
        }

    best_name = min(
        candidates,
        key=lambda name: diagnostics["curated_sentinelfix_holdout"]["candidates"][name][
            "crps_celsius_normalized"
        ],
    )
    report = {
        "created_at": pd.Timestamp.now("UTC").isoformat(),
        "primary_model": str(args.primary_model),
        "fallback_model": str(args.fallback_model),
        "candidate_models": {name: str(path) for name, path in candidate_paths.items()},
        "curated_dataset": str(args.curated_dataset),
        "baseline_dataset": str(args.baseline_dataset),
        "configs": {name: asdict(config) for name, config in candidate_configs.items()},
        "quick_eval_results": quick_eval_results,
        "diagnostics": diagnostics,
        "best_curated_candidate": best_name,
        "city_country_diagnostics_for_best": _city_country_diagnostics(
            frame=curated_holdout,
            primary=primary,
            fallback=fallback,
            candidate=candidates[best_name],
        )[:30],
        "notes": [
            "variance_only_disagreement keeps the primary mean and only inflates std.",
            "positive_shrink_* also shrinks the mean toward GFS-only when multi-source is hotter than GFS-only.",
            "These are diagnostic post-hoc wrappers, not promotion candidates without recent-core/backtest gates.",
        ],
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(
        json.dumps(
            {
                "report_path": str(args.report_path),
                "candidate_models": {name: str(path) for name, path in candidate_paths.items()},
                "best_curated_candidate": best_name,
                "quick_eval_results": {
                    "curated_sentinelfix_holdout": quick_eval_results["curated_sentinelfix_holdout"],
                    "baseline_holdout": quick_eval_results["baseline_holdout"],
                },
                "adjustments": {
                    "curated_sentinelfix_holdout": diagnostics["curated_sentinelfix_holdout"][
                        "adjustments"
                    ],
                    "baseline_holdout": diagnostics["baseline_holdout"]["adjustments"],
                },
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
