"""Run train-side source-gating experiments for curated multi-source LGBM."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pmtmax.modeling.quick_eval import (
    _gaussian_crps_vectorized,
    _market_units,
    _values_to_celsius,
    evaluate_saved_model,
    quick_eval_holdout,
)
from pmtmax.modeling.source_gating import (
    SourceGatedGaussianModel,
    SourceGatingConfig,
    fit_source_gated_model,
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
    / "artifacts/workspaces/historical_real/models/curated_multisource_sourcegate_20260426"
)
DEFAULT_REPORT = REPO_ROOT / "artifacts/curated_multisource_sourcegate_experiment_20260426.json"


def _load_split(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = sanitize_model_frame(pd.read_parquet(path))
    fit, holdout = quick_eval_holdout(frame)
    return fit.reset_index(drop=True), holdout.reset_index(drop=True)


def _predict_gaussian_c(model: Any, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    units = _market_units(frame)
    mean, std = model.predict(frame)
    mean_c = _values_to_celsius(np.asarray(mean).reshape(-1).astype(float), units)
    std_c = np.maximum(
        _values_to_celsius(np.asarray(std).reshape(-1).astype(float), units, scale=True),
        0.1,
    )
    return units, mean_c, std_c


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


def _oracle_summary(primary: Any, fallback: Any, frame: pd.DataFrame) -> dict[str, float]:
    units, primary_mean_c, primary_std_c = _predict_gaussian_c(primary, frame)
    _, fallback_mean_c, fallback_std_c = _predict_gaussian_c(fallback, frame)
    truth_c = _values_to_celsius(frame["realized_daily_max"].to_numpy(dtype=float), units)
    valid = (
        np.isfinite(truth_c)
        & np.isfinite(primary_mean_c)
        & np.isfinite(primary_std_c)
        & np.isfinite(fallback_mean_c)
        & np.isfinite(fallback_std_c)
    )
    primary_crps = _gaussian_crps_vectorized(
        primary_mean_c[valid],
        primary_std_c[valid],
        truth_c[valid],
    )
    fallback_crps = _gaussian_crps_vectorized(
        fallback_mean_c[valid],
        fallback_std_c[valid],
        truth_c[valid],
    )
    oracle = np.minimum(primary_crps, fallback_crps)
    return {
        "n": float(valid.sum()),
        "fallback_better_rate": float(np.mean(fallback_crps < primary_crps)) if valid.any() else 0.0,
        "oracle_min_crps_celsius_normalized": float(np.mean(oracle)) if len(oracle) else float("nan"),
        "primary_minus_fallback_mean_diff_c": float(np.mean(primary_mean_c[valid] - fallback_mean_c[valid]))
        if valid.any()
        else float("nan"),
        "primary_minus_fallback_abs_diff_c": float(
            np.mean(np.abs(primary_mean_c[valid] - fallback_mean_c[valid])),
        )
        if valid.any()
        else float("nan"),
    }


def _gate_weight_summary(model: SourceGatedGaussianModel, frame: pd.DataFrame) -> dict[str, float]:
    weights = model.predict_fallback_weight(frame)
    return {
        "mean": float(np.mean(weights)) if len(weights) else float("nan"),
        "median": float(np.median(weights)) if len(weights) else float("nan"),
        "p10": float(np.quantile(weights, 0.1)) if len(weights) else float("nan"),
        "p90": float(np.quantile(weights, 0.9)) if len(weights) else float("nan"),
        "share_ge_0_5": float(np.mean(weights >= 0.5)) if len(weights) else float("nan"),
    }


def _save_wrapper(path: Path, model: SourceGatedGaussianModel) -> None:
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
    parser.add_argument("--crps-margin-c", type=float, default=0.0)
    parser.add_argument("--min-fallback-weight", type=float, default=0.0)
    parser.add_argument("--max-fallback-weight", type=float, default=1.0)
    args = parser.parse_args()

    primary = load_model(args.primary_model)
    fallback = load_model(args.fallback_model)
    curated_fit, curated_holdout = _load_split(args.curated_dataset)
    _, baseline_holdout = _load_split(args.baseline_dataset)

    candidate_configs = {
        "sourcegate_binary": SourceGatingConfig(
            name="sourcegate_binary",
            crps_margin_c=args.crps_margin_c,
            min_fallback_weight=args.min_fallback_weight,
            max_fallback_weight=args.max_fallback_weight,
            sample_weight_mode="uniform",
        ),
        "sourcegate_regret_weighted": SourceGatingConfig(
            name="sourcegate_regret_weighted",
            crps_margin_c=args.crps_margin_c,
            min_fallback_weight=args.min_fallback_weight,
            max_fallback_weight=args.max_fallback_weight,
            sample_weight_mode="fallback_regret",
            random_state=1730,
        ),
        "sourcegate_absregret_weighted": SourceGatingConfig(
            name="sourcegate_absregret_weighted",
            crps_margin_c=args.crps_margin_c,
            min_fallback_weight=args.min_fallback_weight,
            max_fallback_weight=args.max_fallback_weight,
            sample_weight_mode="absolute_regret",
            random_state=1731,
        ),
    }
    sourcegates: dict[str, SourceGatedGaussianModel] = {}
    sourcegate_paths: dict[str, Path] = {}
    gate_fit_diagnostics: dict[str, dict[str, float | str]] = {}
    for name, config in candidate_configs.items():
        sourcegate, diagnostics = fit_source_gated_model(
            curated_fit,
            primary_model=primary,
            fallback_model=fallback,
            config=config,
        )
        sourcegate_path = args.output_dir / f"lgbm_emos__high_neighbor_oof_{name}.pkl"
        _save_wrapper(sourcegate_path, sourcegate)
        sourcegates[name] = sourcegate
        sourcegate_paths[name] = sourcegate_path
        gate_fit_diagnostics[name] = diagnostics

    eval_models = {
        "raw_primary_multisource_sentinelfix": args.primary_model,
        "raw_fallback_gfs": args.fallback_model,
        **sourcegate_paths,
    }
    eval_datasets = {
        "curated_sentinelfix_holdout": curated_holdout,
        "baseline_holdout": baseline_holdout,
    }
    quick_eval_results: dict[str, dict[str, dict[str, float] | None]] = {}
    for dataset_name, holdout in eval_datasets.items():
        quick_eval_results[dataset_name] = {}
        for model_name, model_path in eval_models.items():
            quick_eval_results[dataset_name][model_name] = evaluate_saved_model(
                Path(model_path),
                holdout,
            )

    diagnostic_frames = {
        "curated_fit": curated_fit,
        "curated_sentinelfix_holdout": curated_holdout,
        "baseline_holdout": baseline_holdout,
    }
    gaussian_diagnostics: dict[str, dict[str, Any]] = {}
    for frame_name, frame in diagnostic_frames.items():
        gaussian_diagnostics[frame_name] = {
            "primary": _crps_summary(primary, frame),
            "fallback": _crps_summary(fallback, frame),
            "oracle": _oracle_summary(primary, fallback, frame),
            "sourcegates": {
                name: _crps_summary(sourcegate, frame)
                for name, sourcegate in sourcegates.items()
            },
            "fallback_weight": {
                name: _gate_weight_summary(sourcegate, frame)
                for name, sourcegate in sourcegates.items()
            },
        }

    report = {
        "created_at": pd.Timestamp.now("UTC").isoformat(),
        "primary_model": str(args.primary_model),
        "fallback_model": str(args.fallback_model),
        "sourcegate_models": {name: str(path) for name, path in sourcegate_paths.items()},
        "curated_dataset": str(args.curated_dataset),
        "baseline_dataset": str(args.baseline_dataset),
        "configs": {name: asdict(config) for name, config in candidate_configs.items()},
        "gate_fit_diagnostics": gate_fit_diagnostics,
        "feature_columns": {
            name: sourcegate.feature_columns for name, sourcegate in sourcegates.items()
        },
        "city_categories": {
            name: sourcegate.city_categories for name, sourcegate in sourcegates.items()
        },
        "horizon_categories": {
            name: sourcegate.horizon_categories for name, sourcegate in sourcegates.items()
        },
        "quick_eval_results": quick_eval_results,
        "gaussian_diagnostics": gaussian_diagnostics,
        "notes": [
            "Gate labels are fit on the quick_eval fit partition, not on either holdout.",
            "Primary and fallback component models were already fit before this post-hoc gate.",
            "Treat as diagnostic until validated by recent-core or backtest gate.",
        ],
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(
        json.dumps(
            {
                "report_path": str(args.report_path),
                "sourcegate_models": {name: str(path) for name, path in sourcegate_paths.items()},
                "quick_eval_results": quick_eval_results,
                "gate_fit_diagnostics": gate_fit_diagnostics,
                "weight_summary": {
                    frame_name: gaussian_diagnostics[frame_name]["fallback_weight"]
                    for frame_name in ["curated_sentinelfix_holdout", "baseline_holdout"]
                },
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
