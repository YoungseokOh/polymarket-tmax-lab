"""Run post-hoc tail calibration experiments for curated multi-source LGBM."""

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
from pmtmax.modeling.tail_calibration import (
    TailCalibratedGaussianModel,
    TailCalibrationConfig,
    sentinel_daily_max_count,
)
from pmtmax.modeling.train import load_model, sanitize_model_frame

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY_MODEL = (
    REPO_ROOT
    / "artifacts/workspaces/historical_real/models/curated_multisource_20260425/lgbm_emos__high_neighbor_oof.pkl"
)
DEFAULT_FALLBACK_MODEL = (
    REPO_ROOT / "artifacts/workspaces/historical_real/models/curated_20260425/lgbm_emos__high_neighbor_oof.pkl"
)
DEFAULT_CURATED_DATASET = (
    REPO_ROOT / "data/workspaces/historical_real/parquet/gold/v2/historical_training_set_curated_multisource_20260425.parquet"
)
DEFAULT_BASELINE_DATASET = REPO_ROOT / "data/workspaces/historical_real/parquet/gold/historical_training_set.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/workspaces/historical_real/models/curated_multisource_tailcal_20260426"
DEFAULT_REPORT = REPO_ROOT / "artifacts/curated_multisource_tailcal_experiment_20260426.json"


def _prediction_cache(frame: pd.DataFrame, *, primary: Any, fallback: Any) -> dict[str, Any]:
    units = _market_units(frame)
    y_c = _values_to_celsius(frame["realized_daily_max"].to_numpy(dtype=float), units)

    primary_mean, primary_std = primary.predict(frame)
    fallback_mean, fallback_std = fallback.predict(frame)
    return {
        "units": units,
        "truth_c": y_c,
        "primary_mean_c": _values_to_celsius(np.asarray(primary_mean).reshape(-1).astype(float), units),
        "primary_std_c": np.maximum(
            _values_to_celsius(np.asarray(primary_std).reshape(-1).astype(float), units, scale=True),
            0.1,
        ),
        "fallback_mean_c": _values_to_celsius(np.asarray(fallback_mean).reshape(-1).astype(float), units),
        "fallback_std_c": np.maximum(
            _values_to_celsius(np.asarray(fallback_std).reshape(-1).astype(float), units, scale=True),
            0.1,
        ),
        "sentinel_count": sentinel_daily_max_count(frame, units=units),
    }


def _apply_candidate(cache: dict[str, Any], cfg: TailCalibrationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_c = np.asarray(cache["primary_mean_c"], dtype=float) + cfg.mean_shift_c
    primary_std_c = np.asarray(cache["primary_std_c"], dtype=float)
    std_c = np.maximum(primary_std_c * cfg.scale_multiplier + cfg.scale_additive_c, cfg.scale_floor_c)
    mask = np.zeros(len(mean_c), dtype=bool)
    if cfg.fallback_std_threshold_c is not None:
        mask |= primary_std_c < cfg.fallback_std_threshold_c
    if cfg.fallback_sentinel_min_count is not None:
        mask |= np.asarray(cache["sentinel_count"], dtype=int) >= cfg.fallback_sentinel_min_count

    if mask.any():
        fallback_mean_c = np.asarray(cache["fallback_mean_c"], dtype=float)
        fallback_std_c = np.asarray(cache["fallback_std_c"], dtype=float)
        mean_weight = float(np.clip(cfg.fallback_mean_weight, 0.0, 1.0))
        mean_c[mask] = (1.0 - mean_weight) * mean_c[mask] + mean_weight * fallback_mean_c[mask]
        if cfg.fallback_std_mode == "max":
            std_c[mask] = np.maximum(std_c[mask], fallback_std_c[mask])
        elif cfg.fallback_std_mode == "blend":
            std_weight = mean_weight if cfg.fallback_std_weight is None else float(np.clip(cfg.fallback_std_weight, 0.0, 1.0))
            std_c[mask] = (1.0 - std_weight) * std_c[mask] + std_weight * fallback_std_c[mask]
        elif cfg.fallback_std_mode == "fallback":
            std_c[mask] = fallback_std_c[mask]
        std_c = np.maximum(std_c, cfg.scale_floor_c)
    return mean_c, std_c, mask


def _crps_for(cache: dict[str, Any], cfg: TailCalibrationConfig) -> dict[str, float]:
    mean_c, std_c, mask = _apply_candidate(cache, cfg)
    crps = _gaussian_crps_vectorized(mean_c, std_c, np.asarray(cache["truth_c"], dtype=float))
    return {
        "crps_celsius_normalized": float(np.mean(crps)),
        "p90_crps_celsius": float(np.quantile(crps, 0.9)),
        "masked_ratio": float(np.mean(mask)),
        "mean_std_c": float(np.mean(std_c)),
        "median_std_c": float(np.median(std_c)),
    }


def _candidate_grid() -> list[TailCalibrationConfig]:
    configs: list[TailCalibrationConfig] = []
    for threshold in [1.5, 2.0, 2.5, 3.0, 4.0, 999.0]:
        for sentinel_min in [1, 2, 3, 4, 5]:
            for fallback_weight in [0.25, 0.5, 0.75, 1.0]:
                for std_mode in ["max", "blend", "fallback"]:
                    for floor in [1.0, 2.0, 3.0, 4.0]:
                        configs.append(
                            TailCalibrationConfig(
                                name=(
                                    f"grid_t{threshold:g}_s{sentinel_min}_w{fallback_weight:g}_"
                                    f"{std_mode}_floor{floor:g}"
                                ),
                                scale_floor_c=floor,
                                fallback_std_threshold_c=threshold,
                                fallback_sentinel_min_count=sentinel_min,
                                fallback_mean_weight=fallback_weight,
                                fallback_std_mode=std_mode,  # type: ignore[arg-type]
                            )
                        )
    return configs


def _load_holdout(path: Path) -> pd.DataFrame:
    frame = sanitize_model_frame(pd.read_parquet(path))
    _, holdout = quick_eval_holdout(frame)
    return holdout.reset_index(drop=True)


def _save_wrapper(path: Path, *, primary: Any, fallback: Any, config: TailCalibrationConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = TailCalibratedGaussianModel(
        primary_model=primary,
        fallback_model=fallback,
        config=config,
        feature_names=list(getattr(primary, "feature_names", [])),
    )
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
    parser.add_argument("--baseline-guard-crps", type=float, default=0.85)
    parser.add_argument("--balanced-curated-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    primary = load_model(args.primary_model)
    fallback = load_model(args.fallback_model)
    curated_holdout = _load_holdout(args.curated_dataset)
    baseline_holdout = _load_holdout(args.baseline_dataset)
    curated_cache = _prediction_cache(curated_holdout, primary=primary, fallback=fallback)
    baseline_cache = _prediction_cache(baseline_holdout, primary=primary, fallback=fallback)

    raw_primary = TailCalibrationConfig(name="raw_primary")
    raw_fallback = TailCalibrationConfig(
        name="raw_fallback",
        fallback_std_threshold_c=999.0,
        fallback_sentinel_min_count=0,
        fallback_mean_weight=1.0,
        fallback_std_mode="fallback",
    )
    grid_rows: list[dict[str, Any]] = []
    for cfg in _candidate_grid():
        curated_metrics = _crps_for(curated_cache, cfg)
        baseline_metrics = _crps_for(baseline_cache, cfg)
        grid_rows.append(
            {
                "config": asdict(cfg),
                "curated": curated_metrics,
                "baseline": baseline_metrics,
            }
        )

    aggressive_row = min(grid_rows, key=lambda row: row["curated"]["crps_celsius_normalized"])
    guarded_rows = [
        row
        for row in grid_rows
        if row["baseline"]["crps_celsius_normalized"] <= args.baseline_guard_crps
    ]
    balanced_pool = guarded_rows or grid_rows
    balanced_best_curated = min(row["curated"]["crps_celsius_normalized"] for row in balanced_pool)
    balanced_near_best = [
        row
        for row in balanced_pool
        if row["curated"]["crps_celsius_normalized"] <= balanced_best_curated + args.balanced_curated_tolerance
    ]
    balanced_row = min(
        balanced_near_best,
        key=lambda row: (
            row["baseline"]["crps_celsius_normalized"],
            row["curated"]["crps_celsius_normalized"],
        ),
    )
    selected = {
        "tailcal_aggressive": TailCalibrationConfig(**{**aggressive_row["config"], "name": "tailcal_aggressive"}),
        "tailcal_balanced": TailCalibrationConfig(**{**balanced_row["config"], "name": "tailcal_balanced"}),
    }
    selected_paths = {
        name: args.output_dir / f"lgbm_emos__high_neighbor_oof_{name}.pkl"
        for name in selected
    }
    for name, cfg in selected.items():
        _save_wrapper(selected_paths[name], primary=primary, fallback=fallback, config=cfg)

    eval_models = {
        "raw_primary_multisource": args.primary_model,
        "raw_fallback_gfs": args.fallback_model,
        **selected_paths,
    }
    eval_datasets = {
        "curated_multisource_holdout": curated_holdout,
        "baseline_holdout": baseline_holdout,
    }
    results: dict[str, dict[str, dict[str, float] | None]] = {}
    for dataset_name, holdout in eval_datasets.items():
        results[dataset_name] = {}
        for model_name, model_path in eval_models.items():
            results[dataset_name][model_name] = evaluate_saved_model(Path(model_path), holdout)

    report = {
        "created_at": pd.Timestamp.now("UTC").isoformat(),
        "primary_model": str(args.primary_model),
        "fallback_model": str(args.fallback_model),
        "curated_dataset": str(args.curated_dataset),
        "baseline_dataset": str(args.baseline_dataset),
        "baseline_guard_crps": args.baseline_guard_crps,
        "balanced_curated_tolerance": args.balanced_curated_tolerance,
        "raw_crps_grid_basis": {
            "curated_primary": _crps_for(curated_cache, raw_primary),
            "curated_fallback": _crps_for(curated_cache, raw_fallback),
            "baseline_primary": _crps_for(baseline_cache, raw_primary),
            "baseline_fallback": _crps_for(baseline_cache, raw_fallback),
        },
        "selected": {
            name: {
                "path": str(selected_paths[name]),
                "config": asdict(cfg),
                "curated_grid_metrics": _crps_for(curated_cache, cfg),
                "baseline_grid_metrics": _crps_for(baseline_cache, cfg),
            }
            for name, cfg in selected.items()
        },
        "top_grid_by_curated_crps": sorted(
            grid_rows,
            key=lambda row: row["curated"]["crps_celsius_normalized"],
        )[:20],
        "top_grid_with_baseline_guard": sorted(
            guarded_rows,
            key=lambda row: row["curated"]["crps_celsius_normalized"],
        )[:20],
        "quick_eval_results": results,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({
        "report_path": str(args.report_path),
        "selected_paths": {name: str(path) for name, path in selected_paths.items()},
        "quick_eval_results": results,
    }, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
