#!/usr/bin/env python3
"""Run the historical_real model research agent and update checker state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pmtmax.modeling.research_agent import (
    DEFAULT_AUTORESEARCH_ROOT,
    DEFAULT_BASELINE_VARIANT,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_ARTIFACTS_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_PANEL_PATH,
    DEFAULT_WEATHER_PRETRAIN_PATH,
    run_model_research_agent,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-path", type=Path, default=Path("checker/model_research_status.md"))
    parser.add_argument("--log-path", type=Path, default=Path("checker/model_research_log.md"))
    parser.add_argument(
        "--markets-path",
        type=Path,
        default=Path("configs/market_inventory/full_training_set_snapshots.json"),
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--baseline-variant", default=DEFAULT_BASELINE_VARIANT)
    parser.add_argument("--model-artifacts-dir", type=Path, default=DEFAULT_MODEL_ARTIFACTS_DIR)
    parser.add_argument("--autoresearch-root", type=Path, default=DEFAULT_AUTORESEARCH_ROOT)
    parser.add_argument("--pretrained-weather-model", type=Path, default=DEFAULT_WEATHER_PRETRAIN_PATH)
    parser.add_argument("--auto-candidate-limit", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=1)
    parser.add_argument("--enable-training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-paper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-promote", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-publish", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recent-core-summary-path", type=Path, default=None)
    parser.add_argument("--force-baseline-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-tag", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_model_research_agent(
        status_path=args.status_path,
        log_path=args.log_path,
        markets_path=args.markets_path,
        dataset_path=args.dataset_path,
        panel_path=args.panel_path,
        model_name=args.model_name,
        baseline_variant=args.baseline_variant,
        model_artifacts_dir=args.model_artifacts_dir,
        autoresearch_root=args.autoresearch_root,
        pretrained_weather_model=args.pretrained_weather_model,
        auto_candidate_limit=args.auto_candidate_limit,
        max_candidates=args.max_candidates,
        enable_training=bool(args.enable_training),
        enable_gate=bool(args.enable_gate),
        enable_paper=bool(args.enable_paper),
        enable_promote=bool(args.enable_promote),
        enable_publish=bool(args.enable_publish),
        recent_core_summary_path=args.recent_core_summary_path,
        force_baseline_train=bool(args.force_baseline_train),
        run_tag_override=args.run_tag,
    )
    print(json.dumps(summary.__dict__, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
