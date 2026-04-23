---
name: pmtmax-autoresearch
description: Use when running the LGBM autoresearch loop, editing candidate YAML specs, gating new variants, or promoting winners in polymarket-tmax-lab.
---

# pmtmax-autoresearch

Use this skill for the agent-driven autoresearch loop around `lgbm_emos`.

## First reads
1. Read `AGENTS.md`.
2. Read:
   - `docs/agent-skills/autoresearch.md`
   - `docs/agent-skills/research-loop.md`
   - `docs/codebase/modeling.md`

## Critical rules
- current public champion variant is `high_neighbor_oof`; new champion-adjacent runs should pass `--baseline-variant high_neighbor_oof`.
- existing run manifests are authoritative; do not silently rewrite older `recency_neighbor_oof` runs.
- create or edit one candidate YAML at a time under `artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/`.
- run autoresearch against real-only `historical_real` data; synthetic augmentation is not allowed.
- use `autoresearch-step` for quick keep/discard/crash, then `autoresearch-gate`, then `autoresearch-analyze-paper`.
- canonical `historical_training_set*` / `historical_backtest_panel` stay immutable unless explicitly promoted elsewhere.
- public `champion` publish is never implicit; `autoresearch-promote` only promotes YAML specs and `publish-champion` is the separate public-alias command.
- promotion is fail-closed: require CLI gate leaderboard JSON/CSV, matching dataset/panel signatures, candidate calibrator, and paper `overall_gate_decision=GO`; `INCONCLUSIVE` blocks promotion.
- report CRPS, DirAcc, and ECE together; do not promote from CRPS alone.
- quick-eval CRPS is Celsius-normalized in code and also reports raw market-unit CRPS for audit.

## CRPS unit normalization
The training dataset (`data/workspaces/<workspace>/parquet/gold/v2/historical_training_set.parquet`
or the active `PMTMAX_PARQUET_DIR`) contains **mixed temperature units**:
- Celsius (C): Seoul, Tokyo, London, etc.
- Fahrenheit (F): NYC, Chicago, Atlanta, Dallas, Miami, Seattle, etc.

Raw CRPS on mixed units is not comparable. `evaluate_saved_model` and `scripts/quick_eval.py`
now report `crps_celsius_normalized` and `crps_market_unit`; use the normalized value for
model comparisons and keep the raw value for audit.

## Evaluation must include CRPS + DirAcc + ECE — all three
Never report CRPS alone. Always include:
- **CRPS (°C-normalized)**: unit-normalized predictive accuracy
- **DirAcc**: directional accuracy — fraction of markets where top-predicted bin = actual winner. Target: >73%
- **ECE**: Expected Calibration Error — lower is better, well-calibrated model ~0.007

Use `evaluate_saved_model(model_path, holdout)` and report all three side-by-side for every candidate comparison.

## Focus
- run scaffold: `artifacts/workspaces/historical_real/autoresearch/<run_tag>/`
- promoted spec registry: `configs/autoresearch/lgbm_emos/promoted/`
- candidate artifacts: `artifacts/workspaces/historical_real/autoresearch/<run_tag>/models/`
- shared program: `artifacts/workspaces/historical_real/autoresearch/<run_tag>/program.md`
