---
name: pmtmax-research-loop
description: Use when working on dataset materialization, model training, backtesting, scan-edge signals, paper-trading workflows, or experiment artifacts in polymarket-tmax-lab.
---

# pmtmax-research-loop

Use this skill for the research and trading simulation loop.

## First reads
1. Read `AGENTS.md`.
2. Read:
   - `docs/agent-skills/research-loop.md`
   - `docs/codebase/modeling.md`
   - `docs/codebase/backtest-execution.md`

## Critical rules
- `build-dataset` MUST use `--markets-path configs/market_inventory/full_training_set_snapshots.json`.
  Running without it rebuilds with only 12 example rows (destroys training data).
- canonical `historical_training_set*` / `historical_backtest_panel` overwrite requires `--allow-canonical-overwrite`.
- overwrite is promotion-only; use variant `--output-name` values for experiments and rely on the automatic `artifacts/recovery/` backup when promoting canonical output.
- `scan-edge` MUST include `--min-model-prob 0.05 --max-model-prob 0.95`.
- Model training: `train-advanced --model-name lgbm_emos --variant <variant>`.
- Quick eval: `uv run python scripts/quick_eval.py` (top 5 fast variants only).
- Champion is `recency_neighbor_fast` (CRPS 0.4769).

## Focus
- gold dataset: `data/parquet/gold/v2/historical_training_set.parquet`
- model artifacts: `artifacts/models/v2/`
- champion alias: `artifacts/models/v2/champion.json`
- daily signals: `artifacts/signals/v2/scan_edge_latest.json`
- paper trades: `artifacts/signals/v2/forward_paper_trades.json`
- cron log: `logs/daily_experiment.log`
