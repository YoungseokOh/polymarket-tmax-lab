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
- cron-based 2-hour price checks MUST call `scripts/run_price_check.sh` and redirect to an absolute `logs/price_check.log` path. Do not put raw `uv run python scripts/log_gamma_prices.py` commands directly in crontab.
- Model training: `train-advanced --model-name lgbm_emos --variant <variant>`.
- Quick eval: `uv run python scripts/quick_eval.py` (champion baseline + OOF variants).
- observation weather-station loop: `observation-report`, `observation-shadow`, `observation-daemon`, `approve-live-candidate`.
- zero-fill paper diagnostics: `paper-multimodel-report`, `execution-sensitivity-report`, `market-bottleneck-report`.
- zero-fill playbook promotion: `execution-watchlist-playbook`.
- observation source priority: `exact_public intraday -> documented research intraday -> METAR fallback`, target-day only.
- station dashboard loop: `station-dashboard`, `station-dashboard-daemon`.
- station dashboard consumes `artifacts/signals/v2/execution_watchlist_playbook.json` when present and raises Tier A ask-threshold alerts without changing live guardrails.
- station orchestrator loop: `station-cycle`, `station-daemon`.
- revenue gate / station cycle default benchmark input: `artifacts/benchmarks/v2/benchmark_summary.json`
- Champion is `recency_neighbor_oof` (CRPS 0.7463 honest, MAE 0.591, σ calibrated 2–5°).
- Previous fast variants (ultra_high_neighbor_fast 등) had σ=0.5 collapse — scale clip floor raised to 2.0.
- Use `pmtmax-autoresearch` when you are exploring new `lgbm_emos` candidates around `recency_neighbor_oof`.

## Focus
- gold dataset: `data/parquet/gold/v2/historical_training_set.parquet`
- model artifacts: `artifacts/models/v2/`
- champion alias: `artifacts/models/v2/champion.json`
- daily signals: `artifacts/signals/v2/scan_edge_latest.json`
- observation queue: `artifacts/signals/v2/live_pilot_queue.json`
- paper trades: `artifacts/signals/v2/forward_paper_trades.json`
- cron log: `logs/daily_experiment.log`
- 2-hour cron wrapper: `scripts/run_price_check.sh`
- 2-hour price check log: `logs/price_check.log`
- paper exploration preset: `configs/paper-exploration.yaml`
- paper all-supported horizon policy: `configs/paper-all-supported-horizon-policy.yaml`
- execution watchlist playbook: `artifacts/signals/v2/execution_watchlist_playbook.json`
