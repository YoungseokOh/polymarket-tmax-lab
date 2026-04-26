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

## Paper trading — real CLOB only
- Never record `paper-trader` output as forward evidence when `book_source` is missing or fixture-only.
- After `paper-trader`, inspect `artifacts/workspaces/ops_daily/signals/v2/paper_signals_latest.json`; only CLOB-backed rows are evidence.
- Daily temperature markets usually have real prices after about 09:00 KST, so the daily cron should run after that on KST hosts.

## Critical rules
- Weather pretraining belongs in `weather_train` with `weather_real`; it must not include Polymarket ids, rules, price history, CLOB books, or publish evidence.
- `collect-weather-training` should now be treated as a monitored batch step:
  keep stderr progress enabled unless a wrapper needs quiet output, and prefer
  short HTTP bounds (`--http-timeout-seconds 15 --http-retries 1`) for free-tier
  Open-Meteo historical-forecast collection. Keep `--max-consecutive-429 2`;
  two consecutive Open-Meteo `429` responses are a daily-limit hit, so cancel
  the remaining chunk and record `rate-limit-cancelled` in checker state.
- The default orchestrator for repeated weather collection is the queue agent:
  `scripts/pmtmax-workspace weather_train uv run python scripts/run_weather_train_queue_agent.py`
  It advances the next `7`-day checker queue and auto-refreshes
  `gaussian_emos` pretrain once the row-gap threshold is reached. It defaults
  to the two-consecutive-`429` cancel rule.
- Latest observed collection state on April 25, 2026: `weather_train` gold has
  11,499 rows. Full successful ranges are `2024-01-03..2024-05-24`,
  `2024-06-01..2024-12-30`, and `2026-01-01..2026-01-14`; retry-only/free-path
  daily-limit ranges include `2025-01-07..2025-01-27` and
  `2026-01-22..2026-01-28`. Read live details from
  `checker/weather_train_status.md`.
- Operational continuity for weather collection lives under `checker/`; update
  `checker/weather_train_status.md` and `checker/weather_train_collection_log.md`
  after every run.
- Daily continuity for official price-history recovery also lives under
  `checker/`; update `checker/historical_price_status.md` and
  `checker/historical_price_collection_log.md` after every recovery run.
- Parallel research ops rule: it is acceptable to run `weather_train` queue
  collection in parallel with `historical_real backfill-price-history` because
  those jobs use separate workspace roots. Do not overlap multiple mutating
  `historical_real` jobs; finish `backfill-price-history` before
  `materialize-backtest-panel`, and finish panel rebuild before benchmark or
  publish gates.
- The default daily orchestrator for official price-history recovery is:
  `scripts/pmtmax-workspace historical_real uv run python scripts/run_historical_price_recovery_agent.py`
  It advances one checker shard by default and records the latest panel-ready
  decision rows plus the latest backtest `priced_decision_rows` anchor.
- The default checker-driven orchestrator for baseline retraining plus
  autoresearch is:
  `scripts/pmtmax-workspace historical_real uv run python scripts/run_model_research_agent.py`
  It keeps one active autoresearch run aligned to the current dataset/panel
  signatures, auto-creates the next small candidate YAML when needed, and
  tracks train/step/gate/paper/promote state under `checker/model_research_*`.
- Market calibration, backtest, benchmark, and champion publish remain `historical_real` / `real_market` only.
- For curated historical variants, pass repeated `--model` values when
  materializing from the warehouse. The base config can default to GFS-only;
  compare source feature counts before training so a row-count increase is not
  mistaken for a multi-source dataset.
- before paper-trader records forward trades, verify active real CLOB books are available; missing/fixture paper trades are diagnostics and must not be recorded as forward evidence.
- current paper commands reject legacy `book_source=fixture`; missing CLOB books must be `missing_book`, not tradable signals.
- synthetic inventories, `synthetic_` market ids, fixture forecasts, and fabricated books are trust violations in canonical research.
- `build-dataset` MUST use `--markets-path configs/market_inventory/full_training_set_snapshots.json`.
  Running without it rebuilds with only 12 example rows (destroys training data).
- `full_training_set_snapshots.json` is a checked-in training inventory. It is not the auto-refreshed canonical historical backlog.
- If you need the latest collected historical markets, refresh `configs/market_inventory/historical_temperature_snapshots.json` first and then intentionally regenerate `full_training_set_snapshots.json` from that curated inventory.
- canonical `historical_training_set*` / `historical_backtest_panel` overwrite requires `--allow-canonical-overwrite`.
- overwrite is promotion-only; use variant `--output-name` values for experiments and rely on the automatic `artifacts/recovery/` backup when promoting canonical output.
- `scan-edge` MUST include `--min-model-prob 0.05 --max-model-prob 0.95`.
- cron-based daily collection on this KST host MUST use `0 9 * * *` for `scripts/daily_experiment.sh`; `0 0 * * *` is only the equivalent on a UTC host.
- cron-based 2-hour price checks MUST call `scripts/run_price_check.sh` and redirect to an absolute `logs/price_check.log` path. Do not put raw `uv run python scripts/log_gamma_prices.py` commands directly in crontab.
- Model training: `train-advanced --model-name lgbm_emos --variant <variant>`.
  Passing `--pretrained-weather-model` injects `weather_pretrain_*` features at
  fit/predict time, not just sidecar lineage. The first `high_neighbor_oof`
  quick eval with active injection worsened `CRPS_C` from `0.8004` to `0.8241`,
  and the `delta_only` follow-up still worsened to `0.8111`; do not promote
  weather-pretrain feature candidates without a stronger candidate-specific gate.
- April 25, 2026 curated multi-source check: `historical_training_set_curated_multisource_20260425`
  restored all four NWP source feature groups and improved canonical baseline
  holdout CRPS_C (`0.8004` -> `0.6749`), but on the curated holdout it worsened
  CRPS_C versus GFS-only (`2.0559` -> `2.8771`) despite better Brier/DirAcc due
  to under-dispersed, cold-biased tails. Do not promote before variance/tail
  calibration.
- April 26, 2026 tail calibration check: `scripts/run_tail_calibration_experiment.py`
  produced diagnostic wrappers under
  `artifacts/workspaces/historical_real/models/curated_multisource_tailcal_20260426/`.
  Balanced tailcal reduced curated holdout CRPS_C to `1.7195` and kept baseline
  holdout near canonical baseline at `0.8043`, but the root signal was
  zero-Celsius NWP daily-max sentinels (`32F` / `0C`) marked available. Treat
  this as diagnostic; fix feature validity upstream before promotion.
- April 26, 2026 sentinel-fix follow-up:
  `historical_training_set_curated_multisource_sentinelfix_20260426` removed
  all-source zero-C sentinel rows (`1083 -> 0`) and retrained
  `lgbm_emos/high_neighbor_oof`, but curated holdout CRPS_C remained worse than
  same-holdout GFS-only (`2.7690` vs `2.1103`).
- April 26, 2026 source-gating follow-up:
  `scripts/run_source_gating_experiment.py` trained binary/regret-weighted
  train-side gates between sentinel-fix multi-source and GFS-only. Best curated
  holdout CRPS_C was only `2.7607` vs raw primary `2.7690` and GFS-only
  `2.1103`; baseline holdout stayed safe (`0.4932` best vs raw `0.4965`).
  Gate fallback use stayed near `2%` on curated holdout, so this is diagnostic
  only. Next experiment should target source-disagreement/region-driven variance
  calibration or proper OOF stacking, not simple binary fallback gating.
- April 26, 2026 disagreement calibration follow-up:
  `scripts/run_disagreement_calibration_experiment.py` showed variance-only
  source-disagreement calibration improves curated holdout CRPS_C to `2.2342`
  but still loses to GFS-only `2.1103`. Positive primary-vs-GFS disagreement
  mean shrink plus variance calibration improves curated holdout CRPS_C to
  `1.3153` with Brier `0.0748`, but worsens baseline holdout CRPS_C to
  `0.5627` from raw `0.4965`. Treat as the strongest diagnostic wrapper so far,
  not promotion. Next serious candidate should be a non-holdout-tuned LGBM
  disagreement feature variant or proper OOF stacker.
- Quick eval: `uv run python scripts/quick_eval.py` (champion baseline + OOF variants).
- observation weather-station loop: `observation-report`, `observation-shadow`, `observation-daemon`, `approve-live-candidate`.
- zero-fill paper diagnostics: `paper-multimodel-report`, `execution-sensitivity-report`, `market-bottleneck-report`.
- zero-fill playbook promotion: `execution-watchlist-playbook`.
- observation source priority: `exact_public intraday -> documented research intraday -> METAR fallback`, target-day only.
- station dashboard loop: `station-dashboard`, `station-dashboard-daemon`.
- station dashboard consumes `artifacts/workspaces/ops_daily/signals/v2/execution_watchlist_playbook.json` when present and raises Tier A ask-threshold alerts without changing live guardrails.
- station orchestrator loop: `station-cycle`, `station-daemon`.
- revenue gate / station cycle default benchmark input: `artifacts/workspaces/ops_daily/benchmarks/v2/benchmark_summary.json`
- Public champion alias lives at `artifacts/public_models/champion.json`.
- Current public champion is `lgbm_emos / high_neighbor_oof`.
- Previous fast variants (ultra_high_neighbor_fast 등) had σ=0.5 collapse — scale clip floor raised to 2.0.
- Use `pmtmax-autoresearch` when you are exploring new `lgbm_emos` candidates around the current champion or the run manifest baseline.
- Public champion aliases missing recent-core `publish_gate.decision=GO` are invalid; republish with `publish-champion` after a real recent-core GO instead of using or repairing them manually.

## Common commands
- Weather-real pretrain:
  `scripts/pmtmax-workspace weather_train uv run pmtmax collect-weather-training --city Seoul --date-from 2024-01-01 --date-to 2024-01-07 --http-timeout-seconds 15 --http-retries 1 --missing-only`
  `scripts/pmtmax-workspace weather_train uv run pmtmax train-weather-pretrain --model-name gaussian_emos`
- Canonical training build with existing forecast coverage reuse:
  `scripts/pmtmax-workspace historical_real uv run pmtmax build-dataset --markets-path configs/market_inventory/full_training_set_snapshots.json --forecast-missing-only --allow-canonical-overwrite`
- Official-price panel rebuild:
  `scripts/pmtmax-workspace historical_real uv run pmtmax materialize-backtest-panel --markets-path configs/market_inventory/full_training_set_snapshots.json --allow-canonical-overwrite`
- Price-history recovery before backtest:
  `scripts/pmtmax-workspace historical_real uv run pmtmax summarize-price-history-coverage --markets-path configs/market_inventory/full_training_set_snapshots.json`
  `scripts/pmtmax-workspace historical_real uv run pmtmax backfill-price-history --markets-path configs/market_inventory/full_training_set_snapshots.json --only-missing --price-no-cache --limit-markets 25`
  `scripts/pmtmax-workspace historical_real uv run pmtmax materialize-backtest-panel --markets-path configs/market_inventory/full_training_set_snapshots.json --allow-canonical-overwrite`
- Bootstrap refresh with existing forecast coverage reuse:
  `uv run pmtmax bootstrap-lab --forecast-missing-only`
- Canonical historical forecast top-off before rebuild:
  `scripts/pmtmax-workspace historical_real uv run pmtmax backfill-forecasts --markets-path configs/market_inventory/historical_temperature_snapshots.json --strict-archive --missing-only`
- If single-run horizons are part of the rebuild:
  `scripts/pmtmax-workspace historical_real uv run pmtmax backfill-forecasts --markets-path configs/market_inventory/historical_temperature_snapshots.json --strict-archive --missing-only --single-run-horizon market_open --single-run-horizon previous_evening --single-run-horizon morning_of`
- Curated multi-source materialization without canonical overwrite:
  `scripts/pmtmax-workspace historical_real uv run pmtmax materialize-training-set --markets-path configs/market_inventory/historical_temperature_snapshots.json --model ecmwf_ifs025 --model ecmwf_aifs025_single --model kma_gdps --model gfs_seamless --output-name <variant>`
- Curated multi-source tail calibration diagnostic:
  `scripts/pmtmax-workspace historical_real uv run python scripts/run_tail_calibration_experiment.py`
- Curated multi-source source-gating diagnostic:
  `scripts/pmtmax-workspace historical_real uv run python scripts/run_source_gating_experiment.py`
- Curated multi-source disagreement calibration diagnostic:
  `scripts/pmtmax-workspace historical_real uv run python scripts/run_disagreement_calibration_experiment.py`
- Recent-core LGBM promotion gate:
  `scripts/run_recent_core_benchmark_local.sh --model-name lgbm_emos --variant <variant> --retrain-stride 1`
- Recent-core gate from trusted historical-real parquet:
  `scripts/pmtmax-workspace recent_core_eval uv run python scripts/run_recent_core_benchmark.py --model-name lgbm_emos --variant <variant> --retrain-stride 10 --backtest-last-n 60 --prebuilt-dataset-path data/workspaces/historical_real/parquet/gold/historical_training_set.parquet --prebuilt-panel-path data/workspaces/historical_real/parquet/gold/historical_backtest_panel.parquet --prebuilt-last-n-market-days 90`

## Focus
- historical-real dataset: `data/workspaces/historical_real/parquet/gold/historical_training_set.parquet`
- workspace-local model artifacts: `artifacts/workspaces/<workspace>/models/v2/`
- public champion alias: `artifacts/public_models/champion.json`
- daily signals: `artifacts/workspaces/ops_daily/signals/v2/scan_edge_latest.json`
- observation queue: `artifacts/workspaces/ops_daily/signals/v2/live_pilot_queue.json`
- paper trades: `artifacts/workspaces/ops_daily/signals/v2/forward_paper_trades.json`
- cron log: `logs/daily_experiment.log`
- 2-hour cron wrapper: `scripts/run_price_check.sh`
- 2-hour price check log: `logs/price_check.log`
- paper exploration preset: `configs/paper-exploration.yaml`
- paper all-supported horizon policy: `configs/paper-all-supported-horizon-policy.yaml`
- execution watchlist playbook: `artifacts/workspaces/ops_daily/signals/v2/execution_watchlist_playbook.json`
