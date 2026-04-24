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

## Paper trading — REAL CLOB ONLY (CRITICAL)
**NEVER record paper-trader output when book_source is missing or fixture-only** — those rows are diagnostics and must not pollute forward_paper_trades.json.

Before running paper-trader, ALWAYS verify real temperature markets exist:
```python
import httpx
r = httpx.get('https://gamma-api.polymarket.com/markets?active=true&limit=50', timeout=10)
data = r.json()
temp = [m for m in data if any(k in m.get('question','').lower() for k in ['temperature','highest','celsius','°c','°f'])]
print(f'Active temperature markets: {len(temp)}')  # Must be > 0
```
After paper-trader, check book_source in `artifacts/workspaces/ops_daily/signals/v2/paper_signals_latest.json`:
- ✅ ONLY record trades if `book_source` contains `"clob"`
- ❌ ABORT and discard if `book_source` is missing or fixture-only

**Temperature markets open daily** — real prices available ~KST 09:00. Run paper-trader after markets open.

`forward_paper_trades.json` was reset 2026-04-22 after non-CLOB priced trades were found.

## Critical rules
- Weather pretraining belongs in `weather_train` with `weather_real`; it must not include Polymarket ids, rules, price history, CLOB books, or publish evidence.
- `collect-weather-training` should now be treated as a monitored batch step:
  keep stderr progress enabled unless a wrapper needs quiet output, and prefer
  short HTTP bounds (`--http-timeout-seconds 15 --http-retries 1`) for free-tier
  Open-Meteo historical-forecast collection.
- The default orchestrator for repeated weather collection is the queue agent:
  `scripts/pmtmax-workspace weather_train uv run python scripts/run_weather_train_queue_agent.py`
  It advances the next `7`-day checker queue and auto-refreshes
  `gaussian_emos` pretrain once the row-gap threshold is reached.
- Latest observed collection state on April 24, 2026: `weather_train` gold has
  4,992 rows. Full successful ranges are `2024-01-03..2024-05-24` and
  `2026-01-01..2026-01-14`; partial coverage extends through `2024-05-30` and
  `2026-01-21`. `2026-01-22..2026-01-28` remained retry-only.
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
- before paper-trader records forward trades, verify active real CLOB books are available; missing/fixture paper trades are diagnostics and must not be recorded as forward evidence.
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
