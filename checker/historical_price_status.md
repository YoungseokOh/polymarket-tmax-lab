# Historical Price Status

Updated: 2026-04-24 KST

## Current Snapshot
- workspace: `historical_real`
- dataset profile: `real_market`
- inventory: `configs/market_inventory/full_training_set_snapshots.json`
- inventory markets: `1,834`
- gold panel: `data/workspaces/historical_real/parquet/gold/historical_backtest_panel.parquet`
- coverage artifact: `artifacts/workspaces/historical_real/coverage/latest_price_history_coverage.json`
- tracked token request rows: `14,674`
- tracked decision rows: `5,364`

## Request Coverage
- status `ok`: `5,213`
- status `empty`: `9,461`
- status `http_error`: `0`
- status `error`: `0`
- `empty` with `last_trade_present=true`: `9,461` / `9,461`
- last-trade probes recorded: `9,461`

## Panel Readiness
- token coverage `ok`: `11,886`
- token coverage `missing`: `30,618`
- token coverage `stale`: `0`
- panel-ready decision rows: `1,207` / `5,364` (`22.5%`)
- latest backtest `priced_decision_rows`: `1,155`
- latest backtest `PnL`: `1029.24`
- latest backtest metrics artifact: `artifacts/workspaces/historical_real/backtests/v2/backtest_metrics_real_history.json`
- latest backtest artifact updated at: `2026-04-24T01:27:31.623783+09:00`

## Current Judgment
- Daily price agent runs one shard at a time (`25` markets default) and keeps `backfill-price-history -> materialize-backtest-panel -> summarize-price-history-coverage` serialized.
- Latest successful shard: `0..24 / 1834`; decision-ready delta `+0`.
- Dominant blocker: official `/prices-history` empties still outweigh recovered rows, and many empties show `last_trade_present`, so retention-limited history remains the main constraint.

## Next Recovery Queue
1. Continue missing-price recovery from shard `25..49` (`25` markets, dates `2026-01-22..2026-03-19`, cities `Ankara`).
2. `weather_train` queue can run in parallel in a separate session; do not overlap another mutating `historical_real` job.
3. Re-run `real_history` evaluation after a meaningful panel-ready gain or after one full shard cycle completes.

## Daily Agent Command

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_historical_price_recovery_agent.py
```
