# Historical Price Status

Updated: 2026-04-27 KST

## Current Snapshot
- workspace: `historical_real`
- dataset profile: `real_market`
- inventory: `configs/market_inventory/full_training_set_snapshots.json`
- inventory markets: `1,834`
- gold panel: `data/workspaces/historical_real/parquet/gold/historical_backtest_panel.parquet`
- coverage artifact: `artifacts/workspaces/historical_real/coverage/latest_price_history_coverage.json`
- related market/truth/forecast checker: `checker/historical_real_status.md`
- tracked token request rows for the latest checked-in inventory recovery agent snapshot: `14,674`
- tracked decision rows for the latest checked-in inventory recovery agent snapshot: `5,364`
- targeted Ankara coverage artifact: `artifacts/targeted_historical_refresh_20260426/ankara_price_coverage_summary.json`
- targeted Dallas/Atlanta/Miami coverage artifact: `artifacts/targeted_historical_refresh_20260426/dallas_atlanta_miami_price_coverage_summary.json`
- targeted Beijing/Chengdu/Chongqing/Madrid coverage artifact: `artifacts/targeted_historical_refresh_20260427/east_madrid_price_coverage_summary.json`

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
- Latest successful shard: `475..499 / 1834`; decision-ready delta `+0`.
- Dominant blocker: official `/prices-history` empties still outweigh recovered rows, and many empties show `last_trade_present`, so retention-limited history remains the main constraint.
- April 27 curated market/truth/forecast collection is now tracked in `checker/historical_real_status.md`: `2,169` curated snapshots are forecast/truth-ready, and the current non-canonical gold variant materializes `2,165` markets.

## Latest Targeted Price Backfill
- Scope: `artifacts/targeted_historical_refresh_20260427/east_madrid_snapshots.json` (`68` Beijing/Chengdu/Chongqing/Madrid markets).
- `backfill-price-history --only-missing --price-no-cache`: selected `68` markets, wrote `748` ok request rows and `36,030` official price points.
- Target panel:
  `data/workspaces/historical_real/parquet/gold/v2/targeted_east_madrid_backtest_panel_20260427.parquet`
  has `2,211` token rows with coverage `ok=1,868`, `missing=302`, `stale=41`.
- Full local backlog panel:
  `data/workspaces/historical_real/parquet/gold/v2/historical_backtest_panel_curated_multisource_targeted_east_madrid_20260427.parquet`
  has `54,261` token rows with coverage `ok=17,820`, `missing=36,349`, `stale=92`.
- This targeted run does not advance the checked-in inventory daily shard queue; continue from `500..524 / 1834` if running the price recovery agent.

## Previous Targeted Price Backfill
- Scope: `artifacts/targeted_historical_refresh_20260426/dallas_atlanta_miami_snapshots.json` (`60` Dallas/Atlanta/Miami markets).
- `backfill-price-history --only-missing --price-no-cache`: selected `60` markets, wrote `660` ok request rows and `35,365` official price points.
- Target panel:
  `data/workspaces/historical_real/parquet/gold/v2/targeted_dallas_atlanta_miami_backtest_panel_20260426.parquet`
  has `1,980` token rows with coverage `ok=1,941`, `missing=11`, `stale=28`.
- Full local backlog panel:
  `data/workspaces/historical_real/parquet/gold/v2/historical_backtest_panel_curated_multisource_targeted_south_20260426.parquet`
  has `52,050` token rows with coverage `ok=15,952`, `missing=36,047`, `stale=51`.
- This targeted run does not advance the checked-in inventory daily shard queue; continue from `500..524 / 1834` if running the price recovery agent.

## Earlier Targeted Price Backfill
- Scope: `artifacts/targeted_historical_refresh_20260426/ankara_snapshots.json` (`20` Ankara markets).
- `backfill-price-history --only-missing --price-no-cache`: selected `20` markets, wrote `220` ok request rows and `11,036` official price points.
- Target panel:
  `data/workspaces/historical_real/parquet/gold/v2/targeted_ankara_backtest_panel_20260426.parquet`
  has `660` token rows with coverage `ok=641`, `missing=14`, `stale=5`.
- Full local backlog panel:
  `data/workspaces/historical_real/parquet/gold/v2/historical_backtest_panel_curated_multisource_targeted_ankara_20260426.parquet`
  has `50,070` token rows with coverage `ok=14,011`, `missing=36,036`, `stale=23`.

## Next Recovery Queue
1. If expanding the training inventory, materialize a non-canonical variant from `historical_temperature_snapshots.json` first; do not conflate that with official price-history recovery.
2. If staying on the current checked-in inventory, continue missing-price recovery from shard `500..524` (`25` markets, dates `2025-12-02..2025-12-31`, cities `London`).
3. `weather_train` queue can run in parallel in a separate session; do not overlap another mutating `historical_real` job.
4. Re-run `real_history` evaluation after a meaningful panel-ready gain or after the training inventory is intentionally expanded.

## Daily Agent Command

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_historical_price_recovery_agent.py
```
