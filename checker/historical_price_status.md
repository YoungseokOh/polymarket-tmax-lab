# Historical Price Status

Updated: 2026-04-27 KST

## Current Snapshot
- workspace: `historical_real`
- dataset profile: `real_market`
- inventory: `configs/market_inventory/full_training_set_snapshots.json`
- inventory markets: `2,602`
- canonical v2 gold panel: `data/workspaces/historical_real/parquet/gold/v2/historical_backtest_panel.parquet`
- latest coverage artifact: `artifacts/price_history_coverage.json`
- related market/truth/forecast checker: `checker/historical_real_status.md`
- tracked token request rows for the promoted inventory coverage snapshot: `21,441`
- tracked decision rows for the promoted inventory coverage snapshot: `7,794`
- targeted Ankara coverage artifact: `artifacts/targeted_historical_refresh_20260426/ankara_price_coverage_summary.json`
- targeted Dallas/Atlanta/Miami coverage artifact: `artifacts/targeted_historical_refresh_20260426/dallas_atlanta_miami_price_coverage_summary.json`
- targeted Beijing/Chengdu/Chongqing/Madrid coverage artifact: `artifacts/targeted_historical_refresh_20260427/east_madrid_price_coverage_summary.json`
- targeted discovery120 coverage artifact: `artifacts/targeted_historical_refresh_20260427/discovery120_price_coverage_summary.json`

## Request Coverage
- status `ok`: `11,651`
- status `empty`: `9,790`
- status `http_error`: `0`
- status `error`: `0`
- `empty` with `last_trade_present=true`: tracked in `artifacts/price_history_coverage.json`
- last-trade probes recorded: tracked in `artifacts/price_history_coverage.json`

## Panel Readiness
- token coverage `ok`: `30,615`
- token coverage `missing`: `37,501`
- token coverage `stale`: `434`
- panel-ready decision rows: `2,706` / `7,794` (`34.7%`)
- latest backtest `priced_decision_rows`: `1,155`
- latest backtest `PnL`: `1029.24`
- latest backtest metrics artifact: `artifacts/workspaces/historical_real/backtests/v2/backtest_metrics_real_history.json`
- latest backtest artifact updated at: `2026-04-24T01:27:31.623783+09:00`

## Current Judgment
- Daily price agent runs one shard at a time (`25` markets default) and keeps `backfill-price-history -> materialize-backtest-panel -> summarize-price-history-coverage` serialized.
- Latest successful recovery-agent shard was `475..499 / 1834` before inventory promotion. After promotion, the checked-in inventory denominator is `2,602`, so the next recovery turn should refresh shard accounting against the promoted inventory before comparing deltas.
- Dominant blocker: official `/prices-history` empties still outweigh recovered rows, and many empties show `last_trade_present`, so retention-limited history remains the main constraint.
- April 27 promotion is tracked in `checker/historical_real_status.md`: `2,602` snapshots are now the checked-in training inventory, canonical v2 training materializes `2,598` markets, and canonical v2 panel has `68,550` token rows.

## Latest Canonical Promotion Panel
- Scope: promoted `configs/market_inventory/full_training_set_snapshots.json` (`2,602` snapshots).
- Training set:
  `data/workspaces/historical_real/parquet/gold/v2/historical_training_set.parquet`
  has `7,794` rows / `2,598` markets.
- Backtest panel:
  `data/workspaces/historical_real/parquet/gold/v2/historical_backtest_panel.parquet`
  has `68,550` token rows with coverage `ok=30,615`, `missing=37,501`, `stale=434`.
- Coverage summary:
  `artifacts/price_history_coverage.json` has `21,441` request rows and `68,550` panel detail rows.

## Latest Targeted Price Backfill
- Scope: `artifacts/targeted_historical_refresh_20260427/discovery120_snapshots.json` (`1` Tokyo market).
- `backfill-price-history --only-missing --price-no-cache`: selected `1` market, wrote `11` ok request rows and `565` official price points.
- Target panel:
  `data/workspaces/historical_real/parquet/gold/v2/targeted_discovery120_backtest_panel_20260427.parquet`
  has `33` token rows with coverage `ok=22`, `missing=11`, `stale=0`.
- Full local backlog panel:
  `data/workspaces/historical_real/parquet/gold/v2/historical_backtest_panel_curated_multisource_discovery120_20260427.parquet`
  has `68,550` token rows with coverage `ok=30,615`, `missing=37,501`, `stale=434`.
- This targeted run does not advance the checked-in inventory daily shard queue; continue from `500..524 / 1834` if running the price recovery agent.

## Previous Targeted Price Backfill
- Scope: `artifacts/targeted_historical_refresh_20260427/east_madrid_snapshots.json` (`68` Beijing/Chengdu/Chongqing/Madrid markets).
- `backfill-price-history --only-missing --price-no-cache`: selected `68` markets, wrote `748` ok request rows and `36,030` official price points.
- Target panel:
  `data/workspaces/historical_real/parquet/gold/v2/targeted_east_madrid_backtest_panel_20260427.parquet`
  has `2,211` token rows with coverage `ok=1,868`, `missing=302`, `stale=41`.
- Full local backlog panel:
  `data/workspaces/historical_real/parquet/gold/v2/historical_backtest_panel_curated_multisource_targeted_east_madrid_20260427.parquet`
  has `54,261` token rows with coverage `ok=17,820`, `missing=36,349`, `stale=92`.
- This targeted run does not advance the checked-in inventory daily shard queue; continue from `500..524 / 1834` if running the price recovery agent.

## Earlier Targeted Price Backfill
- Scope: `artifacts/targeted_historical_refresh_20260426/dallas_atlanta_miami_snapshots.json` (`60` Dallas/Atlanta/Miami markets).
- `backfill-price-history --only-missing --price-no-cache`: selected `60` markets, wrote `660` ok request rows and `35,365` official price points.
- Target panel:
  `data/workspaces/historical_real/parquet/gold/v2/targeted_dallas_atlanta_miami_backtest_panel_20260426.parquet`
  has `1,980` token rows with coverage `ok=1,941`, `missing=11`, `stale=28`.
- Full local backlog panel:
  `data/workspaces/historical_real/parquet/gold/v2/historical_backtest_panel_curated_multisource_targeted_south_20260426.parquet`
  has `52,050` token rows with coverage `ok=15,952`, `missing=36,047`, `stale=51`.
- This targeted run does not advance the checked-in inventory daily shard queue; continue from `500..524 / 1834` if running the price recovery agent.

## Oldest Targeted Price Backfill
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
