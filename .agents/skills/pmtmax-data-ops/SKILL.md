---
name: pmtmax-data-ops
description: Use when working on bootstrap-lab, warehouse management, seed export or restore, legacy raw/parquet cleanup, or canonical data layout in polymarket-tmax-lab.
---

# pmtmax-data-ops

Use this skill for storage, bootstrap, and long-history data operations.

## First reads
1. Read `AGENTS.md`.
2. Read `docs/codebase/storage-and-configs.md`.
3. Read:
   - `docs/agent-skills/data-ops.md`
   - `docs/agent-skills/safety-and-rules.md`

## Focus
- `weather_train` weather-real station/date pretrain data
- canonical warehouse paths
- `bootstrap-lab`, `export-seed`, `restore-seed`
- `inventory-legacy-runs`, `archive-legacy-runs`
- manifest, raw, parquet, and lock-file behavior
- canonical historical collection inventory:
  `configs/market_inventory/historical_temperature_snapshots.json`
- checked-in training inventory:
  `configs/market_inventory/full_training_set_snapshots.json`

## Critical rules
- canonical `historical_training_set*` / `historical_backtest_panel` overwrite is opt-in only.
- canonical data is real-only: synthetic inventories, `synthetic_` market ids, fixture forecasts, and fabricated books are trust violations.
- `weather_train` uses `PMTMAX_DATASET_PROFILE=weather_real` and must not contain Polymarket market ids, rule JSON, CLOB books, price history, or publish evidence.
- `historical_real` / benchmark / publish commands remain `PMTMAX_DATASET_PROFILE=real_market`.
- run `trust-check` before canonical overwrite, long collection, benchmark, or release validation.
- Open-Meteo weather-training collection now supports short HTTP controls:
  `--http-timeout-seconds`, `--http-retries`,
  `--http-retry-wait-min-seconds`, `--http-retry-wait-max-seconds`, plus
  stderr progress by default. Keep `--max-consecutive-429 2`: two consecutive
  Open-Meteo `429` responses mean the free-path daily limit is effectively hit.
- Historical `backfill-forecasts` also supports `--max-consecutive-429 2`.
  If two consecutive Open-Meteo `429` responses appear, the run flushes already
  collected rows, cancels, and should be recorded in checker state as
  rate-limit-cancelled; do not immediately retry the same free-path job.
- Prefer the queue agent for repeated older-gap weather collection:
  `scripts/pmtmax-workspace weather_train uv run python scripts/run_weather_train_queue_agent.py`
  It reads `checker/weather_train_status.md`, advances the next `7`-day chunk,
  updates checker markdown after every chunk, and auto-refreshes
  `gaussian_emos` pretrain when the configured row-gap threshold is met. When
  it sees two consecutive `429` responses, it cancels the remaining chunk,
  records `rate-limit-cancelled`, and waits for cooldown/API-key collection.
- Prefer the daily recovery agent for official price-history recovery:
  `scripts/pmtmax-workspace historical_real uv run python scripts/run_historical_price_recovery_agent.py`
  It reads `checker/historical_price_status.md`, advances the next shard,
  rebuilds the canonical backtest panel, refreshes the latest coverage JSON,
  and updates the historical-price checker markdown files.
- Latest observed field state on April 25, 2026:
  `weather_train` gold has 11,499 rows. Full successful ranges are
  `2024-01-03..2024-05-24`, `2024-06-01..2024-12-30`, and
  `2026-01-01..2026-01-14`; partial coverage includes
  `2024-12-31..2025-01-06` and `2026-01-15..2026-01-21`.
  `2025-01-07..2025-01-27` and `2026-01-22..2026-01-28` are retry-only/free-path
  daily-limit ranges. Read the live row count from `checker/weather_train_status.md`.
- Latest observed `historical_real` collection state on April 26, 2026:
  targeted Ankara plus Dallas/Atlanta/Miami expansion appended 80 event URLs,
  merged the local curated backlog to 2,101 snapshots, and materialized
  `historical_training_set_curated_multisource_targeted_south_20260426`
  with 6,294 rows / 2,098 markets. Target price-history coverage was strong:
  Ankara `641/660` token rows `ok`; Dallas/Atlanta/Miami `1,941/1,980` `ok`.
  Read live details from `checker/historical_real_status.md`.
- Keep `checker/weather_train_status.md` and
  `checker/weather_train_collection_log.md` synchronized after every
  collection/pretrain turn.
- Keep `checker/historical_price_status.md` and
  `checker/historical_price_collection_log.md` synchronized after every daily
  price recovery turn.
- Safe parallelism: `weather_train` queue collection can run at the same time
  as `historical_real` price-history collection because they write to separate
  workspace roots and hit different upstream APIs. Do not run two mutating
  `historical_real` jobs at once; `backfill-price-history`,
  `materialize-backtest-panel`, `build-dataset`, and `backfill-forecasts`
  should stay serialized within that workspace.
- use variant `--output-name` values by default; only pass `--allow-canonical-overwrite` for intentional canonical promotion.
- canonical overwrite now snapshots the old parquet + manifest under `artifacts/recovery/` first.
- lag recovery truth probes should default to `--truth-per-source-limit 1`, and use `--truth-no-cache` when cached truth payloads look stale or malformed.
- `historical_temperature_snapshots.json` is the canonical curated backlog for data collection. Use this when the task is “collect the latest historical markets”.
- `full_training_set_snapshots.json` is a checked-in training inventory, not an auto-refreshed mirror of the canonical historical backlog.
- latest audit semantics: `full_training_set_snapshots.json` has 1,834 trusted market snapshots, which materialize to 5,478 training rows across supported horizons; this is not a hard cap. The local curated collection backlog currently has 2,101 snapshots, but it is not automatically used as the checked-in training inventory.
- more real data can be collected, but it must be closed, parseable, truth-ready, forecast-backed, and intentionally curated before replacing the checked-in training inventory.
- daily `ops_daily` collection records forward evidence only; it does not auto-append to `full_training_set_snapshots.json` or retrain the champion.
- When topping off an existing warehouse, prefer `backfill-forecasts --missing-only` so the run only fetches forecast request keys that are absent from `bronze_forecast_requests`.
- For multi-source historical variants, pass `--model` explicitly on
  `backfill-forecasts`, `build-dataset`, or `materialize-training-set`.
  The base config can default to `gfs_seamless`; a curated rebuild without
  explicit models can keep the same row count but silently become GFS-only.
- Warehouse materialization treats all-zero target-day temperature aggregates as
  invalid forecast features, not real `0C` / `32F` forecasts. After changing
  this logic, rebuild a non-canonical variant and compare sentinel/source
  availability counts before training.

## Common commands
- Weather-real slow crawl with explicit HTTP bounds:
  `scripts/pmtmax-workspace weather_train uv run pmtmax collect-weather-training --date-from 2026-01-22 --date-to 2026-01-22 --model gfs_seamless --missing-only --rate-limit-profile free --http-timeout-seconds 15 --http-retries 1 --http-retry-wait-min-seconds 1 --http-retry-wait-max-seconds 8`
- Latest curated historical collection:
  `scripts/pmtmax-workspace historical_real uv run pmtmax backfill-forecasts --markets-path configs/market_inventory/historical_temperature_snapshots.json --strict-archive --missing-only`
- Historical collection with single-run horizons:
  `scripts/pmtmax-workspace historical_real uv run pmtmax backfill-forecasts --markets-path configs/market_inventory/historical_temperature_snapshots.json --model ecmwf_ifs025 --strict-archive --missing-only --single-run-horizon market_open --single-run-horizon previous_evening --single-run-horizon morning_of --max-consecutive-429 2`
- Existing warehouse multi-source variant materialization without forecast refetch:
  `scripts/pmtmax-workspace historical_real uv run pmtmax materialize-training-set --markets-path configs/market_inventory/historical_temperature_snapshots.json --model ecmwf_ifs025 --model ecmwf_aifs025_single --model kma_gdps --model gfs_seamless --output-name <variant>`
- Existing warehouse dataset rebuild without forecast refetch:
  `scripts/pmtmax-workspace historical_real uv run pmtmax build-dataset --markets-path configs/market_inventory/full_training_set_snapshots.json --forecast-missing-only --allow-canonical-overwrite`
- Official-price panel rebuild:
  `scripts/pmtmax-workspace historical_real uv run pmtmax backfill-price-history --markets-path configs/market_inventory/full_training_set_snapshots.json --only-missing --price-no-cache --limit-markets 25`
  `scripts/pmtmax-workspace historical_real uv run pmtmax materialize-backtest-panel --markets-path configs/market_inventory/full_training_set_snapshots.json --allow-canonical-overwrite`
- Price-history coverage recovery loop:
  `scripts/pmtmax-workspace historical_real uv run pmtmax summarize-price-history-coverage --markets-path configs/market_inventory/full_training_set_snapshots.json`
  `scripts/pmtmax-workspace historical_real uv run pmtmax backfill-price-history --markets-path configs/market_inventory/full_training_set_snapshots.json --only-missing --price-no-cache --limit-markets 25 --offset-markets 0`
  `scripts/pmtmax-workspace historical_real uv run pmtmax materialize-backtest-panel --markets-path configs/market_inventory/full_training_set_snapshots.json --allow-canonical-overwrite`
  `scripts/pmtmax-workspace historical_real uv run python scripts/run_historical_price_recovery_agent.py`
- Seed/bootstrap refresh without forecast refetch:
  `uv run pmtmax bootstrap-lab --forecast-missing-only`

## Training Inventory Rule
- `full_training_set_snapshots.json` does not regenerate itself from `historical_temperature_snapshots.json`.
- If you need to refresh it, first rebuild and validate `historical_temperature_snapshots.json`, then explicitly curate and replace `full_training_set_snapshots.json` in the same change.
- Treat that refresh as an intentional checked-in inventory update, not as part of routine canonical backfill.
- `restore-seed` rebuilds the local warehouse from canonical parquet mirrors directly in DuckDB; it should not need pandas full-table reads.
