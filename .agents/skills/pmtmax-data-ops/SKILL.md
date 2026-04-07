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
- use variant `--output-name` values by default; only pass `--allow-canonical-overwrite` for intentional canonical promotion.
- canonical overwrite now snapshots the old parquet + manifest under `artifacts/recovery/` first.
- lag recovery truth probes should default to `--truth-per-source-limit 1`, and use `--truth-no-cache` when cached truth payloads look stale or malformed.
- `historical_temperature_snapshots.json` is the canonical curated backlog for data collection. Use this when the task is “collect the latest historical markets”.
- `full_training_set_snapshots.json` is a checked-in training inventory, not an auto-refreshed mirror of the canonical historical backlog.
- When topping off an existing warehouse, prefer `backfill-forecasts --missing-only` so the run only fetches forecast request keys that are absent from `bronze_forecast_requests`.

## Common commands
- Latest curated historical collection:
  `uv run pmtmax backfill-forecasts --markets-path configs/market_inventory/historical_temperature_snapshots.json --strict-archive --missing-only`
- Historical collection with single-run horizons:
  `uv run pmtmax backfill-forecasts --markets-path configs/market_inventory/historical_temperature_snapshots.json --strict-archive --missing-only --single-run-horizon market_open --single-run-horizon previous_evening --single-run-horizon morning_of`
- Existing warehouse dataset rebuild without forecast refetch:
  `uv run pmtmax build-dataset --markets-path configs/market_inventory/full_training_set_snapshots.json --forecast-missing-only --allow-canonical-overwrite`

## Training Inventory Rule
- `full_training_set_snapshots.json` does not regenerate itself from `historical_temperature_snapshots.json`.
- If you need to refresh it, first rebuild and validate `historical_temperature_snapshots.json`, then explicitly curate and replace `full_training_set_snapshots.json` in the same change.
- Treat that refresh as an intentional checked-in inventory update, not as part of routine canonical backfill.
