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

## Critical rules
- canonical `historical_training_set*` / `historical_backtest_panel` overwrite is opt-in only.
- use variant `--output-name` values by default; only pass `--allow-canonical-overwrite` for intentional canonical promotion.
- canonical overwrite now snapshots the old parquet + manifest under `artifacts/recovery/` first.
- lag recovery truth probes should default to `--truth-per-source-limit 1`, and use `--truth-no-cache` when cached truth payloads look stale or malformed.
