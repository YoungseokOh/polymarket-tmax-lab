# Checker

Agent-maintained operational markdowns for recurring dataset checks and
collection checkpoints.

Use this folder when a turn needs to continue an in-flight data collection
without rediscovering the whole context.

## Files
- `weather_train_status.md`: current authoritative snapshot for the weather-real
  pretrain dataset, including coverage, blockers, artifacts, and next targets.
- `weather_train_collection_log.md`: append-only collection history with ranges,
  outcomes, and row deltas.
- `weather_train_runbook.md`: step-by-step commands and decision rules for
  continuing weather collection and training.

## Update Rules
1. Read `weather_train_status.md` before starting a new collection run.
2. After every collection attempt:
   - update `weather_train_status.md`
   - append a new entry to `weather_train_collection_log.md`
3. If a new pretrain artifact is produced, update the artifact path and dataset
   signature in `weather_train_status.md`.
4. Keep the markdown factual. Do not speculate about rows or coverage that have
   not been verified from parquet or command output.
