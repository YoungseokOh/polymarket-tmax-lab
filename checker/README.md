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
- `historical_price_status.md`: current authoritative snapshot for
  `historical_real` official price-history recovery, including token coverage,
  panel-ready decision rows, blockers, and the next shard queue.
- `historical_price_collection_log.md`: append-only recovery history for
  official Polymarket price-history shards and panel-readiness deltas.
- `historical_price_runbook.md`: step-by-step commands and decision rules for
  continuing daily price-history recovery.
- `historical_real_status.md`: current authoritative snapshot for curated
  `historical_real` market/truth/forecast collection, readiness, and next
  materialization strategy.
- `historical_real_collection_log.md`: append-only collection history for
  curated historical market inventories, truth backfills, and forecast top-offs.
- `model_research_status.md`: current authoritative snapshot for baseline
  training, autoresearch run state, candidate statuses, and publish blockers.
- `model_research_log.md`: append-only operational log for model-research-agent
  turns, including train/init/step/gate/paper/promote/publish actions.
- `model_research_runbook.md`: step-by-step commands and decision rules for
  continuing model research and champion-adjacent promotion work.

## Update Rules
1. Read the matching `*_status.md` before starting a new collection or recovery run.
2. After every collection attempt:
   - update the matching `*_status.md`
   - append a new entry to the matching `*_collection_log.md`
3. If a new artifact is produced, update the artifact path and the verified
   metrics that depend on it in the matching status board.
4. Keep the markdown factual. Do not speculate about rows or coverage that have
   not been verified from parquet, checker agent output, or command output.
