# Historical Price Recovery Runbook

Use this runbook when continuing official Polymarket price-history recovery in
`historical_real`.

## Read First
1. `checker/historical_price_status.md`
2. `checker/historical_price_collection_log.md`
3. `docs/agent-skills/data-ops.md`
4. `docs/agent-skills/research-loop.md`

## Daily Default
Run one shard per day unless you are intentionally accelerating recovery:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_historical_price_recovery_agent.py
```

Default behavior:
- reads `checker/historical_price_status.md`
- infers the next shard offset
- runs `backfill-price-history --only-missing --price-no-cache` semantics on one shard
- rebuilds the canonical `historical_backtest_panel`
- refreshes `artifacts/workspaces/historical_real/coverage/latest_price_history_coverage.json`
- appends `checker/historical_price_collection_log.md`
- rewrites `checker/historical_price_status.md`

## Safety Rules
- Keep `historical_real` mutating steps serialized.
- Do not overlap this agent with another `historical_real` writer such as:
  - `backfill-forecasts`
  - `build-dataset`
  - `materialize-backtest-panel`
  - `benchmark-models`
- It is safe to run this agent in parallel with the `weather_train` queue agent
  because the workspaces and upstream APIs differ.
- Canonical panel overwrite is intentional here. The run will create recovery
  backups under `artifacts/recovery/`.

## Useful Overrides
Run a specific shard:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_historical_price_recovery_agent.py \
  --shard-start 250
```

Run more than one shard in a monitored session:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_historical_price_recovery_agent.py \
  --max-shards 3 \
  --sleep-seconds 2
```

Change shard size:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_historical_price_recovery_agent.py \
  --shard-size 50
```

## When To Re-Evaluate
- Re-run `real_history` backtest or benchmark after a meaningful increase in
  checker `panel-ready decision rows`.
- Use the checker `latest backtest priced_decision_rows` field as the last
  verified evaluation anchor. It only moves when a new backtest or benchmark is run.
