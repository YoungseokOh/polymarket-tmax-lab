# Model Research Runbook

Use this runbook when continuing baseline training, autoresearch, or
champion-adjacent promotion work in `historical_real`.

## Read First
1. `checker/model_research_status.md`
2. `checker/model_research_log.md`
3. `docs/agent-skills/research-loop.md`
4. `docs/agent-skills/autoresearch.md`

## Default Daily Agent
```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_model_research_agent.py
```

Default behavior:
- verifies `historical_real` trust-check
- retrains baseline only when dataset signature or recorded weather-pretrain
  lineage changed
- reuses the current autoresearch run when dataset/panel signatures still match
- auto-creates the next small YAML candidate if no pending candidate exists
- processes one candidate through `step -> gate -> paper -> promote`
- updates `checker/model_research_status.md`
- appends `checker/model_research_log.md`

Public champion publish is disabled by default. Use it only after a
candidate-specific recent-core GO summary exists:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_model_research_agent.py \
  --enable-publish \
  --recent-core-summary-path artifacts/workspaces/recent_core_eval/.../recent_core_benchmark_summary.json
```

## Safety Rules
- Keep `historical_real` mutating jobs serialized around this agent.
- Do not overlap this agent with:
  - `build-dataset`
  - `materialize-backtest-panel`
  - `benchmark-models`
  - `backfill-price-history`
- Promotion is fail-closed:
  benchmark gate must pass, paper gate must be `GO`, and public publish still
  requires a recent-core `GO` summary.

## Useful Overrides
Process more than one candidate in a single monitored turn:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_model_research_agent.py \
  --max-candidates 2
```

Force baseline retraining:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_model_research_agent.py \
  --force-baseline-train
```

Disable heavy stages temporarily:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_model_research_agent.py \
  --no-enable-gate \
  --no-enable-paper \
  --no-enable-promote
```
