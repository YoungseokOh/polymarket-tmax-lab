# Model Research Status

Updated: 2026-04-24 KST

## Current Snapshot
- workspace: `historical_real`
- dataset profile: `real_market`
- dataset path: `data/workspaces/historical_real/parquet/gold/historical_training_set.parquet`
- panel path: `data/workspaces/historical_real/parquet/gold/historical_backtest_panel.parquet`
- dataset signature: `db5d3f197787ebe5a62e337d60f1df11748f6d8ec33562a9aa7972428fa99c07`
- panel signature: `c487b21d4a34412d431dcefe8cf3d80a4056a59dba65119d8b31dd4aadf7b308`
- baseline variant: `high_neighbor_oof`
- baseline artifact: `artifacts/workspaces/historical_real/models/v2/lgbm_emos__high_neighbor_oof.pkl`
- weather pretrain lineage: `artifacts/workspaces/weather_train/models/v2/gaussian_emos.pkl`
- active autoresearch run: `20260424-lgbm-high_neighbor_oof-agent`
- current public champion: `high_neighbor_oof`

## Candidate Ledger
- total candidate specs: `1`
- status counts: `{'discard': 1}`

## Recent Candidates
- `mr_20260424_lgbm_hi_01_lr_down_leaves_up`: `discard` (next `done`)

## Current Judgment
- Latest agent turn: `train_baseline, init_run, create_candidate, step:mr_20260424_lgbm_hi_01_lr_down_leaves_up` -> `discard` on `mr_20260424_lgbm_hi_01_lr_down_leaves_up`.
- Public champion publish is disabled by default; promotion stops at promoted YAML unless explicitly enabled.

## Next Queue
1. No pending candidate stage is open; the next run will auto-create the next candidate if capacity remains.
2. Keep `historical_real` mutating jobs serialized around this agent.
3. Re-run recent-core benchmark for a promoted candidate before any public publish.

## Daily Agent Command

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_model_research_agent.py
```
