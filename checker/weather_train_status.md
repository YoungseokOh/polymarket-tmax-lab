# Weather Train Status

Updated: 2026-04-24 KST

## Current Snapshot
- workspace: `weather_train`
- dataset profile: `weather_real`
- gold parquet: `data/workspaces/weather_train/parquet/gold/weather_training_set.parquet`
- total rows: `8,442`
- stations: `30`
- target dates: `287`
- model values in dataset: `gfs_seamless`
- `realized_daily_max` missing rows: `0`
- max target date present: `2026-01-21`

## Coverage

### Full-Coverage Dates
- `2024-01-03..2024-05-24`: `30` rows per day
- `2024-06-01..2024-09-23`: `30` rows per day
- `2026-01-01..2026-01-14`: `30` rows per day

### Partial-Coverage Dates
- `2024-01-01..2024-01-02`: `1` rows per day
- `2024-05-25..2024-05-28`: `22` rows per day
- `2024-05-29..2024-05-30`: `21` rows per day
- `2026-01-15..2026-01-17`: `22` rows per day
- `2026-01-18..2026-01-21`: `21` rows per day

### Retry-Only Dates
- `2024-05-31`: `0/30 available`, `30/30 retryable_error` every day
- `2026-01-22..2026-01-28`: `0/30 available`, `30/30 retryable_error` every day

## Current Judgment
- Queue agent advances older gap-fill in `7`-day chunks and stops on the first `retryable_error` / `429` chunk.
- Latest successful collection range: `2024-09-17..2024-09-23`; `+210` rows.
- Latest throttled range on record: `2024-06-01..2024-06-03`; outcome `retry-only`.
- Working interpretation: older backfill can reopen after cooldown windows, while recent-date historical-forecast collection remains materially weaker on the free path.

## Current Artifacts
- latest weather pretrain artifact:
  `/home/seok436/projects/polymarket-tmax-lab/artifacts/workspaces/weather_train/models/v2/gaussian_emos.pkl`
- artifact metadata:
  `artifacts/workspaces/weather_train/models/v2/gaussian_emos.json`
- pretrain dataset signature:
  `f273242a8dd2f0427351e80de93861391d4a0aeea7c3855d45e752d43c1a182a`
- pretrain trained at:
  `2026-04-24T13:48:32.398464Z`

## Next Collection Queue
1. Continue older gap-fill from `2024-09-24` forward with `7`-day chunks while the free path remains open.
2. Keep isolated retry-only gaps as separate probes; do not block the forward older-backfill queue on them.
3. Retry `2026-01-22..2026-01-28` only as low-frequency probes or after moving to a paid/API-key path.

## Training Ready State
- weather pretrain artifact is aligned with the current dataset at `8,442` rows
- historical fine-tune input exists:
  `data/workspaces/historical_real/parquet/gold/historical_training_set.parquet`
- next recommended market fine-tune command:

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax train-advanced \
  --model-name lgbm_emos \
  --variant high_neighbor_oof \
  --pretrained-weather-model artifacts/workspaces/weather_train/models/v2/gaussian_emos.pkl
```
