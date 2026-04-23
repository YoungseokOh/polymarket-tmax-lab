# Weather Train Status

Updated: 2026-04-24 KST

## Current Snapshot
- workspace: `weather_train`
- dataset profile: `weather_real`
- gold parquet: `data/workspaces/weather_train/parquet/gold/weather_training_set.parquet`
- total rows: `4,992`
- stations: `30`
- target dates: `172`
- model values in dataset: `gfs_seamless`
- `realized_daily_max` missing rows: `0`
- max target date present: `2026-01-21`

## Coverage

### Full-Coverage Dates
- `2024-01-03..2024-05-24`: `30` rows per day
- `2026-01-01..2026-01-14`: `30` rows per day

### Partial-Coverage Dates
- `2024-01-01..2024-01-02`: Seoul-only smoke (`1` row per day)
- `2024-05-25..2024-05-28`: `22` rows per day
- `2024-05-29..2024-05-30`: `21` rows per day
- `2026-01-15..2026-01-17`: `22` rows per day
- `2026-01-18..2026-01-21`: `21` rows per day

### Retry-Only Dates
- `2024-05-31`: `0/30 available`, `30/30 retryable_error`
- `2026-01-22..2026-01-28`: `0/30 available`, `30/30 retryable_error` every day

## Current Judgment
- This does **not** look like a pure “daily hard cap exhausted” state.
- Evidence: after recent-date probes returned `0/30`, the same day still added
  `130` rows from the older range `2024-05-25..2024-05-30`.
- Working interpretation: Open-Meteo historical-forecast throttling is stronger
  for newer dates and can also partially affect older backfill windows.

## Current Artifacts
- latest weather pretrain artifact:
  `artifacts/workspaces/weather_train/models/v2/gaussian_emos.pkl`
- artifact metadata:
  `artifacts/workspaces/weather_train/models/v2/gaussian_emos.json`
- pretrain dataset signature:
  `80659d2c41efaa9d661c5d9e732182e81dc5144e48fc79c31c1aac10312f21db`
- pretrain trained at:
  `2026-04-23T15:51:26.253126Z`

## Next Collection Queue
1. Retry `2024-05-31` as a single-day past-gap probe.
2. Continue older gap-fill from `2024-06-01` forward with `1`-day or `3`-day
   chunks when partial success is acceptable.
3. Retry `2026-01-22..2026-01-28` only as low-frequency probes or after moving
   to a paid/API-key path.

## Training Ready State
- weather pretrain: complete on `4,992` rows
- historical fine-tune input exists:
  `data/workspaces/historical_real/parquet/gold/historical_training_set.parquet`
- next recommended market fine-tune command:

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax train-advanced \
  --model-name lgbm_emos \
  --variant high_neighbor_oof \
  --pretrained-weather-model artifacts/workspaces/weather_train/models/v2/gaussian_emos.pkl
```
