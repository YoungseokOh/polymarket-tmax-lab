# Weather Train Status

Updated: 2026-04-25 KST

## Current Snapshot
- workspace: `weather_train`
- dataset profile: `weather_real`
- gold parquet: `data/workspaces/weather_train/parquet/gold/weather_training_set.parquet`
- total rows: `11,499`
- stations: `30`
- target dates: `392`
- model values in dataset: `gfs_seamless`
- `realized_daily_max` missing rows: `0`
- max target date present: `2026-01-21`

## Coverage

### Full-Coverage Dates
- `2024-01-03..2024-05-24`: `30` rows per day
- `2024-06-01..2024-12-30`: `30` rows per day
- `2026-01-01..2026-01-14`: `30` rows per day

### Partial-Coverage Dates
- `2024-01-01..2024-01-02`: `1` rows per day
- `2024-05-25..2024-05-28`: `22` rows per day
- `2024-05-29..2024-05-30`: `21` rows per day
- `2024-12-31..2025-01-04`: `17` rows per day
- `2025-01-05..2025-01-06`: `16` rows per day
- `2026-01-15..2026-01-17`: `22` rows per day
- `2026-01-18..2026-01-21`: `21` rows per day

### Retry-Only Dates
- `2024-05-31`: `0/30 available` on recorded probes; `retryable_error` / free-path daily-limit
- `2025-01-07..2025-01-27`: `0/30 available` on recorded probes; `retryable_error` / free-path daily-limit
- `2026-01-22..2026-01-28`: `0/30 available` on recorded probes; `retryable_error` / free-path daily-limit

## Current Judgment
- Queue agent advances older gap-fill in `7`-day chunks; `2` consecutive Open-Meteo `429` responses are treated as a daily-limit hit and cancel the remaining chunk.
- Latest successful collection range: `2024-12-24..2024-12-30`; `+210` rows.
- Latest throttled range on record: `2025-01-21..2025-01-27`; outcome `retry-only`.
- Latest free-path daily-limit signal: `2025-01-21..2025-01-27` produced repeated explicit `429`; future runs should record `rate-limit-cancelled` instead of exhausting the planned request set.
- Working interpretation: older backfill can reopen after cooldown windows, while recent-date historical-forecast collection remains materially weaker on the free path.

## Current Artifacts
- latest weather pretrain artifact:
  `/home/seok436/projects/polymarket-tmax-lab/artifacts/workspaces/weather_train/models/v2/gaussian_emos.pkl`
- artifact metadata:
  `artifacts/workspaces/weather_train/models/v2/gaussian_emos.json`
- pretrain dataset signature:
  `8cc575eedb0ce67c35c82a29fbe75f154b43a039e5460dc7693f6dcf28102c45`
- pretrain trained at:
  `2026-04-24T16:43:18.164735Z`

## Next Collection Queue
1. Continue older gap-fill from `2025-01-28` forward with `7`-day chunks only after the free path cooldown/reset.
2. Keep `2025-01-07..2025-01-27` and other isolated retry-only gaps as separate probes; do not block the forward older-backfill queue on them.
3. Retry `2026-01-22..2026-01-28` only as low-frequency probes or after moving to a paid/API-key path.

## Training Ready State
- weather pretrain artifact is aligned with the current dataset at `11,499` rows
- historical fine-tune input exists:
  `data/workspaces/historical_real/parquet/gold/historical_training_set.parquet`
- next recommended market fine-tune command:

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax train-advanced \
  --model-name lgbm_emos \
  --variant high_neighbor_oof \
  --pretrained-weather-model artifacts/workspaces/weather_train/models/v2/gaussian_emos.pkl
```
