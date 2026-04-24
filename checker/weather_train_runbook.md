# Weather Train Runbook

This file is the repeatable procedure for continuing `weather_train` collection
and keeping the markdown state in sync.

## 1. Preflight
Read:
1. `checker/weather_train_status.md`
2. `checker/weather_train_collection_log.md`

Then verify the workspace is still canonical:

```bash
scripts/pmtmax-workspace weather_train uv run pmtmax trust-check --workflow weather_training --json
```

Inspect the current gold state:

```bash
uv run python - <<'PY'
import pandas as pd
from pathlib import Path
path = Path('data/workspaces/weather_train/parquet/gold/weather_training_set.parquet')
df = pd.read_parquet(path, columns=['target_date'])
print({
    'rows': int(len(df)),
    'max_date': str(df['target_date'].max())[:10],
    'distinct_dates': int(df['target_date'].nunique()),
})
PY
```

## 2. Default Collection Settings
Use these unless there is a clear reason to change them:

```bash
--model gfs_seamless
--missing-only
--rate-limit-profile free
--http-timeout-seconds 15
--http-retries 1
--http-retry-wait-min-seconds 1
--http-retry-wait-max-seconds 8
```

Keep stderr progress on for manual runs. Use `--no-progress` only when a wrapper
script needs clean stdout JSON.

## 2a. Automated Queue Agent
For repeated older backfill, prefer the queue agent. It reads the checker
state, advances the next `7`-day chunk, updates `checker/weather_train_status.md`
and `checker/weather_train_collection_log.md` after every chunk, and stops on
the first throttled chunk.

```bash
scripts/pmtmax-workspace weather_train uv run python scripts/run_weather_train_queue_agent.py
```

Useful overrides:

```bash
scripts/pmtmax-workspace weather_train uv run python scripts/run_weather_train_queue_agent.py \
  --queue-start 2024-07-02 \
  --chunk-days 7 \
  --max-chunks 3 \
  --pretrain-refresh-threshold-rows 500 \
  --http-timeout-seconds 15 \
  --http-retries 1
```

Default behavior: when the current `weather_train` gold row count exceeds the
latest weather pretrain metadata by `500` rows or more, the queue agent runs
`gaussian_emos` pretrain refresh automatically and records that in the checker
log before continuing.

## 3. Past Gap-Fill Procedure
When recent dates are heavily throttled, fill older gaps first.

Recommended order:
1. `2024-05-31`
2. `2024-06-01..`

Single-day probe:

```bash
scripts/pmtmax-workspace weather_train uv run pmtmax collect-weather-training \
  --date-from 2024-05-31 \
  --date-to 2024-05-31 \
  --model gfs_seamless \
  --missing-only \
  --rate-limit-profile free \
  --http-timeout-seconds 15 \
  --http-retries 1 \
  --http-retry-wait-min-seconds 1 \
  --http-retry-wait-max-seconds 8
```

Small chunk probe:

```bash
scripts/pmtmax-workspace weather_train uv run pmtmax collect-weather-training \
  --date-from 2024-06-01 \
  --date-to 2024-06-03 \
  --model gfs_seamless \
  --missing-only \
  --rate-limit-profile free \
  --http-timeout-seconds 15 \
  --http-retries 1 \
  --http-retry-wait-min-seconds 1 \
  --http-retry-wait-max-seconds 8
```

Interpretation:
- any `available > 0`: continue older backfill; this is not a full-day hard stop
- all `retryable_error`: back off, log it, and try a different day later

## 4. Recent-Date Probe Procedure
Use this only as a low-frequency probe while free-tier throttling remains active.

```bash
scripts/pmtmax-workspace weather_train uv run pmtmax collect-weather-training \
  --date-from 2026-01-22 \
  --date-to 2026-01-22 \
  --model gfs_seamless \
  --missing-only \
  --rate-limit-profile free \
  --http-timeout-seconds 15 \
  --http-retries 1 \
  --http-retry-wait-min-seconds 1 \
  --http-retry-wait-max-seconds 8
```

If several consecutive days remain `0/30 retryable_error`, stop the recent-date
probe and return to past gap-fill or switch to a paid/API-key path.

## 5. Logging Rules
After every collection run:
1. update `checker/weather_train_status.md`
2. append one row to `checker/weather_train_collection_log.md`

Always record:
- attempted range
- chunk style (`single day`, `3-day`, `7-day`)
- rows added
- whether the result was `success`, `partial`, `retry-only`, or `interrupted`
- any interpretation about throttling

## 6. Pretrain Refresh
When the dataset grows materially, refresh the weather pretrain artifact:

```bash
scripts/pmtmax-workspace weather_train uv run pmtmax train-weather-pretrain --model-name gaussian_emos
```

Then update `checker/weather_train_status.md` with:
- row count
- artifact path
- metadata path
- dataset signature
- trained timestamp

## 7. Next Model Step
Once a current weather pretrain artifact exists, run the market fine-tune path:

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax train-advanced \
  --model-name lgbm_emos \
  --variant high_neighbor_oof \
  --pretrained-weather-model artifacts/workspaces/weather_train/models/v2/gaussian_emos.pkl
```
