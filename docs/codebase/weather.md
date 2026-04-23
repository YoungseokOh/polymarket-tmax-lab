# weather

## Responsibility
`src/pmtmax/weather` owns forecast ingestion, archived forecast reconstruction,
feature building, lagged pseudo-ensemble support, and official truth retrieval.

## Key Modules
- `openmeteo_client.py`: public forecast and archive access
- `training_data.py`: weather-real station/date collection for `weather_train` pretraining, without Polymarket market ids or prices
- `historical_forecast.py` and `previous_runs.py`: admissible forecast reconstruction helpers
- `ecmwf_adapter.py` and `kma_adapter.py`: provider-specific feature extraction
- `lagged_ensemble.py`: public-data pseudo-ensemble construction
- `features.py`: target-day and hourly-trajectory feature engineering
- `intraday_observation.py`: live target-day lower-bound adapters for HKO/CWA/AIR_CALP observation paths
- `truth_sources/base.py`: common truth adapter contract
- `truth_sources/wunderground.py`, `hko.py`, `cwa.py`: source-specific truth retrieval

## Inputs And Outputs
- Input: station coordinates, issue/target dates, weather model identifiers, official source metadata
- Output: model-ready forecast features and finalized daily-max observations

## Operational Rules
- Do not silently substitute another station or another source.
- Intraday observation paths are target-day only and should prefer documented source-specific lower bounds before aviation fallback.
- If official truth is unavailable, fail closed or use an explicitly documented public-station override for the same airport, such as AMO `AIR_CALP` for Seoul / RKSI or the Wunderground public historical API for London / EGLC and NYC / KLGA.
- Wunderground historical access may use `PMTMAX_WU_API_KEY` when you have one, but the research path may also derive the documented public front-end Weather.com key from the station page at runtime. Do not hardcode any key material into source.
- Historical forecast reconstruction uses Open-Meteo's generic archive forecast endpoint and must avoid lookahead.
- Exact decision-horizon backfills may add `single_run` rows on top of generic archive rows. Gold materialization should prefer those horizon-specific rows when they exist.
- Research backfills default to strict archive mode; real-only CLI paths reject fixture fallback and trust-check treats existing fixture rows as non-canonical.
- Archive coverage should be inspected with `pmtmax summarize-forecast-availability` before changing the default model list.
- Open-Meteo bulk weather collection should run only through `scripts/pmtmax-workspace weather_train uv run pmtmax collect-weather-training`, with small resumable date chunks and `--missing-only` by default.
- `collect-weather-training` now has explicit HTTP control flags for Open-Meteo:
  `--http-timeout-seconds`, `--http-retries`,
  `--http-retry-wait-min-seconds`, and `--http-retry-wait-max-seconds`.
  It also emits station/date stderr progress by default while keeping the final
  summary JSON on stdout.
- Field note as of April 24, 2026: `weather_train` successfully collected
  full ranges `2024-01-03..2024-05-24` and `2026-01-01..2026-01-14`, with
  partial coverage extending through `2024-05-30` and `2026-01-21`. A slow
  day-by-day crawl over `2026-01-22..2026-01-28` still returned only
  `retryable_error`, while `2024-05-25..2024-05-30` still added `130` rows,
  indicating date-sensitive upstream throttling on the free path rather than a
  simple whole-day stop.

## Change Checklist
- Truth-source changes should be reflected in `docs/markets/market-rules.md` when settlement behavior changes.
- Feature changes should be checked against `backtest/dataset_builder.py` and the training registry.
- New providers or archived-run logic belong here, not in `modeling/`.
