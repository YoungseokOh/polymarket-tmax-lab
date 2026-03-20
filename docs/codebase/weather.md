# weather

## Responsibility
`src/pmtmax/weather` owns forecast ingestion, archived forecast reconstruction,
feature building, lagged pseudo-ensemble support, and official truth retrieval.

## Key Modules
- `openmeteo_client.py`: public forecast and archive access
- `historical_forecast.py` and `previous_runs.py`: admissible forecast reconstruction helpers
- `ecmwf_adapter.py` and `kma_adapter.py`: provider-specific feature extraction
- `lagged_ensemble.py`: public-data pseudo-ensemble construction
- `features.py`: target-day and hourly-trajectory feature engineering
- `truth_sources/base.py`: common truth adapter contract
- `truth_sources/wunderground.py`, `hko.py`, `cwa.py`: source-specific truth retrieval

## Inputs And Outputs
- Input: station coordinates, issue/target dates, weather model identifiers, official source metadata
- Output: model-ready forecast features and finalized daily-max observations

## Operational Rules
- Do not silently substitute another station or another source.
- If official truth is unavailable, fail closed or use an explicitly documented public-station override for the same airport, such as AMO `AIR_CALP` for Seoul / RKSI or NOAA Global Hourly for other Wunderground-family cities.
- Wunderground historical Weather.com access must read `PMTMAX_WU_API_KEY` from runtime settings; do not check API keys into source.
- Historical forecast reconstruction uses Open-Meteo's generic archive forecast endpoint and must avoid lookahead.
- Exact decision-horizon backfills may add `single_run` rows on top of generic archive rows. Gold materialization should prefer those horizon-specific rows when they exist.
- Research backfills default to strict archive mode; fixture fallback is demo-only and must be logged in bronze tables.
- Archive coverage should be inspected with `pmtmax summarize-forecast-availability` before changing the default model list.

## Change Checklist
- Truth-source changes should be reflected in `docs/market-rules.md` when settlement behavior changes.
- Feature changes should be checked against `backtest/dataset_builder.py` and the training registry.
- New providers or archived-run logic belong here, not in `modeling/`.
