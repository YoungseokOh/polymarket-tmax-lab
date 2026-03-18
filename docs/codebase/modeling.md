# modeling

## Responsibility
`src/pmtmax/modeling` turns station-aligned forecast features into probabilistic
daily-max forecasts, calibrates them, and maps them onto Polymarket outcomes.

## Key Modules
- `baselines/`: climatology, raw-model baselines, Gaussian EMOS, tsEMOS, lead-time-continuous baseline
- `advanced/`: det2prob neural path plus scaffolded ensemble-aware and blend models
- `train.py`: model registry, feature selection, artifact serialization
- `predict.py`: model loading, forecast generation, and outcome-probability mapping
- `daily_max.py`: hourly-to-daily-max transforms
- `bin_mapper.py`: exact Polymarket outcome mapping for parametric or sample forecasts
- `calibration.py`, `evaluation.py`, `sampling.py`, `champion.py`: supporting evaluation and selection utilities

## Data Contract
- Training expects a tabular dataset from `backtest/dataset_builder.py`.
- Shared targets are `realized_daily_max` and, when available, `winning_outcome`.
- Model outputs feed `ProbForecast` and then execution logic.

## What To Keep Aligned
- Baseline and advanced model names must stay in sync with CLI defaults and docs.
- Probability mapping logic must remain consistent with `MarketSpec.outcome_schema`.
- If feature names change, check `train.py`, `predict.py`, and the dataset builder together.

## Change Checklist
- Changes to market outcome mapping require `tests/test_bin_mapper.py`.
- Changes to daily-max transforms require `tests/test_daily_max.py`.
- If a model becomes the preferred champion, update README workflow examples only if the default user path changes.
