# Modeling

## Target Quantity
The model target is not generic “city temperature.” It is the official source-specific daily maximum temperature that resolves the market.

## Supported Formulation

### Direct daily-max postprocessing
- uses target-day predictors such as daily max, midday temperature, humidity, wind, cloud cover, lead time, and seasonality
- supports classical baselines and fast public-data workflows
- advanced candidates now add contextual availability flags, city/horizon indicators, and seasonality features through a shared design matrix

## Benchmark Ladder
- `gaussian_emos`
- `tuned_ensemble`
- `det2prob_nn`

## Public Candidate Shapes
- `gaussian_emos`: simple heteroscedastic linear baseline
- `tuned_ensemble`: contextual mixture-of-experts combining linear, lead-time polynomial, and tree experts behind a learned gate
- `det2prob_nn`: standalone mixture-density neural network that emits Gaussian-mixture daily-max forecasts

## Contract
- canonical datasets live under `data/parquet/gold/v2/`
- backtests use grouped leakage-safe splits with `market_day` or `target_day`
- artifacts carry `contract_version`, `dataset_signature`, `split_policy`, `seed`, and optional `calibration_path`
- paper/live/opportunity paths consume calibrated probabilities when available and fail closed as `missing_calibrator` when a calibrator is absent

## Calibration
- predictive distributions are mapped to market outcomes
- outcome probabilities can be calibrated via isotonic regression
- training fits calibrators on a held-out calibration split and stores them next to the model artifact
- champion selection should use both meteorological skill and market-oriented metrics
- advanced models may emit `gaussian_mixture` forecasts; calibration still happens on mapped outcome probabilities rather than on component weights directly

## Champion Selection
- `benchmark-models` writes leaderboard artifacts under `artifacts/benchmarks/v2/`
- the active champion alias is published to `artifacts/models/v2/champion.pkl`
- champion scoring combines CRPS, Brier, calibration gap, forecast error, and trading-like PnL
