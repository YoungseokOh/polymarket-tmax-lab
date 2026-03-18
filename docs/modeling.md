# Modeling

## Target Quantity
The model target is not generic “city temperature.” It is the official source-specific daily maximum temperature that resolves the market.

## Two Supported Formulations

### 1. Direct daily-max postprocessing
- uses target-day predictors such as daily max, midday temperature, humidity, wind, cloud cover, lead time, and seasonality
- supports classical baselines and fast public-data workflows

### 2. Hourly trajectory to daily max
- models hourly 2m temperature over the settlement date
- derives daily maximum from sampled or structured hourly trajectories
- closer to the actual settlement quantity and preferred for advanced models

## Benchmark Ladder
- trivial baselines
- Gaussian EMOS and heteroscedastic baselines
- lead-time aware baselines
- det2prob neural model
- permutation-invariant ensemble approximation
- flexible probabilistic NN approximation
- transformer approximation
- spatial extension scaffold
- AI + NWP blend

## Calibration
- predictive distributions are mapped to market outcomes
- outcome probabilities can be calibrated via isotonic regression
- champion selection should use both meteorological skill and market-oriented metrics

## Champion Selection
- CRPS
- calibration gap / reliability
- bin-level Brier score
- realized market EV in research backtests

