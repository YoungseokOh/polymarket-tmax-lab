# Backtesting

## Two Modes

### Research backtest
- available across the full historical market set
- uses exact truth and no-lookahead forecast reconstruction
- evaluates point skill, probabilistic skill, fair-vs-market edge, and counterfactual EV

### Execution replay
- requires exact archived L2 snapshots
- v1 includes the code path and fixture-driven tests
- older history without public exact L2 remains research-only for execution analysis

## Decision Horizons
- market open
- previous evening local time
- morning of settlement date
- optional higher-frequency refreshes near close

## Metrics
- MAE
- RMSE
- CRPS
- NLL
- reliability / calibration diagnostics
- bin-level Brier score
- paper PnL
- hit rate
- average edge captured

## Guardrails In Backtests
- minimum liquidity
- spread threshold
- stale forecast skip
- city-level exposure cap
- global exposure cap

