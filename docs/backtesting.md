# Backtesting

## Two Modes

### Research backtest
- available across the full historical market set
- uses exact truth and no-lookahead forecast reconstruction
- has two pricing paths:
  - `synthetic`: uses the bundled research book around captured outcome prices
  - `real_history`: uses official Polymarket historical token prices at each decision timestamp
- `synthetic` remains explicit research-only pricing; active scanning, paper trading, and live previews no longer auto-fallback to synthetic books when CLOB data is missing

### Execution replay
- still requires exact archived L2 snapshots
- older history without public exact L2 remains outside the exact replay subset
- `real_history` is decision-time evaluation, not exact historical book replay

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
- `synthetic` keeps spread/liquidity sizing logic
- `real_history` skips decision rows when official market price history is missing or stale
- `real_history` uses flat notional entry and holds to settlement because historical L2 depth is not public
- paper market-making metrics should be read as mark-to-market outputs with separate realized and unrealized PnL, not cash-only flow
