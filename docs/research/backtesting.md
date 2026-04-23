# Backtesting

## Two Modes

### Research backtest
- available across the full historical market set
- uses exact truth and no-lookahead forecast reconstruction
- has two pricing paths:
  - `real_history`: uses official Polymarket historical token prices at each decision timestamp
  - `quote_proxy`: diagnostic only, using official price history plus an explicit entry half-spread proxy
- champion promotion and GO/NO-GO decisions use `real_history`; `quote_proxy` is not a canonical promotion source

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
- `real_history` skips decision rows when official market price history is missing or stale
- `real_history` and `quote_proxy` should be rebuilt from archived official history already stored in the warehouse; do not assume a late live refetch will still return old token histories
- `real_history` uses flat notional entry and holds to settlement because historical L2 depth is not public
- `quote_proxy` uses the same decision-time price coverage as `real_history`, but entry executes at `last_price + half_spread_proxy` for buys, making it a stricter execution approximation than raw last-price replay
- backtest and benchmark summaries record dataset/panel paths plus artifact and frame signatures so stale or mixed artifacts can be rejected during review
- synthetic inventories, fixture forecasts, and fabricated books are rejected by trust-check before canonical build or publish paths
- paper market-making metrics should be read as mark-to-market outputs with separate realized and unrealized PnL, not cash-only flow
