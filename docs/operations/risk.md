# Risk

## Model Risk
- forecast distributions can be miscalibrated
- source/station mismatches can invalidate an otherwise good model

## Data Risk
- public endpoint shapes can change
- archived forecast availability is imperfect across providers
- exact-source retrieval can fail, especially for brittle official pages

## Execution Risk
- public price history is not exact historical L2
- live liquidity can vanish near resolution
- spread and slippage can erase nominal edge

## Operational Risk
- stale caches
- broken schedulers
- parser regressions
- timezone mistakes

## Policy
- fail closed on missing exact truth
- fail closed on missing live-trading confirmations
- do not present approximations as exact execution history

