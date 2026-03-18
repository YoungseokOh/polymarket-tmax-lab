# backtest and execution

## Responsibility
`src/pmtmax/backtest` and `src/pmtmax/execution` separate research evaluation
from signal generation and broker behavior.

## backtest
- `dataset_builder.py`: builds no-lookahead training rows from market specs, forecast inputs, and official truth
- `rolling_origin.py`: train/test split logic for research evaluation
- `market_replay.py`: replay helpers for archived market states
- `metrics.py` and `pnl.py`: forecast skill and realized paper-PnL utilities

## execution
- `edge.py`: fair-value versus executable-price comparison
- `fees.py` and `slippage.py`: execution cost estimates
- `sizing.py`: bankroll-aware size selection
- `guardrails.py`: spread, freshness, and exposure limits
- `paper_broker.py`: conservative paper fill simulation
- `live_broker.py`: official live path behind explicit feature flags

## Boundary Between The Two
- `backtest/` measures what the model knew and when it knew it.
- `execution/` decides whether an actionable edge survives spread, fees, slippage, and guardrails.
- Live trading must remain gated and isolated from the default research/paper path.

## Change Checklist
- Dataset-column changes affect both training and paper-trading workflows.
- Guardrail changes should be reflected in `docs/live-trading.md` and agent safety docs.
- Execution changes must not weaken the default live-trading gate.
