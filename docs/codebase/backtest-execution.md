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
- `fees.py` and `slippage.py`: token-specific fee-rate lookups plus level-walk execution cost estimates
- `sizing.py`: bankroll-aware size selection
- `guardrails.py`: spread, freshness, and exposure limits
- `book_utils.py`: explicit `clob` / `missing` / optional `fixture` book loading contract
- `paper_broker.py`: conservative paper fill simulation
- `paper_market_maker.py`: two-sided fill simulation with realized/unrealized mark-to-market PnL
- `live_broker.py`: official live path behind explicit feature flags
- `live_market_maker.py`: guarded quote refresh path that skips a cycle if cancel fails
- `opportunity_shadow.py`: near-term live-market shadow validation and raw-gap / after-cost-edge summaries

## Boundary Between The Two
- `backtest/` measures what the model knew and when it knew it.
- grouped split policies (`market_day` or `target_day`) are mandatory; row-level splits are not part of the supported workflow.
- `execution/` decides whether an actionable edge survives spread, fees, slippage, and guardrails.
- Live trading must remain gated and isolated from the default research/paper path.
- `opportunity-report`, `paper-trader`, `live-trader`, `paper-mm`, and `live-mm` should treat missing live books as explicit skip states, not synthetic liquidity.
- signal paths should also treat missing calibrators or forecast-contract mismatches as explicit fail-closed states.
- `opportunity-shadow` reuses the same guardrails but logs the best raw gap and after-cost edge even for rejected markets, so “strategy is dead” and “book is unusable” remain distinct diagnoses.
- `opportunity-shadow` and `open-phase-shadow` summaries now also emit `by_horizon`, `by_city_horizon`, and top-level `gate_decision` / `gate_reason` so revenue gating can stay mechanical.
- `open-phase-shadow` is the listing/opening observer. It filters active markets by the earliest `acceptingOrdersTimestamp`/`createdAt` metadata and evaluates only recently opened markets with the configured horizon.
- Execution diagnostics should distinguish `raw_gap_non_positive`, `fee_killed_edge`, `slippage_killed_edge`, and `after_cost_positive_but_spread_too_wide` instead of collapsing everything into a generic “no edge”.
- `backtest --pricing-source quote_proxy` still is not exact replay. It keeps official historical last-price coverage but overlays a configurable half-spread proxy so execution assumptions are stricter than raw `real_history`.
- `benchmark-models` is the canonical model-selection path. It writes the leaderboard under `artifacts/benchmarks/v2/` and publishes both the research `champion` alias and the execution-oriented `trading_champion` alias for consumer commands.
- `revenue-gate-report` is the promotion checkpoint that combines recent-core benchmark results with shadow/open-phase viability before any small live pilot.

## Change Checklist
- Dataset-column changes affect both training and paper-trading workflows.
- Guardrail changes should be reflected in `docs/live-trading.md` and agent safety docs.
- Execution changes must not weaken the default live-trading gate.
