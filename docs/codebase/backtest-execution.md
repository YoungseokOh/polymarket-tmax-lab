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
- `src/pmtmax/markets/book_utils.py`: explicit `clob` / `missing` book loading contract used by execution paths; fabricated book fallback is not allowed
- `paper_broker.py`: conservative paper fill simulation
- `paper_market_maker.py`: two-sided fill simulation with realized/unrealized mark-to-market PnL
- `live_broker.py`: official live path behind explicit feature flags
- `live_market_maker.py`: guarded quote refresh path that skips a cycle if cancel fails
- `opportunity_shadow.py`: near-term live-market shadow validation and raw-gap / after-cost-edge summaries
- `monitoring/observation_station.py`: observation-driven shadow runner, live-pilot queue writer, and station-summary helpers
- `monitoring/station_dashboard.py`: static JSON/HTML renderer that fuses discovery, observation, execution, and revenue-gate artifacts into one station view

## Boundary Between The Two
- `backtest/` measures what the model knew and when it knew it.
- grouped split policies (`market_day` or `target_day`) are mandatory; row-level splits are not part of the supported workflow.
- `execution/` decides whether an actionable edge survives spread, fees, slippage, and guardrails.
- Live trading must remain gated and isolated from the default research/paper path.
- `opportunity-report`, `paper-trader`, `live-trader`, `paper-mm`, and `live-mm` should treat missing live books as explicit skip states, not synthetic liquidity.
- `paper-trader`, `paper-multimodel-report`, and `execution-sensitivity-report` reject any legacy `book_source=fixture` rows; unavailable CLOB data must be recorded as `missing_book`.
- signal paths should also treat missing calibrators or forecast-contract mismatches as explicit fail-closed states.
- `opportunity-shadow` reuses the same guardrails but logs the best raw gap and after-cost edge even for rejected markets, so “strategy is dead” and “book is unusable” remain distinct diagnoses.
- `observation-report` / `observation-shadow` sit one layer above the same execution diagnostics and zero out outcome bins already made impossible by the latest live observation before ranking candidates.
- `opportunity-shadow` and `open-phase-shadow` summaries now also emit `by_horizon`, `by_city_horizon`, and top-level `gate_decision` / `gate_reason` so revenue gating can stay mechanical.
- `approve-live-candidate` is the manual-approval bridge from observation queue to the guarded live broker.
- `open-phase-shadow` is the listing/opening observer. It filters active markets by the earliest `acceptingOrdersTimestamp`/`createdAt` metadata and evaluates only recently opened markets with the configured horizon.
- `hope-hunt-report` / `hope-hunt-daemon` sit on top of the same execution diagnostics but constrain discovery to supported Wunderground-family `research_public` cities and rank fresh listings without placing orders.
- Execution diagnostics should distinguish `raw_gap_non_positive`, `fee_killed_edge`, `slippage_killed_edge`, and `after_cost_positive_but_spread_too_wide` instead of collapsing everything into a generic “no edge”.
- `backtest --pricing-source quote_proxy` is diagnostic only. It keeps official historical last-price coverage but overlays a configurable half-spread proxy, and it must not drive champion promotion.
- `benchmark-models` is the workspace-local model-selection path. It writes the leaderboard under the active workspace benchmark root and does not mutate the public alias.
- `publish-champion` is the only public-alias promotion path. It copies one calibrated artifact into `artifacts/public_models/champion.*` only after the recent-core benchmark summary is `GO`.
- public champion metadata is part of the safety contract: missing `publish_gate.decision=GO` or non-`real_market` metadata must fail closed.
- autoresearch promotion is YAML-only and requires CLI-generated gate leaderboard artifacts, matching dataset/panel signatures, a candidate calibrator, and paper `overall_gate_decision=GO`.
- `revenue-gate-report` is the promotion checkpoint that combines recent-core benchmark results with shadow/open-phase viability before any small live pilot.

## Change Checklist
- Dataset-column changes affect both training and paper-trading workflows.
- Guardrail changes should be reflected in `docs/operations/live-trading.md` and agent safety docs.
- Execution changes must not weaken the default live-trading gate.
