# Live Trading

## Default State
Live trading is implemented but disabled by default.

## Required Flags
- `PMTMAX_LIVE_TRADING=true`
- `PMTMAX_CONFIRM_LIVE_TRADING=YES_I_UNDERSTAND`

## Required Credentials
- `PMTMAX_POLY_PRIVATE_KEY`
- `PMTMAX_POLY_API_KEY`
- `PMTMAX_POLY_API_SECRET`
- `PMTMAX_POLY_PASSPHRASE`
- `PMTMAX_POLY_CHAIN_ID` defaults to Polygon `137`
- `PMTMAX_POLY_SIGNATURE_TYPE` is optional and only needed for non-EOA signing paths
- `PMTMAX_POLY_FUNDER_ADDRESS` is optional and only needed for proxy or smart-wallet funding setups

## SDK Policy
- official `py-clob-client` only
- no unofficial live-trading clients

## Safety Rules
- fail closed if flags are absent
- fail closed if credentials are absent
- fail closed if the current CLOB book is missing; do not auto-replace it with a synthetic book in live or dry-run live paths
- fail closed if the forecast contract is not v2 or if calibrated probabilities are unavailable (`missing_calibrator`)
- fail closed if `live-mm` cannot cancel existing live orders before posting a refreshed quote set
- no market orders by default
- limit-order logic only
- keep live execution isolated from research and paper modes

## Operational Notes
- verify legal and regional eligibility before any live use
- validate fee, nonce, and auth handling against current official docs before enabling
- use `uv run pmtmax live-trader --dry-run` first to collect a preflight report and signed-order previews
- use `uv run pmtmax opportunity-report --core-recent-only --model-name trading_champion` before any live or paper session to distinguish `missing_book` from genuine `no_positive_edge`
- use `uv run pmtmax benchmark-models` before relying on the default `champion` or `trading_champion` alias in paper/live/opportunity paths
- use `uv run pmtmax revenue-gate-report` before promoting the small-cap live pilot; benchmark `GO` without shadow/open-phase confirmation remains insufficient
- `live-trader`, `scan-daemon`, and `opportunity-report` now default to the
  checked-in recent horizon policy (`configs/recent-core-horizon-policy.yaml`);
  disallowed city/date combinations are surfaced as `policy_filtered`
- `configs/revenue-pilot-core.yaml` is the conservative live-pilot preset: keep bankroll at roughly `$500`, city exposure at `100`, global exposure at `200`, and stay on the recent-core city set with manual approval
- signal outputs are written under `artifacts/signals/v2/`

## See Also
- `docs/codebase/backtest-execution.md` for the execution folder split
- `docs/commit-convention.md` for the `type: subject` commit convention used by trading-related changes
