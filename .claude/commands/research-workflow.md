Read `AGENTS.md`, `README.md`, and `docs/agent-skills/research-loop.md`.

Focus on `${ARGUMENTS}` when provided.

Return the shortest safe workflow for:
- market scanning
- bootstrap and dataset construction
- model training
- official `real_history` backtesting with a materialized price-history panel
- paper trading

Include the exact commands to run, the expected artifacts, and the docs that
must stay in sync if the workflow changes. Use `scripts/pmtmax-workspace`
for `historical_real`, `recent_core_eval`, and `ops_daily`; do not recommend
synthetic data, fixture forecasts, fabricated books, or `quote_proxy` as a
promotion source.
