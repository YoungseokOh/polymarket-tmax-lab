Read `AGENTS.md`, `README.md`, and `docs/agent-skills/release-checklist.md`.

Focus on `${ARGUMENTS}` when provided.

Return the shortest safe checklist for:
- bootstrap
- baseline training
- official `real_history` backtest smoke with a materialized price-history panel
- seed export or restore when relevant

Include the exact commands, required artifacts, and the first failure condition
that should stop the workflow. Treat synthetic inventories, fixture forecasts,
fabricated books, or `quote_proxy`-only evidence as release blockers.
