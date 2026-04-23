---
name: pmtmax-release-checklist
description: Use when validating that polymarket-tmax-lab is experiment-ready after bootstrap, training, backtest, or machine-to-machine seed restore.
---

# pmtmax-release-checklist

Use this skill for research-environment release checks.

## First reads
1. Read `AGENTS.md`.
2. Read:
   - `docs/agent-skills/release-checklist.md`
   - `docs/agent-skills/data-ops.md`
   - `docs/agent-skills/research-loop.md`

## Focus
- `bootstrap-lab`
- baseline training
- official `real_history` backtest smoke with a materialized price-history panel
- seed restore validation
- real-only trust-check blockers before release readiness
