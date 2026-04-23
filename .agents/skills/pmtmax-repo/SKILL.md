---
name: pmtmax-repo
description: Use when working anywhere inside polymarket-tmax-lab to onboard quickly, choose the right subsystem docs, and keep agent docs, workflows, and commit rules aligned.
---

# pmtmax-repo

Use this skill for general work inside `polymarket-tmax-lab`.

## First reads
1. Read `AGENTS.md`.
2. Read `docs/overview/README.md`.
3. Read `docs/codebase/index.md`.
4. Read the shared skill refs you need:
   - repo structure: `docs/agent-skills/repo-map.md`
   - data and warehouse ops: `docs/agent-skills/data-ops.md`
   - market parsing and truth fidelity: `docs/agent-skills/market-rules.md`
   - training, backtest, and paper workflow: `docs/agent-skills/research-loop.md`
   - safety and commit rules: `docs/agent-skills/safety-and-rules.md`
5. For recurring dataset work, read `checker/README.md`.

## Operating rules
- Treat `AGENTS.md` as the source of truth.
- Prefer the smallest relevant doc set instead of loading everything.
- If workflow, safety, or commit policy changes, update the matching docs in the same change.
