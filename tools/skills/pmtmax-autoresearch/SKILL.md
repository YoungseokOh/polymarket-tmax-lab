---
name: pmtmax-autoresearch
description: Use when running the recency_neighbor_oof-centered autoresearch loop, editing candidate YAML specs, gating new LGBM variants, or promoting winners in polymarket-tmax-lab.
---

# pmtmax-autoresearch

Use this skill for the agent-driven autoresearch loop around `lgbm_emos`.

## First reads
1. Read `AGENTS.md`.
2. Read:
   - `docs/agent-skills/autoresearch.md`
   - `docs/agent-skills/research-loop.md`
   - `docs/codebase/modeling.md`

## Critical rules
- baseline is `recency_neighbor_oof` unless the run manifest says otherwise.
- create or edit one candidate YAML at a time under `artifacts/autoresearch/<run_tag>/candidates/`.
- use `autoresearch-step` for quick keep/discard/crash, then `autoresearch-gate`, then `autoresearch-analyze-paper`.
- canonical `historical_training_set*` / `historical_backtest_panel` stay immutable unless explicitly promoted elsewhere.
- `champion` publish is never implicit; `autoresearch-promote` never publishes aliases.

## Focus
- run scaffold: `artifacts/autoresearch/<run_tag>/`
- promoted spec registry: `configs/autoresearch/lgbm_emos/promoted/`
- candidate artifacts: `artifacts/autoresearch/<run_tag>/models/`
- shared program: `artifacts/autoresearch/<run_tag>/program.md`
