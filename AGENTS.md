# AGENTS

## Purpose
This repository is a research-first, trading-aware system for one market family only:
`Highest temperature in [city] on [date]?`

Use this file as the shared operating contract for both Codex and Claude. If another
agent-facing file disagrees with this file, this file wins.

## Collaboration Defaults
- Respond to the user in Korean unless they explicitly ask for another language.
- Keep live trading disabled unless the task explicitly targets the gated live path.
- Prefer settlement fidelity over modeling novelty. Do not silently swap stations or truth sources.
- Update documentation as part of the same change when folder ownership, workflows, or safety rules change.

## First Reads
Read these before broad changes:

1. `README.md`
2. `docs/architecture.md`
3. `docs/codebase/index.md`
4. `docs/market-rules.md`

Then load the subsystem-specific codebase guide that matches the task.

## Working Agreement
- `src/pmtmax/markets` owns market discovery, market filtering, rule parsing, and `MarketSpec`.
- `src/pmtmax/weather` owns forecast ingestion, archived forecast reconstruction, and official truth adapters.
- `src/pmtmax/modeling` owns probabilistic models, calibration, daily-max transforms, and bin mapping.
- `src/pmtmax/backtest` owns no-lookahead dataset construction, replay, metrics, and PnL evaluation.
- `src/pmtmax/execution` owns edge calculation, fees, slippage, sizing, guardrails, and brokers.
- `src/pmtmax/storage` owns DuckDB/Parquet persistence and shared schemas.

## Documentation Update Rules
- If a major folder's responsibility changes, update the matching file under `docs/codebase/`.
- If the CLI workflow changes, update `README.md` and the matching docs under `docs/agent-skills/`.
- If commit message rules change, update both `docs/commit-convention.md` and `scripts/check_commit_message.py`.
- If live-trading safety behavior changes, update `docs/live-trading.md`, `.claude/commands/trading-safety.md`, and `docs/agent-skills/safety-and-rules.md`.

## Standard Commands
```bash
uv sync --all-extras
pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run pmtmax scan-markets
uv run pmtmax build-dataset --city Seoul --city NYC
uv run pmtmax train-baseline --model-name gaussian_emos
uv run pmtmax backtest --model-name gaussian_emos
uv run pmtmax paper-trader --model-name gaussian_emos
uv run pytest
```

## Agent Assets
- `CLAUDE.md` is a compatibility bridge for Claude and points back to this file.
- `.claude/commands/` contains Claude-specific prompt entrypoints.
- `.claude/skills/` contains Claude project skills.
- `tools/skills/` contains Codex repo-local skills.
- `docs/agent-skills/` contains shared skill reference docs used by both tools.
- `docs/codebase/` is the folder-by-folder map of the codebase.

Recommended shared skills:
- `pmtmax-repo`
- `pmtmax-data-ops`
- `pmtmax-market-rules`
- `pmtmax-research-loop`
- `pmtmax-commit`
- `pmtmax-release-checklist`

## Commit Rules
Use the conventional commit policy in `docs/commit-convention.md`.
Install the `commit-msg` pre-commit hook if you want local enforcement.
