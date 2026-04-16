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
uv run pmtmax backfill-truth
uv run pmtmax backfill-forecasts
# ⚠️ ALWAYS pass --markets-path. Without it, only 12 example rows are built.
uv run pmtmax build-dataset \
    --markets-path configs/market_inventory/full_training_set_snapshots.json \
    --allow-canonical-overwrite
uv run pmtmax train-advanced --model-name lgbm_emos --variant recency_neighbor_oof
uv run python scripts/quick_eval.py
scripts/pmtmax-workspace historical_real uv run pmtmax benchmark-models --retrain-stride 30
scripts/pmtmax-workspace research_synth uv run pmtmax autoresearch-init
scripts/pmtmax-workspace research_synth uv run pmtmax autoresearch-step --spec-path artifacts/workspaces/research_synth/autoresearch/<run_tag>/candidates/my_candidate.yaml
scripts/run_recent_core_benchmark_local.sh
uv run pmtmax publish-champion /path/to/workspace/model.pkl --recent-core-summary-path artifacts/workspaces/recent_core_eval/recent_core_benchmark/recent_core_benchmark_summary.json
scripts/pmtmax-workspace ops_daily uv run pmtmax observation-report --model-name champion
scripts/pmtmax-workspace ops_daily uv run pmtmax observation-shadow --model-name champion --max-cycles 1
uv run pmtmax station-dashboard
scripts/pmtmax-workspace ops_daily uv run pmtmax station-cycle --model-name champion
scripts/pmtmax-workspace ops_daily uv run pmtmax scan-edge \
    --model-name champion \
    --min-edge 0.15 \
    --min-model-prob 0.05 \
    --max-model-prob 0.95 \
    --output artifacts/workspaces/ops_daily/signals/v2/scan_edge_latest.json
uv run pytest
```

## Safety Rules
- `build-dataset` without `--markets-path` destroys the training dataset (overwrites with 12 rows).
  A shrinkage guard (50% threshold) is in place, but always be explicit.
- canonical gold/panel outputs are immutable by default.
  `historical_training_set*` and `historical_backtest_panel` require `--allow-canonical-overwrite` once they already exist.
- canonical overwrite is promotion-only.
  Prefer variant `--output-name` values for experiments, and only unlock canonical writes when you intend to replace the checked-in research baseline.
- canonical overwrite now creates a timestamped backup under `artifacts/recovery/` before replacing parquet/manifests.
- lag recovery should prefer `--truth-no-cache` plus source-family concurrency `--truth-per-source-limit 1`.
- autoresearch loops should stay on YAML candidate specs under `artifacts/autoresearch/<run_tag>/candidates/`.
  Do not rewrite canonical aliases or canonical datasets inside quick keep/discard loops.
- observation-driven live pilots must stay manual-approval-first.
  Queue candidates via `observation-report` / `observation-shadow`, then use `approve-live-candidate` for preview/post.
- if a live candidate comes from a `research_public` market, do not hide its tier or risk flags and keep sizing more conservative than `exact_public`.
- observation overrides are target-day only and should prefer `exact_public intraday -> documented research intraday -> METAR fallback`.
- `scan-edge` without `--min-model-prob`/`--max-model-prob` generates 0%/100% model-prob signals (overconfident noise).
- Never run `benchmark-models` without `--retrain-stride 30` — default stride=1 takes 10+ hours.
- `benchmark-models` no longer publishes aliases. Public promotion must go through `publish-champion` with a recent-core `GO` summary.
- public alias is single-source-of-truth: `artifacts/public_models/champion.*`.
- keep workspace roots separate via `scripts/pmtmax-workspace`: `ops_daily`, `historical_real`, `research_synth`, `recent_core_eval`.
- Synthetic single-run NWP re-collection → always use `synthetic_historical_snapshots.json` (103,110 markets).
  Never substitute `full_training_set_snapshots.json` (1,834 real markets) or `synthetic_gfs_eligible.json` (42,870 subset).
  Correct command:
  ```bash
  uv run pmtmax backfill-forecasts \
    --markets-path configs/market_inventory/synthetic_historical_snapshots.json \
    --model gfs_seamless --model ecmwf_ifs025 --model kma_gdps \
    --single-run-horizon morning_of --single-run-horizon previous_evening --single-run-horizon market_open \
    --missing-only
  ```
  Model archive start dates (requests before these are silently skipped by `SINGLE_RUN_MODEL_MIN_DATE`):
  - `gfs_seamless`: 2021-07-01
  - `ecmwf_ifs025`: 2024-03-01
  - `kma_gdps`: 2025-01-01

## Agent Assets
- `CLAUDE.md` is a compatibility bridge for Claude and points back to this file.
- `.claude/commands/` contains Claude-specific prompt entrypoints.
- `.claude/skills/` contains Claude project skills.
- `.agents/skills/` contains Codex repo-local skills (mirrors `.claude/skills/`).
- `docs/agent-skills/` contains shared skill reference docs used by both tools.
- `docs/codebase/` is the folder-by-folder map of the codebase.

Recommended shared skills:
- `pmtmax-repo`
- `pmtmax-data-ops`
- `pmtmax-market-rules`
- `pmtmax-research-loop`
- `pmtmax-autoresearch`
- `pmtmax-commit`
- `pmtmax-release-checklist`

## Commit Rules
Use the conventional commit policy in `docs/commit-convention.md`.
Install the `commit-msg` pre-commit hook if you want local enforcement.
