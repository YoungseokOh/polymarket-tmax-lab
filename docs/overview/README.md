# Documentation Map

The `docs/` root is organized by folders only. `AGENTS.md` remains the operating
contract when another agent-facing file disagrees.

## Start Here
- `docs/overview/architecture.md`: end-to-end market -> weather -> dataset -> model -> execution flow.
- `docs/codebase/index.md`: folder-by-folder ownership and edit impact.
- `docs/agent-skills/index.md`: shared Codex/Claude workflow references.
- `docs/markets/market-rules.md`: settlement-source, station, unit, and rule-parsing policy.

## Folder Guides
- `overview/`: high-level documentation map and architecture notes.
- `codebase/`: maps repository folders to responsibilities and change checklists.
- `agent-skills/`: shared skill reference docs mirrored by `.agents/skills/` and
  `.claude/skills/`.
- `../checker/`: agent-maintained markdown status boards and collection runbooks
  for recurring dataset checks.
- `markets/`: market-rule and settlement-source policy.
- `research/`: modeling, backtesting, and paper references.
- `operations/`: live trading, risk, and operating safety.
- `development/`: contribution and commit conventions.
- `references/`: external or long-form reference notes.

## Domain Policy Docs
- `docs/research/modeling.md`: modeling contract, champion selection, and calibration policy.
- `docs/research/backtesting.md`: real-history backtest semantics and promotion-safe metrics.
- `docs/operations/live-trading.md`: live and manual-approval safety rules.
- `docs/operations/risk.md`: risk posture and constraints.
- `docs/research/papers.md`: research references and idea backlog.
- `docs/development/commit-convention.md`: commit-message policy.

## Current Research Source Of Truth
- Training inventory: `configs/market_inventory/full_training_set_snapshots.json`.
- Curated historical backlog: `configs/market_inventory/historical_temperature_snapshots.json`.
- Canonical training parquet: `data/workspaces/historical_real/parquet/gold/historical_training_set.parquet`.
- Canonical official-price panel: `data/workspaces/historical_real/parquet/gold/historical_backtest_panel.parquet`.
- Public champion alias: `artifacts/public_models/champion.*`.

`full_training_set_snapshots.json` is a checked-in training inventory. It is not
an auto-refreshed mirror of the historical backlog. Daily `ops_daily` collection
does not retrain models or mutate `historical_real`; it only records active-market
operational evidence and forward diagnostics.
