# polymarket-tmax-lab

`polymarket-tmax-lab` is a research-first, trading-aware repository for one market family only:

`Highest temperature in [city] on [date]?`

The repo does not implement a generic prediction-market framework first. It models the actual settlement target for recurring Polymarket temperature contracts:

- official source
- official station
- official local date
- official unit
- official binning and finalization rules

The system is designed to work in:

- research mode with only public market data and public weather data
- paper-trading mode without wallet credentials
- live-trading mode only behind explicit feature flags and confirmations

## Why These Markets Are Special
- Settlement is not “city weather” in the generic sense. Markets resolve against a specific source and station, such as Wunderground at Incheon Intl Airport, LaGuardia Airport, Hong Kong Observatory Daily Extract, or CWA Taipei station data.
- Outcome labels are not generic buckets. They encode whole-degree bins, two-degree Fahrenheit ranges, and low/high catch-all bins.
- Forecast quality is not enough by itself. The repo must also parse rules correctly, reconstruct only admissible historical forecasts, and map predictive distributions to exact market outcomes.

## What The Repo Does
- discovers active Polymarket max-temperature markets
- parses rule text into structured `MarketSpec`
- retrieves public weather forecast inputs from Open-Meteo
- backfills bronze/silver market, forecast, and truth tables into DuckDB + Parquet
- reconstructs historical forecast slices without lookahead
- retrieves official truth via source adapters
- trains probabilistic postprocessing models
- maps predictive distributions to Polymarket outcomes
- computes edge versus market prices
- runs rolling-origin research backtests
- runs paper trading with public data
- keeps live trading implemented but disabled by default

## Data Sources
- Polymarket Gamma API for market discovery
- Polymarket public CLOB read endpoints for books and price history
- Polymarket market websocket for live public updates
- Open-Meteo forecast, historical forecast, previous-runs, and ensemble endpoints
- Wunderground station pages
- Hong Kong Observatory open data `CLMMAXT`
- Central Weather Administration official station/CODiS snapshots

## Zero To First Signal
1. Install `uv`.
2. Sync dependencies:

```bash
uv sync --all-extras
```

3. Build a ready-to-use research data environment in one command:

```bash
uv run pmtmax bootstrap-lab
```

4. Train a baseline probabilistic model:

```bash
uv run pmtmax train-baseline --model-name gaussian_emos
```

5. Run a research backtest:

```bash
uv run pmtmax backtest --model-name gaussian_emos
```

6. Emit paper-trading signals:

```bash
uv run pmtmax paper-trader --model-name gaussian_emos
```

## Install
```bash
uv sync --all-extras
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

Python 3.12 is required. The project uses `uv`, `ruff`, `mypy`, `pytest`, and GitHub Actions CI.

## One-Shot Bootstrap
Use the one-shot bootstrap when you want a fresh machine to become experiment-ready with a single command.

```bash
uv run pmtmax bootstrap-lab
```

By default this command:

- archives legacy raw/parquet run directories into `data/archive/legacy-runs/`
- restores from `artifacts/bootstrap/pmtmax_seed.tar.gz` if it exists and the local warehouse is missing
- backfills bundled Seoul, NYC, Hong Kong, and Taipei market history
- materializes both tabular and sequence gold datasets
- writes `artifacts/bootstrap/bootstrap_manifest.json`

If you want to carry data from one machine to another, export a portable seed on the source machine:

```bash
uv run pmtmax export-seed
```

Then copy `artifacts/bootstrap/pmtmax_seed.tar.gz` to the target machine and run:

```bash
uv run pmtmax restore-seed
uv run pmtmax bootstrap-lab
```

## Agent Workflow
- Shared agent guidance: [AGENTS.md](AGENTS.md)
- Claude compatibility bridge: [CLAUDE.md](CLAUDE.md)
- Shared skill references: [docs/agent-skills/](docs/agent-skills)
- Claude project skills: [.claude/skills/](.claude/skills)
- Claude slash-command prompts: [.claude/commands/](.claude/commands)
- Codex repo-local skills: [tools/skills/](tools/skills)

Useful shared skills:
- `pmtmax-repo`
- `pmtmax-data-ops`
- `pmtmax-market-rules`
- `pmtmax-research-loop`
- `pmtmax-commit`
- `pmtmax-release-checklist`

## Codebase Guide
- Folder-by-folder codebase map: [docs/codebase/index.md](docs/codebase/index.md)
- Commit convention: [docs/commit-convention.md](docs/commit-convention.md)

## Training Workflow
```bash
uv run pmtmax bootstrap-lab
uv run pmtmax train-baseline --dataset-path data/parquet/gold/historical_training_set.parquet
uv run pmtmax train-advanced --model-name det2prob_nn
```

`uv run pmtmax build-dataset` remains available as a wrapper that runs the backfill
and materialization steps in one command. Research mode defaults to strict Open-Meteo
archive usage, and `build-dataset` will request exact `single_run` archives for the
selected decision horizons when available. Bundled fixture fallback is demo-only and
must be explicitly enabled via `--no-strict-archive --allow-demo-fixture-fallback`.
The canonical warehouse defaults to `data/duckdb/warehouse.duckdb`, and warehouse
parquet mirrors live under `data/parquet/{bronze,silver,gold}`.

If you prefer step-by-step control instead of the one-shot bootstrap, the lower-level
commands remain available: `init-warehouse`, `backfill-*`, `materialize-training-set`,
and `summarize-forecast-availability`.

## Legacy Cutover
If you have older smoke or experimental DuckDB files under `data/duckdb/`, merge them
into the canonical warehouse with:

```bash
uv run pmtmax migrate-legacy-warehouse --archive-legacy
uv run pmtmax compact-warehouse
```

This writes:

- `data/manifests/legacy_inventory.json`
- `data/manifests/migration_report.json`
- `data/manifests/warehouse_manifest.json`

After a successful cutover, `data/duckdb/` should contain only `warehouse.duckdb`.
Older files are moved to `data/archive/legacy-duckdb/`.
During an active writer command, a temporary `warehouse.duckdb.lock` file may appear
next to the canonical database and disappear when the command exits.

For legacy raw/parquet run directories created before the canonical layout, use:

```bash
uv run pmtmax inventory-legacy-runs
uv run pmtmax archive-legacy-runs --execute
```

## Backtest Workflow
```bash
uv run pmtmax backtest \
  --dataset-path data/parquet/gold/historical_training_set.parquet \
  --model-path artifacts/models/gaussian_emos.pkl \
  --model-name gaussian_emos
```

Outputs are written under `artifacts/`.

## Paper Trading Workflow
```bash
uv run pmtmax paper-trader \
  --model-path artifacts/models/gaussian_emos.pkl \
  --model-name gaussian_emos
```

Paper trading uses active discovered markets when available. If no active max-temperature markets are currently listed on Polymarket, the command exits cleanly with no fills.

## Dry-Run Live Workflow
```bash
uv run pmtmax live-trader \
  --model-path artifacts/models/gaussian_emos.pkl \
  --model-name gaussian_emos \
  --dry-run
```

This performs preflight checks and signed-order previews when a private key is configured. Actual posting remains gated.

## Live Trading Is Gated
Live trading exists for future use but is off by default.

Required flags:

- `PMTMAX_LIVE_TRADING=true`
- `PMTMAX_CONFIRM_LIVE_TRADING=YES_I_UNDERSTAND`

Required credentials:

- `PMTMAX_POLY_PRIVATE_KEY`
- `PMTMAX_POLY_API_KEY`
- `PMTMAX_POLY_API_SECRET`
- `PMTMAX_POLY_PASSPHRASE`
- `PMTMAX_POLY_CHAIN_ID` defaults to `137`
- `PMTMAX_POLY_SIGNATURE_TYPE` and `PMTMAX_POLY_FUNDER_ADDRESS` are optional for proxy or smart-wallet setups

The live broker fails closed if flags or credentials are missing.

## Repo Layout
- `src/pmtmax/backfill/`: bronze/silver/gold backfill orchestration
- `src/pmtmax/markets/`: discovery, filtering, parsing, market specs
- `src/pmtmax/weather/`: Open-Meteo ingestion, truth adapters, feature engineering
- `src/pmtmax/modeling/`: baselines, advanced models, mapping, evaluation
- `src/pmtmax/backtest/`: dataset builder, replay, rolling-origin logic, metrics
- `src/pmtmax/execution/`: edge, fees, slippage, sizing, paper and live brokers
- `docs/`: architecture, papers, modeling, rule families, live-trading, risks
- `docs/codebase/`: folder-by-folder ownership and edit guide
- `tools/skills/pmtmax-repo/`: repo-local skill and references for Codex/Claude workflows

## Known Limitations
- Historical exact L2 orderbook archives are not publicly available for older markets. The repo therefore separates research backtests from exact execution replay.
- The forecast backfill workflow now uses Open-Meteo's generic archive endpoint `historical-forecast-api.open-meteo.com/v1/forecast` for archived model runs. Research backfills default to strict archive mode and do not silently replace missing forecasts with fixtures.
- Bundled Seoul, NYC, Hong Kong, and Taipei forecast fixtures remain available only as an explicit demo fallback via `--no-strict-archive --allow-demo-fixture-fallback`. That fallback is recorded in `bronze_forecast_requests`.
- Open-Meteo Single Runs support exists as an exact-run hook, but the main research pipeline still treats generic archive rows and exact single-run rows as different quality tiers.
- Some model/location pairs are genuinely unsupported by Open-Meteo coverage. In strict mode those rows are skipped and recorded in `bronze_forecast_requests` instead of being silently replaced.
- Bundled historical market snapshots provide a reproducible backtest path for Seoul, NYC, Hong Kong, and Taipei even when no active temperature markets are currently listed.
- The CWA adapter is cache-first and fails closed unless an official snapshot is available locally. It does not substitute another source or station.
- Advanced models beyond the det2prob path are practical public-data approximations of the cited papers, not paper-faithful reproductions of closed or richer operational inputs.
- Firebase sync is a backup mirror for raw/parquet/manifests only. DuckDB remains the local canonical warehouse and is not mirrored.
- Public CLOB endpoint shapes can change; the repo isolates them behind read clients and tests.

## Risk Disclaimer
This repository is for research and systems development. Prediction-market trading involves substantial risk, model risk, execution risk, legal/regulatory risk, data-quality risk, and infrastructure risk. Do not enable live trading unless you understand the code, the exchange mechanics, the settlement rules, and the operational failure modes.
