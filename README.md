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
- Wunderground station pages and Weather.com historical observation endpoints used by those pages
- Hong Kong Observatory open data `CLMMAXT`
- Central Weather Administration official station pages and CODiS station data

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
uv run pmtmax benchmark-models
uv run pmtmax backtest --model-name champion
uv run pmtmax backtest --model-name trading_champion
```

To evaluate against official historical Polymarket prices instead of the synthetic
book, first backfill price history and materialize the decision-time panel, then run:

```bash
uv run pmtmax backfill-price-history --markets-path configs/market_inventory/historical_temperature_snapshots.json
uv run pmtmax materialize-backtest-panel --allow-canonical-overwrite
uv run pmtmax backtest --pricing-source real_history --model-name champion
```

To run a more conservative execution proxy without claiming exact historical bid/ask
replay, use the same panel with `quote_proxy`:

```bash
uv run pmtmax backtest --pricing-source quote_proxy --quote-proxy-half-spread 0.02 --model-name champion
```

6. Emit paper-trading signals:

```bash
uv run pmtmax paper-trader --core-recent-only --model-name trading_champion
```

`paper-trader` defaults to `--horizon policy`, which applies the checked-in
city-specific horizon policy from `configs/recent-core-horizon-policy.yaml`.
Markets outside the recommended horizon set are reported explicitly as
`policy_filtered`. Add `--core-recent-only` when you want the revenue workflow
to stay on `Seoul / NYC / London`.

Paper-only diagnostics can override that policy without touching the live path:

```bash
uv run pmtmax paper-trader \
    --model-name trading_champion \
    --market-scope default \
    --horizon-policy-path configs/paper-all-supported-horizon-policy.yaml
uv run pmtmax paper-multimodel-report --markets-path artifacts/discovered_markets.json
uv run pmtmax execution-sensitivity-report --markets-path artifacts/discovered_markets.json
uv run pmtmax market-bottleneck-report \
    --input-path artifacts/signals/v2/paper_signals.json \
    --opportunity-summary-path artifacts/signals/v2/opportunity_shadow_summary.json \
    --observation-summary-path artifacts/signals/v2/observation_shadow_summary.json
uv run pmtmax execution-watchlist-playbook \
    --champion-bottleneck-path artifacts/signals/v2/market_bottleneck_report__champion_alias.json \
    --challenger-bottleneck-path artifacts/signals/v2/market_bottleneck_report__mega_neighbor_oof.json \
    --fee-watchlist-summary-path artifacts/signals/v2/paper_multimodel/<run_tag>_fee_watchlist/summary.json \
    --policy-watchlist-summary-path artifacts/signals/v2/paper_multimodel/<run_tag>_policy_watchlist/summary.json \
    --sensitivity-summary-path artifacts/signals/v2/execution_sensitivity/<run_tag>/summary.json
```

- `paper-multimodel-report` compares the active champion against the current top challenger pool and writes per-model JSON, `summary.json`, and `leaderboard.csv` under `artifacts/signals/v2/paper_multimodel/`
- `execution-sensitivity-report` sweeps paper-only `min_edge / max_spread_bps / min_liquidity / market_scope / horizon_policy` combinations from `configs/paper-exploration.yaml`
- `market-bottleneck-report` rolls one row-oriented report up into `fee_sensitive_watchlist`, `raw_edge_desert_watchlist`, and `policy_blocked_watchlist`
- `execution-watchlist-playbook` converts those reports into `artifacts/signals/v2/execution_watchlist_playbook.json` plus a Markdown playbook and computes Tier A ask-threshold rules for the dashboard

7. Audit active markets for executable opportunities with explicit book status:

```bash
uv run pmtmax opportunity-report --core-recent-only --model-name trading_champion
```

`opportunity-report`, `opportunity-shadow`, `scan-daemon`, and `live-trader`
use the same policy-aware horizon selection where applicable. `opportunity-report`,
`opportunity-shadow`, `observation-report`, and `observation-shadow` now also
accept `--horizon-policy-path` so paper-only diagnostics can compare the checked-in
recent-core policy against `configs/paper-all-supported-horizon-policy.yaml`
without changing the live defaults.

If you want to run the observation-driven watcher that zeros out already-impossible
lower bins using the strongest target-day lower bound, run:

```bash
uv run pmtmax observation-report --model-name trading_champion
uv run pmtmax observation-shadow --model-name trading_champion --max-cycles 1
uv run pmtmax station-dashboard
uv run pmtmax station-cycle --model-name trading_champion
```

`observation-report` and `observation-shadow` write the latest candidate table,
manual-review alerts, and `live_pilot_queue.json` under `artifacts/signals/v2/`.
The queue is still manual-approval-only; candidates are classified as
`tradable`, `manual_review`, or `blocked`.
The live source stack is layered and fail-closed:
- target-day markets only
- exact-public intraday first when documented (`HKO` text readings for Hong Kong, `CWA CODiS` current-month daily max for Taipei)
- research-public same-airport intraday where documented (`AMO AIR_CALP` for Seoul / RKSI)
- `METAR` fallback after those source-specific paths
`observation_shadow_summary.json` now includes `by_source_family`,
`by_observation_source`, `top_after_cost_edges`, `top_price_vs_observation_gaps`,
`by_reason`, `by_city_reason`, `by_horizon_reason`, and blocker top lists so you
can see which source layer is actually generating after-cost candidates and which
markets are dying on fee/spread/policy.
`station-dashboard` reads the latest opportunity / observation / open-phase / revenue-gate
artifacts plus `execution_watchlist_playbook.json` and writes `artifacts/signals/v2/station_dashboard.json` plus
`artifacts/signals/v2/station_dashboard.html`. `station-dashboard-daemon` keeps the
same outputs refreshed on one host for the Discovery / Observation / Execution view.
When Tier A fee-sensitive rows revisit the same `market_id / outcome_label / horizon`
with `best_ask <= watch_rule_threshold_ask`, the dashboard raises a watchlist alert
instead of changing any live-trading guardrail.
`station-cycle` is the one-shot orchestrator that refreshes opportunity, observation,
open-phase, revenue-gate, and dashboard outputs in sequence. `station-daemon` repeats
that full cycle on an interval.

If you want to test the listing/open-phase hypothesis instead of the near-term
policy path, run:

```bash
uv run pmtmax open-phase-shadow --max-cycles 1
```

`open-phase-shadow` filters active markets by `componentMarkets[*].acceptingOrdersTimestamp`
(falling back to `createdAt` / `deployingTimestamp`) and scores only markets that
opened recently. It defaults to `--horizon market_open` rather than the near-term
policy horizon.

If you want to search for hope outside the recent-core trio and rank fresh
Wunderground-family listings by open-phase age, target-day distance, volume, and
after-cost edge, run:

```bash
uv run pmtmax hope-hunt-report --model-name trading_champion
uv run pmtmax hope-hunt-daemon --model-name trading_champion --max-cycles 1
```

`hope-hunt-report` and `hope-hunt-daemon` default to
`--market-scope supported_wu_open_phase`, which keeps only supported
Wunderground-family `research_public` cities and writes ranked outputs to
`artifacts/signals/v2/hope_hunt_latest.json`,
`artifacts/signals/v2/hope_hunt_history.jsonl`, and
`artifacts/signals/v2/hope_hunt_summary.json`. `opportunity-report`,
`opportunity-shadow`, and `open-phase-shadow` also accept
`--market-scope supported_wu_open_phase` when you want the same filtered market
universe without the ranking wrapper.

To combine recent-core benchmark results and live-path shadow validation into one
promotion decision, run:

```bash
uv run pmtmax revenue-gate-report
```

The revenue gate now also reads `observation_shadow_summary.json` when present, so
benchmark `GO` can be confirmed by the near-term opportunity path, the open-phase
path, or the observation-station path. The default benchmark input is the checked-in
`artifacts/benchmarks/v2/benchmark_summary.json`; if you have a richer recent-core
benchmark summary, pass it explicitly with `--benchmark-summary-path`.

To run the all-supported small live pilot preset with explicit manual approval,
point `PMTMAX_CONFIG` at the checked-in preset and approve one queued candidate:

```bash
PMTMAX_CONFIG=configs/live-pilot-all-supported.yaml uv run pmtmax observation-daemon
PMTMAX_CONFIG=configs/live-pilot-all-supported.yaml uv run pmtmax approve-live-candidate <token> --dry-run
```

The default observation-only preset is `configs/observation-station.yaml`.

The canonical engine is now v2-only:

- grouped backtest splits (`market_day` by default) replace row-level replay
- dataset/model/backtest/signal artifacts live under `data/parquet/gold/v2/`, `artifacts/models/v2/`, `artifacts/backtests/v2/`, and `artifacts/signals/v2/`
- paper/live/opportunity paths require calibrated probabilities and fail closed as `missing_calibrator`

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
- backfills bundled Seoul, NYC, Hong Kong, and Taipei seed snapshots
- materializes both tabular and sequence gold datasets
- writes `artifacts/bootstrap/bootstrap_manifest.json`

`bootstrap-lab` is the reproducible seed/demo path. It does not discover or curate real historical Polymarket event pages by default.

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
- `pmtmax-autoresearch`
- `pmtmax-commit`
- `pmtmax-release-checklist`

## Codebase Guide
- Folder-by-folder codebase map: [docs/codebase/index.md](docs/codebase/index.md)
- Commit convention: [docs/commit-convention.md](docs/commit-convention.md)

## Training Workflow
```bash
uv run pmtmax bootstrap-lab
uv run pmtmax build-dataset --allow-canonical-overwrite
uv run pmtmax materialize-backtest-panel --allow-canonical-overwrite
uv run pmtmax train-baseline --model-name gaussian_emos
uv run pmtmax train-advanced --model-name lgbm_emos --variant recency_neighbor_oof
uv run pmtmax benchmark-models
uv run pmtmax paper-trader --core-recent-only --model-name trading_champion
```

Canonical research paths are now v2-only.
Public model support is `gaussian_emos`, `tuned_ensemble`, `det2prob_nn`, and `lgbm_emos`.
The current active aliases point to `lgbm_emos / recency_neighbor_oof`, while `benchmark-models` still remains the canonical publish path for public aliases.
`benchmark-models` writes the leaderboard under `artifacts/benchmarks/v2/` and publishes both the research `champion` alias and the trading-focused `trading_champion` alias under `artifacts/models/v2/`.
`benchmark-ablations` is internal research tooling for variant-level grouped-holdout diagnostics and writes family-specific leaderboards under `artifacts/benchmarks/v2/`.

## Autoresearch Workflow
`karpathy/autoresearch`-style exploration is wired into the repo as a YAML candidate loop around `lgbm_emos / recency_neighbor_oof`.

```bash
uv run pmtmax autoresearch-init

# Edit one candidate YAML under artifacts/autoresearch/<run_tag>/candidates/
uv run pmtmax autoresearch-step --spec-path artifacts/autoresearch/<run_tag>/candidates/my_candidate.yaml
uv run pmtmax autoresearch-gate --spec-path artifacts/autoresearch/<run_tag>/candidates/my_candidate.yaml
uv run pmtmax autoresearch-analyze-paper --spec-path artifacts/autoresearch/<run_tag>/candidates/my_candidate.yaml
uv run pmtmax autoresearch-promote --spec-path artifacts/autoresearch/<run_tag>/candidates/my_candidate.yaml
```

`scripts/autoresearch.sh` is a thin wrapper over the same commands.
The loop never rewrites canonical datasets or aliases unless promotion explicitly asks to publish them.

## Real Historical Collection
Use the curated inventory workflow when you want real historical temperature markets instead of bundled examples.

If you want one long-running shell entrypoint for the full supported-city batch, use:

```bash
scripts/run_full_historical_batch.sh
```

Useful flags:

```bash
scripts/run_full_historical_batch.sh --city Seoul --max-pages 10
scripts/run_full_historical_batch.sh --skip-refresh --skip-model-smoke
```

Each run writes a timestamped log under `artifacts/batch_logs/`.
`--city`는 refresh/watchlist discovery뿐 아니라 backfill/materialization 입력도 해당 도시로 제한한다.

Closed-event refresh만 장기 배치로 따로 돌리고 싶으면 staged wrapper를 사용한다:

```bash
scripts/run_historical_refresh_pipeline.sh
scripts/run_historical_refresh_pipeline.sh --city London --max-pages 10 --max-events 50
scripts/run_historical_refresh_pipeline.sh --stage classify --status-filter truth_source_lag --truth-no-cache --truth-per-source-limit 1
scripts/run_historical_refresh_pipeline.sh --fill-gaps-only --checkpoint-every 1
```

이 wrapper는 `discover -> fetch-pages -> classify -> publish`를 분리해서 실행할 수 있고, 기본은 `--resume`라서 기존 manifest를 이어받아 partial progress를 보존한다.
`fetch-pages`는 bounded concurrency를 사용하고, `classify`는 exact-source truth probe를 source family별 제한과 함께 병렬화한다.
`--checkpoint-every`를 주면 fetch/classify manifest를 배치 단위로 계속 저장하고, `--fill-gaps-only`는 기존 candidate manifest를 재사용해서 retryable gap만 다시 classify한다.
현재 retryable 상태는 `not_closed`, `not_historical`, `truth_source_lag`, `truth_parse_failed`, `truth_request_failed`다. 각 URL은 한 번의 refresh invocation에서 한 번만 재시도한다.

1. Refresh the closed-event source URL manifest from grouped Polymarket weather events:

```bash
uv run python scripts/refresh_historical_event_urls.py
```

This staged pipeline discovers supported closed grouped events, persists the candidate backlog to `data/manifests/historical_event_candidates.json`, persists page-fetch state to `data/manifests/historical_event_page_fetches.json`, classifies each fetched event into collection statuses in `data/manifests/historical_collection_status.json`, and appends only `collected` URLs to `configs/market_inventory/historical_temperature_event_urls.json`.
`truth_source_lag` and `truth_request_failed` remain retryable manifest states instead of failing the whole refresh.

2. Build or refresh the curated inventory from the checked-in event URL manifest:

```bash
uv run python scripts/build_historical_market_inventory.py
uv run python scripts/validate_historical_market_inventory.py
uv run pmtmax collection-preflight --markets-path configs/market_inventory/historical_temperature_snapshots.json
```

`collection-preflight` now separates exact-public and research-public truth tracks. Wunderground-family markets default to the same-airport public research path. Seoul / RKSI uses AMO `AIR_CALP`, London / EGLC and NYC / KLGA use the Wunderground public historical API for the same station, and other expansion-city mappings may use NOAA Global Hourly when `station_catalog.json` documents that same-airport public path. `PMTMAX_WU_API_KEY` is optional and only used when you want to force an explicit same-source audit key instead of the documented public research path.
When you run `build_historical_market_inventory.py` against the canonical checked-in manifests, it also syncs `data/manifests/historical_collection_status.json` so the `collected` count matches the current curated snapshot inventory.
Wunderground-family truth probes are sensitive to source-family concurrency. Keep the default `--truth-per-source-limit 1`, and add `--truth-no-cache` when repairing lagged or malformed truth payloads.
If `backfill-truth` reports `lag` rows or `materialize-training-set` fails with a public archive lag message, run `uv run pmtmax summarize-truth-coverage` to inspect the latest archive date NOAA advertised for each lagged station.
The default research CLI no longer reads `tests/fixtures/truth`; fixture truth remains test/demo-only unless you wire it explicitly in code.
`build_historical_market_inventory.py` now filters the canonical snapshot output down to truth-ready markets only. Lagged or blocked URLs stay out of `historical_temperature_snapshots.json` and are recorded in `historical_inventory_build_report.json` issue counts instead. Validation results are written separately to `historical_inventory_validate_report.json`.
The repo also ships a recent 3-city benchmark for official history evaluation in `configs/market_inventory/recent_core_temperature_event_urls.json` and `configs/market_inventory/recent_core_temperature_snapshots.json`.
For a reproducible lightweight rerun of that benchmark, use `configs/recent-core-benchmark.yaml`, `configs/recent-core-horizon-policy.yaml`, and `scripts/run_recent_core_benchmark.py`.
If you want a home-machine wrapper that first repairs the local canonical dataset/model and then runs the benchmark, use:

```bash
scripts/run_recent_core_benchmark_local.sh
scripts/run_recent_core_benchmark_local.sh --city Seoul
```

3. Start from a clean canonical warehouse if you want to replace the existing seed data.

4. Backfill from the curated snapshot inventory:

```bash
uv run pmtmax init-warehouse
uv run pmtmax backfill-markets --markets-path configs/market_inventory/historical_temperature_snapshots.json
uv run pmtmax backfill-forecasts \
  --markets-path configs/market_inventory/historical_temperature_snapshots.json \
  --strict-archive \
  --single-run-horizon market_open \
  --single-run-horizon previous_evening \
  --single-run-horizon morning_of
uv run pmtmax backfill-truth --markets-path configs/market_inventory/historical_temperature_snapshots.json --truth-no-cache
uv run pmtmax summarize-truth-coverage
uv run pmtmax summarize-dataset-readiness --markets-path configs/market_inventory/historical_temperature_snapshots.json
uv run pmtmax materialize-training-set \
  --markets-path configs/market_inventory/historical_temperature_snapshots.json \
  --decision-horizon market_open \
  --decision-horizon previous_evening \
  --decision-horizon morning_of \
  --allow-canonical-overwrite
uv run pmtmax backfill-price-history --markets-path configs/market_inventory/historical_temperature_snapshots.json
uv run pmtmax materialize-backtest-panel \
  --dataset-path data/parquet/gold/v2/historical_training_set.parquet \
  --markets-path configs/market_inventory/historical_temperature_snapshots.json \
  --allow-canonical-overwrite
uv run pmtmax summarize-price-history-coverage --markets-path configs/market_inventory/historical_temperature_snapshots.json
uv run pmtmax summarize-forecast-availability
uv run pmtmax compact-warehouse
```

Price-history note: the public CLOB `/prices-history` endpoint is retention-limited in practice. Once older official-history payloads have been captured into `data/raw/bronze`, `bronze_price_history_requests`, and `silver_price_timeseries`, prefer re-materializing `gold_backtest_panel` and `artifacts/price_history_coverage.json` from the archived warehouse. A late uncached refetch can return an empty history for markets that previously had archived points, which reduces request-level coverage without improving research fidelity.

The checked-in starter source URLs live in `configs/market_inventory/historical_temperature_event_urls.json`, and the generated curated `MarketSnapshot[]` inventory lives in `configs/market_inventory/historical_temperature_snapshots.json`.
The staged closed-event manifests live in:

- `data/manifests/historical_event_candidates.json`
- `data/manifests/historical_event_page_fetches.json`
- `data/manifests/historical_collection_status.json`

5. Refresh the active supported-city watchlist when you want the next candidate batch for curation:

```bash
uv run python scripts/build_active_weather_watchlist.py
```

This writes `configs/market_inventory/active_temperature_watchlist.json` without mutating the canonical warehouse.

`uv run pmtmax build-dataset` remains available as a wrapper that runs the backfill
and materialization steps in one command. Research mode defaults to strict Open-Meteo
archive usage, and `build-dataset` will request exact `single_run` archives for the
selected decision horizons when available. Bundled fixture fallback is demo-only and
must be explicitly enabled via `--no-strict-archive --allow-demo-fixture-fallback`.
If the canonical gold already exists, `build-dataset` also requires
`--allow-canonical-overwrite`; otherwise use a variant `--output-name`.
The canonical warehouse defaults to `data/duckdb/warehouse.duckdb`, and warehouse
parquet mirrors live under `data/parquet/{bronze,silver,gold}`.
Canonical `gold/v2/historical_training_set*` and `gold/v2/historical_backtest_panel`
are immutable by default. Use variant `--output-name` values for experiments, and only
pass `--allow-canonical-overwrite` when intentionally promoting a rebuilt canonical
dataset. The writer now snapshots the previous parquet + manifest under
`artifacts/recovery/` first.

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
  --dataset-path data/parquet/gold/v2/historical_training_set.parquet \
  --model-name champion
```

This default path uses the synthetic research book. To evaluate the same historical
dataset against official Polymarket prices, build the panel first:

```bash
uv run pmtmax backfill-price-history --markets-path configs/market_inventory/historical_temperature_snapshots.json
uv run pmtmax materialize-backtest-panel \
  --dataset-path data/parquet/gold/v2/historical_training_set.parquet \
  --markets-path configs/market_inventory/historical_temperature_snapshots.json \
  --allow-canonical-overwrite
uv run pmtmax backtest \
  --dataset-path data/parquet/gold/v2/historical_training_set.parquet \
  --panel-path data/parquet/gold/v2/historical_backtest_panel.parquet \
  --pricing-source real_history \
  --model-name champion
```

If you want an execution-aware proxy on top of the same official last-price panel,
switch the pricing source and set an explicit half-spread penalty:

```bash
uv run pmtmax backtest \
  --dataset-path data/parquet/gold/v2/historical_training_set.parquet \
  --panel-path data/parquet/gold/v2/historical_backtest_panel.parquet \
  --pricing-source quote_proxy \
  --quote-proxy-half-spread 0.02 \
  --model-name champion
```

Synthetic outputs are written to `artifacts/backtests/v2/backtest_metrics.json` and
`artifacts/backtests/v2/backtest_trades.json`. Official-price runs write
`artifacts/backtests/v2/backtest_metrics_real_history.json` and
`artifacts/backtests/v2/backtest_trades_real_history.json`. Quote-proxy runs write
`artifacts/backtests/v2/backtest_metrics_quote_proxy.json` and
`artifacts/backtests/v2/backtest_trades_quote_proxy.json`.

Backtests record `contract_version`, `split_policy`, and `leakage_audit_passed`
in the metrics artifact.

To rerun the current recent `Seoul` / `NYC` / `London` benchmark end-to-end into isolated per-city directories, use:

```bash
uv run python scripts/run_recent_core_benchmark.py
```

The runner writes per-city metrics plus `city x horizon` real-versus-quote-proxy deltas into `recent_core_benchmark_summary.json`. It also applies `configs/recent-core-horizon-policy.yaml`, records policy-filtered metrics for the currently recommended horizons, and adds both top-level aggregate profitability fields and nested `cities.<city>.horizons.<horizon>` summaries. Use `--reuse-existing` when you only want to recompute the summary from existing city runs.

## Paper Trading Workflow
```bash
uv run pmtmax benchmark-models
uv run pmtmax paper-trader --core-recent-only --model-name trading_champion
uv run pmtmax revenue-gate-report
```

Paper trading uses active discovered markets when available. If no active max-temperature markets are currently listed on Polymarket, the command exits cleanly with no fills.
`paper-trader`, `opportunity-report`, `paper-mm`, `live-trader`, and `live-mm` now use real CLOB books by default and report missing books explicitly instead of silently fabricating synthetic liquidity.
`paper-trader`, `live-trader`, `scan-daemon`, `opportunity-report`, and `opportunity-shadow`
also default to the checked-in recent horizon policy. The current recommended
set is `Seoul=market_open+previous_evening+morning_of`,
`NYC=market_open+previous_evening`, and `London=previous_evening`.
Paper/live/opportunity paths use the v2 forecast contract, write outputs under `artifacts/signals/v2/`, and reject uncalibrated forecasts as `missing_calibrator`. For revenue-first operation, the preferred loop is `benchmark-models -> paper/opportunity/open-phase shadow -> revenue-gate-report`.
If recent-core remains `NO_GO`, the exploratory follow-up loop is
`hope-hunt-report -> hope-hunt-daemon` on `supported_wu_open_phase` rather than
loosening the live gate.

## Opportunity Workflow
```bash
uv run pmtmax opportunity-report --core-recent-only --model-name trading_champion
```

This writes `artifacts/signals/v2/opportunity_report.json` and separates `tradable`,
`missing_book`, `raw_gap_non_positive`, `fee_killed_edge`,
`slippage_killed_edge`, `after_cost_positive_but_spread_too_wide`, and other
skip reasons so “no trade” and “no live book” are not conflated.

Opportunity evaluation now uses token-specific CLOB `fee-rate` when available
and a depth walk over visible book levels instead of a flat half-spread penalty.

If you want to validate whether the current live opportunity logic ever becomes
tradable over time before wiring alerts, run the shadow watcher:

```bash
uv run pmtmax opportunity-shadow \
  --core-recent-only \
  --model-name trading_champion \
  --interval 60
```

For a home-machine operational entrypoint that first ensures the canonical
dataset/model exist locally and then starts the watcher, use:

```bash
scripts/run_opportunity_shadow_watch.sh --max-cycles 1
scripts/run_opportunity_shadow_watch.sh --city Seoul --interval 60
```

This keeps a near-term (`today` / `tomorrow` in each market timezone) append-only
 audit trail in `artifacts/signals/v2/opportunity_shadow.jsonl`, plus latest and summary views
under `artifacts/signals/v2/opportunity_shadow_latest.json` and
`artifacts/signals/v2/opportunity_shadow_summary.json`.

If you want to watch the hypothesis that taker alpha exists right after listing,
use the open-phase watcher instead:

```bash
uv run pmtmax open-phase-shadow \
  --market-scope supported_wu_open_phase \
  --model-name trading_champion \
  --open-window-hours 24 \
  --interval 60
```

This writes `artifacts/signals/v2/open_phase_shadow.jsonl`, `artifacts/signals/v2/open_phase_shadow_latest.json`,
and `artifacts/signals/v2/open_phase_shadow_summary.json`. The watcher keys off
`componentMarkets[*].acceptingOrdersTimestamp` when available and falls back to
market creation/deploy timestamps, so it can test whether spreads or raw gaps
look different immediately after listing.

For a ranked, no-order search loop over the same market family, use:

```bash
uv run pmtmax hope-hunt-report --model-name trading_champion
uv run pmtmax hope-hunt-daemon --model-name trading_champion --interval 300
```

This keeps a scope-limited candidate board for supported WU-family active
markets and marks alerts only when `after_cost_edge > 0` or a fresh listing is
blocked solely by spread width.

## Dry-Run Live Workflow
```bash
uv run pmtmax live-trader \
  --core-recent-only \
  --model-name trading_champion \
  --dry-run
```

This performs preflight checks and signed-order previews when a private key is configured. Actual posting remains gated.
Dry-run live paths and market-making previews fail closed on missing CLOB books.
For the conservative small-cap pilot preset, point `PMTMAX_CONFIG` at `configs/revenue-pilot-core.yaml` and keep the explicit live env flags enabled separately.

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
`live-mm` also fails closed if it cannot cancel existing orders before refreshing quotes.

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
- Active grouped-event discovery now follows Polymarket `weather` and `temperature` event tags instead of relying on `/markets` pagination alone.
- The checked-in station catalog covers the current active airport-city universe, with Seoul, NYC, and London marked as the core trading cities.
- Wunderground-family markets keep their official station/source metadata, but default research truth collection uses documented public same-airport paths. Seoul / RKSI uses Korea's Aviation Meteorological Office `AIR_CALP` daily-extremes feed. London / EGLC and NYC / KLGA use the Wunderground public historical API for the same airport station. Some expansion-city WU mappings use NOAA Global Hourly when the checked-in station catalog points to that same-airport public archive. `PMTMAX_WU_API_KEY` is optional for explicit same-source audit collection only.
- The CWA adapter is cache-first but can use the official CODiS station API as an exact-source override for Taipei station data. It still does not substitute another source or station.
- Advanced models beyond the det2prob path are practical public-data approximations of the cited papers, not paper-faithful reproductions of closed or richer operational inputs.
- Firebase sync is a backup mirror for raw/parquet/manifests only. DuckDB remains the local canonical warehouse and is not mirrored.
- Public CLOB endpoint shapes can change; the repo isolates them behind read clients and tests.

## Risk Disclaimer
This repository is for research and systems development. Prediction-market trading involves substantial risk, model risk, execution risk, legal/regulatory risk, data-quality risk, and infrastructure risk. Do not enable live trading unless you understand the code, the exchange mechanics, the settlement rules, and the operational failure modes.
