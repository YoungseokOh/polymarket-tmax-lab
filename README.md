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

3. Optionally build a weather-only pretrain set. This uses real station/date
   weather rows and never writes Polymarket market ids, prices, rules, or CLOB
   books into `weather_train`:

```bash
scripts/pmtmax-workspace weather_train uv run pmtmax collect-weather-training \
  --station-catalog configs/market_inventory/station_catalog.json \
  --date-from 2024-01-01 \
  --date-to 2024-01-07 \
  --model gfs_seamless \
  --http-timeout-seconds 15 \
  --http-retries 1 \
  --max-consecutive-429 2 \
  --missing-only
scripts/pmtmax-workspace weather_train uv run pmtmax train-weather-pretrain --model-name gaussian_emos
```

`collect-weather-training` prints one station/date progress line to stderr by default so long Open-Meteo runs can be monitored without breaking the final JSON on stdout. Use `--no-progress` only when you need a quiet stderr stream for wrapper scripts. The default `--max-consecutive-429 2` treats two consecutive Open-Meteo `429` responses as a free-path daily-limit hit and cancels the remaining batch.

For repeated older-gap backfill, use the queue agent instead of manually
advancing the next week yourself. It reads `checker/weather_train_status.md`,
runs the next `7`-day chunk, appends the collection log, refreshes the status
board, and records `rate-limit-cancelled` when the two-consecutive-`429` rule
fires:

```bash
scripts/pmtmax-workspace weather_train uv run python scripts/run_weather_train_queue_agent.py
```

By default the queue agent also refreshes the `gaussian_emos` weather pretrain
artifact automatically whenever the current `weather_train` gold row count is
at least `500` rows ahead of the latest pretrain metadata. Override with
`--pretrain-refresh-threshold-rows`.

Current observed `weather_train` state on April 25, 2026:
- see `checker/weather_train_status.md` for the current row count, coverage
  ranges, and next queue start
- `2025-01-07..2025-01-27` and `2026-01-22..2026-01-28` are retry-only/free-path
  daily-limit ranges
- older backfill should continue through the queue agent only after cooldown or
  an API-key path when consecutive `429` appears

4. Build the Polymarket real-only research dataset:

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax trust-check \
  --markets-path configs/market_inventory/full_training_set_snapshots.json
scripts/pmtmax-workspace historical_real uv run pmtmax build-dataset \
  --markets-path configs/market_inventory/full_training_set_snapshots.json \
  --forecast-missing-only \
  --allow-canonical-overwrite
scripts/pmtmax-workspace historical_real uv run pmtmax materialize-backtest-panel \
  --markets-path configs/market_inventory/full_training_set_snapshots.json \
  --allow-canonical-overwrite
```

5. Train a baseline probabilistic model:

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax train-baseline --model-name gaussian_emos
```

For `lgbm_emos`, `--pretrained-weather-model` injects `weather_pretrain_*`
prediction features into the market model and stores a wrapped model that
applies the same augmentation at prediction time. Treat this as an experiment:
the first `high_neighbor_oof` quick eval with active injection worsened
`CRPS_C` from `0.8004` to `0.8241`; the `delta_only` follow-up still worsened
to `0.8111`, so promotion still needs a candidate-specific gate.

6. Run a research backtest:

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax benchmark-models --retrain-stride 30
scripts/pmtmax-workspace historical_real uv run pmtmax backtest --pricing-source real_history --model-name champion
```

Backtests use official historical Polymarket prices. If the price panel is stale
or missing, refresh it first:

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax backfill-price-history \
  --markets-path configs/market_inventory/full_training_set_snapshots.json \
  --only-missing \
  --price-no-cache
scripts/pmtmax-workspace historical_real uv run pmtmax materialize-backtest-panel --markets-path configs/market_inventory/full_training_set_snapshots.json --allow-canonical-overwrite
scripts/pmtmax-workspace historical_real uv run pmtmax backtest --pricing-source real_history --model-name champion
```

For repeated daily recovery, use the historical price agent instead of manually
tracking `--offset-markets` yourself. It reads
`checker/historical_price_status.md`, advances the next shard, rebuilds the
canonical panel, refreshes the latest coverage summary, and appends the daily
checker log:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_historical_price_recovery_agent.py
```

By default it processes one shard (`25` markets) per run so the queue can be
tracked day by day. The matching runbook is `checker/historical_price_runbook.md`.

For baseline retraining plus recurring autoresearch, use the model research
agent:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_model_research_agent.py
```

It reuses the current dataset/panel signatures, retrains the baseline only when
needed, auto-creates the next small `lgbm_emos` YAML candidate when the queue
is empty, processes one candidate through `autoresearch-step -> gate -> paper -> promote`,
and updates:
- `checker/model_research_status.md`
- `checker/model_research_log.md`
- `checker/model_research_runbook.md`

Public `champion` publish stays disabled unless you explicitly pass
`--enable-publish` together with a candidate-specific recent-core `GO` summary path.

To run a diagnostic execution proxy without claiming exact historical bid/ask
replay, use the same panel with `quote_proxy`. Do not use `quote_proxy` as a
champion promotion source:

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax backtest --pricing-source quote_proxy --quote-proxy-half-spread 0.02 --model-name champion
```

7. Emit paper-trading signals:

```bash
scripts/pmtmax-workspace ops_daily uv run pmtmax paper-trader --core-recent-only --model-name champion
```

`paper-trader` defaults to `--horizon policy`, which applies the checked-in
city-specific horizon policy from `configs/recent-core-horizon-policy.yaml`.
Markets outside the recommended horizon set are reported explicitly as
`policy_filtered`. Add `--core-recent-only` when you want the revenue workflow
to stay on `Seoul / NYC / London`.
Paper rows are accepted only from real CLOB books or explicit `missing_book`
states; legacy `fixture` book sources are rejected.

Paper-only diagnostics can override that policy without touching the live path:

```bash
uv run pmtmax paper-trader \
    --model-name champion \
    --market-scope default \
    --horizon-policy-path configs/paper-all-supported-horizon-policy.yaml
uv run pmtmax paper-multimodel-report --markets-path artifacts/discovered_markets.json
uv run pmtmax execution-sensitivity-report --markets-path artifacts/discovered_markets.json
uv run pmtmax market-bottleneck-report \
    --input-path artifacts/workspaces/ops_daily/signals/v2/paper_signals.json \
    --opportunity-summary-path artifacts/workspaces/ops_daily/signals/v2/opportunity_shadow_summary.json \
    --observation-summary-path artifacts/workspaces/ops_daily/signals/v2/observation_shadow_summary.json
uv run pmtmax execution-watchlist-playbook \
    --champion-bottleneck-path artifacts/workspaces/ops_daily/signals/v2/market_bottleneck_report__champion_alias.json \
    --challenger-bottleneck-path artifacts/workspaces/ops_daily/signals/v2/market_bottleneck_report__mega_neighbor_oof.json \
    --fee-watchlist-summary-path artifacts/workspaces/ops_daily/signals/v2/paper_multimodel/<run_tag>_fee_watchlist/summary.json \
    --policy-watchlist-summary-path artifacts/workspaces/ops_daily/signals/v2/paper_multimodel/<run_tag>_policy_watchlist/summary.json \
    --sensitivity-summary-path artifacts/workspaces/ops_daily/signals/v2/execution_sensitivity/<run_tag>/summary.json
```

- `paper-multimodel-report` compares the active champion against the current top challenger pool and writes per-model JSON, `summary.json`, and `leaderboard.csv` under `artifacts/workspaces/ops_daily/signals/v2/paper_multimodel/`
- `execution-sensitivity-report` sweeps paper-only `min_edge / max_spread_bps / min_liquidity / market_scope / horizon_policy` combinations from `configs/paper-exploration.yaml`
- `market-bottleneck-report` rolls one row-oriented report up into `fee_sensitive_watchlist`, `raw_edge_desert_watchlist`, and `policy_blocked_watchlist`
- `execution-watchlist-playbook` converts those reports into `artifacts/workspaces/ops_daily/signals/v2/execution_watchlist_playbook.json` plus a Markdown playbook and computes Tier A ask-threshold rules for the dashboard

8. Audit active markets for executable opportunities with explicit book status:

```bash
scripts/pmtmax-workspace ops_daily uv run pmtmax opportunity-report --core-recent-only --model-name champion
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
scripts/pmtmax-workspace ops_daily uv run pmtmax observation-report --model-name champion
scripts/pmtmax-workspace ops_daily uv run pmtmax observation-shadow --model-name champion --max-cycles 1
uv run pmtmax station-dashboard
scripts/pmtmax-workspace ops_daily uv run pmtmax station-cycle --model-name champion
```

`observation-report` and `observation-shadow` write the latest candidate table,
manual-review alerts, and `live_pilot_queue.json` under `artifacts/workspaces/ops_daily/signals/v2/`.
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
artifacts plus `execution_watchlist_playbook.json` and writes `artifacts/workspaces/ops_daily/signals/v2/station_dashboard.json` plus
`artifacts/workspaces/ops_daily/signals/v2/station_dashboard.html`. `station-dashboard-daemon` keeps the
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
scripts/pmtmax-workspace ops_daily uv run pmtmax hope-hunt-report --model-name champion
scripts/pmtmax-workspace ops_daily uv run pmtmax hope-hunt-daemon --model-name champion --max-cycles 1
```

`hope-hunt-report` and `hope-hunt-daemon` default to
`--market-scope supported_wu_open_phase`, which keeps only supported
Wunderground-family `research_public` cities and writes ranked outputs to
`artifacts/workspaces/ops_daily/signals/v2/hope_hunt_latest.json`,
`artifacts/workspaces/ops_daily/signals/v2/hope_hunt_history.jsonl`, and
`artifacts/workspaces/ops_daily/signals/v2/hope_hunt_summary.json`. `opportunity-report`,
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
`artifacts/workspaces/historical_real/benchmarks/v2/benchmark_summary.json`; if you have a richer recent-core
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
- dataset/model/backtest/signal artifacts live under `data/workspaces/<workspace>/parquet/gold/`, `artifacts/workspaces/<workspace>/models/v2/`, `artifacts/workspaces/historical_real/backtests/v2/`, and `artifacts/workspaces/ops_daily/signals/v2/`
- paper/live/opportunity paths require calibrated probabilities and fail closed as `missing_calibrator`
- run `uv run pmtmax trust-check --markets-path <inventory>` before long rebuilds or canonical overwrites; synthetic inventories, `synthetic_` market ids, fixture forecasts, and fabricated books fail closed

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
- Codex repo-local skills: [.agents/skills/](.agents/skills)

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
- Commit convention: [docs/development/commit-convention.md](docs/development/commit-convention.md)

## Training Workflow
```bash
scripts/pmtmax-workspace historical_real uv run pmtmax trust-check --markets-path configs/market_inventory/full_training_set_snapshots.json
scripts/pmtmax-workspace historical_real uv run pmtmax build-dataset --markets-path configs/market_inventory/full_training_set_snapshots.json --forecast-missing-only --allow-canonical-overwrite
scripts/pmtmax-workspace historical_real uv run pmtmax materialize-backtest-panel --markets-path configs/market_inventory/full_training_set_snapshots.json --allow-canonical-overwrite
scripts/pmtmax-workspace historical_real uv run pmtmax train-baseline --model-name gaussian_emos
scripts/pmtmax-workspace historical_real uv run pmtmax train-advanced --model-name lgbm_emos --variant high_neighbor_oof
scripts/pmtmax-workspace historical_real uv run pmtmax benchmark-models --retrain-stride 30
scripts/run_recent_core_benchmark_local.sh --model-name lgbm_emos --variant high_neighbor_oof --retrain-stride 1
uv run pmtmax publish-champion /path/to/workspace/model.pkl --recent-core-summary-path artifacts/workspaces/recent_core_eval/recent_core_benchmark/recent_core_benchmark_summary.json
scripts/pmtmax-workspace ops_daily uv run pmtmax paper-trader --core-recent-only --model-name champion
```

Canonical research paths are now v2-only.
Public model support is `gaussian_emos`, `tuned_ensemble`, `det2prob_nn`, and `lgbm_emos`.
The single public alias lives in `artifacts/public_models/champion.*`.
`benchmark-models` writes workspace-local leaderboards only. Public alias promotion now goes through `publish-champion` after the recent-core benchmark summary is `GO`.
`benchmark-ablations` is internal research tooling for variant-level grouped-holdout diagnostics and writes family-specific leaderboards under `artifacts/workspaces/historical_real/benchmarks/v2/`.

## Autoresearch Workflow
`karpathy/autoresearch`-style exploration is wired into the repo as a YAML candidate loop around `lgbm_emos`.
Run it through `historical_real`; the default v2 path may contain quarantined legacy mixed/synthetic artifacts and is not a promotion source.

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax autoresearch-init --baseline-variant high_neighbor_oof

# Edit one candidate YAML under artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/
scripts/pmtmax-workspace historical_real uv run pmtmax autoresearch-step --spec-path artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/my_candidate.yaml
scripts/pmtmax-workspace historical_real uv run pmtmax autoresearch-gate --spec-path artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/my_candidate.yaml
scripts/pmtmax-workspace historical_real uv run pmtmax autoresearch-analyze-paper --spec-path artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/my_candidate.yaml
scripts/pmtmax-workspace historical_real uv run pmtmax autoresearch-promote --spec-path artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/my_candidate.yaml
```

`scripts/autoresearch.sh` is a thin wrapper over the same commands.
The loop never rewrites canonical datasets or public aliases. Promotion copies the winning YAML only, and now requires gate leaderboard artifacts, matching dataset/panel signatures, candidate calibrator, and paper `overall_gate_decision=GO`; `INCONCLUSIVE` is not promotable. Public alias changes still require `publish-champion`.

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
If `backfill-truth` reports `lag` rows or `materialize-training-set` fails with a public archive lag message, run `scripts/pmtmax-workspace historical_real uv run pmtmax summarize-truth-coverage` to inspect the latest archive date NOAA advertised for each lagged station.
The default research CLI no longer reads `tests/fixtures/truth`; fixture truth remains test/demo-only unless you wire it explicitly in code.
`build_historical_market_inventory.py` now filters the canonical snapshot output down to truth-ready markets only. Lagged or blocked URLs stay out of `historical_temperature_snapshots.json` and are recorded in `historical_inventory_build_report.json` issue counts instead. Validation results are written separately to `historical_inventory_validate_report.json`.
The repo also ships a recent 3-city benchmark for official history evaluation in `configs/market_inventory/recent_core_temperature_event_urls.json` and `configs/market_inventory/recent_core_temperature_snapshots.json`.
For a reproducible lightweight rerun of that benchmark, use `configs/recent-core-benchmark.yaml`, `configs/recent-core-horizon-policy.yaml`, and `scripts/run_recent_core_benchmark.py`.
If you want a home-machine wrapper that first repairs the local canonical dataset/model and then runs the benchmark, use:

```bash
scripts/run_recent_core_benchmark_local.sh
scripts/run_recent_core_benchmark_local.sh --city Seoul
scripts/run_recent_core_benchmark_local.sh --model-name lgbm_emos --variant high_neighbor_oof --retrain-stride 1
scripts/pmtmax-workspace recent_core_eval uv run python scripts/run_recent_core_benchmark.py \
  --model-name lgbm_emos --variant high_neighbor_oof --retrain-stride 10 --backtest-last-n 60 \
  --prebuilt-dataset-path data/workspaces/historical_real/parquet/gold/historical_training_set.parquet \
  --prebuilt-panel-path data/workspaces/historical_real/parquet/gold/historical_backtest_panel.parquet \
  --prebuilt-last-n-market-days 90
```

3. Start from a clean canonical warehouse if you want to replace the existing seed data.

4. Backfill from the curated snapshot inventory:

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax init-warehouse
scripts/pmtmax-workspace historical_real uv run pmtmax backfill-markets --markets-path configs/market_inventory/historical_temperature_snapshots.json
scripts/pmtmax-workspace historical_real uv run pmtmax backfill-forecasts \
  --markets-path configs/market_inventory/historical_temperature_snapshots.json \
  --model ecmwf_ifs025 \
  --strict-archive \
  --missing-only \
  --single-run-horizon market_open \
  --single-run-horizon previous_evening \
  --single-run-horizon morning_of \
  --max-consecutive-429 2
scripts/pmtmax-workspace historical_real uv run pmtmax backfill-truth --markets-path configs/market_inventory/historical_temperature_snapshots.json --truth-no-cache
scripts/pmtmax-workspace historical_real uv run pmtmax summarize-truth-coverage
scripts/pmtmax-workspace historical_real uv run pmtmax summarize-dataset-readiness --markets-path configs/market_inventory/historical_temperature_snapshots.json
scripts/pmtmax-workspace historical_real uv run pmtmax materialize-training-set \
  --markets-path configs/market_inventory/historical_temperature_snapshots.json \
  --model ecmwf_ifs025 \
  --model ecmwf_aifs025_single \
  --model kma_gdps \
  --model gfs_seamless \
  --decision-horizon market_open \
  --decision-horizon previous_evening \
  --decision-horizon morning_of \
  --allow-canonical-overwrite
scripts/pmtmax-workspace historical_real uv run pmtmax backfill-price-history \
  --markets-path configs/market_inventory/historical_temperature_snapshots.json \
  --only-missing \
  --price-no-cache \
  --limit-markets 25
scripts/pmtmax-workspace historical_real uv run pmtmax materialize-backtest-panel \
  --dataset-path data/workspaces/historical_real/parquet/gold/historical_training_set.parquet \
  --markets-path configs/market_inventory/historical_temperature_snapshots.json \
  --allow-canonical-overwrite
scripts/pmtmax-workspace historical_real uv run pmtmax summarize-price-history-coverage --markets-path configs/market_inventory/historical_temperature_snapshots.json
scripts/pmtmax-workspace historical_real uv run pmtmax summarize-forecast-availability
scripts/pmtmax-workspace historical_real uv run pmtmax compact-warehouse
```

If you are topping off an existing warehouse and only want forecast keys that are
absent from `bronze_forecast_requests`, add `--missing-only` to
`backfill-forecasts`. `scripts/run_full_historical_batch.sh` now uses this
incremental forecast top-off by default.
Keep `--max-consecutive-429 2` on free-path Open-Meteo forecast backfills. Two
consecutive `429` responses are treated as a daily-limit hit: the run flushes
already collected rows, cancels, and should be recorded in checker state rather
than retried immediately. For multi-source variants, pass repeated `--model`
values to `backfill-forecasts`, `build-dataset`, or `materialize-training-set`;
otherwise the active config can materialize the same markets as a GFS-only
feature set.

Price-history note: the public CLOB `/prices-history` endpoint is retention-limited in practice. Use `--only-missing --price-no-cache` for gap recovery so tokens that already have archived official price points in `silver_price_timeseries` are not overwritten by late empty responses. Add `--limit-markets` / `--offset-markets` or `--target-date-from` / `--target-date-to` for small checkpointable batches. Once older official-history payloads have been captured into `data/raw/bronze`, `bronze_price_history_requests`, and `silver_price_timeseries`, prefer re-materializing `gold_backtest_panel` and `artifacts/price_history_coverage.json` from the archived warehouse.

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
selected decision horizons when available. Bundled forecast fixtures are test/demo-only
and are rejected by real-only dataset and forecast backfill commands.
`build-dataset` and `materialize-training-set` accept repeated `--model` values;
use them explicitly for multi-source experiments because config defaults may only
include `gfs_seamless`.
If the canonical gold already exists, `build-dataset` also requires
`--allow-canonical-overwrite`; otherwise use a variant `--output-name`.
`bootstrap-lab` accepts the same forecast-top-off shortcut via
`--forecast-missing-only` when you want a seed/bootstrap refresh to reuse
existing forecast request keys instead of refetching them.
The canonical historical research warehouse lives under
`data/workspaces/historical_real/`; root `data/duckdb` and `data/parquet` paths
are legacy/default paths when no workspace wrapper is used.
Canonical `gold/v2/historical_training_set*` and `gold/v2/historical_backtest_panel`
are immutable by default. Use variant `--output-name` values for experiments, and only
pass `--allow-canonical-overwrite` when intentionally promoting a rebuilt canonical
dataset. The writer now snapshots the previous parquet + manifest under
`artifacts/recovery/` first.

If you prefer step-by-step control instead of the one-shot bootstrap, the lower-level
commands remain available: `init-warehouse`, `backfill-*`, `materialize-training-set`,
and `summarize-forecast-availability`.

## Legacy Cutover
If you have older smoke or experimental DuckDB files under root `data/duckdb/`,
treat them as legacy/default data. Audit before merging anything into the
`historical_real` workspace:

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax trust-check \
  --markets-path configs/market_inventory/full_training_set_snapshots.json
scripts/pmtmax-workspace historical_real uv run pmtmax compact-warehouse
```

Do not merge root legacy data into canonical outputs unless a separate audited
promotion explicitly proves it is real-only. During an active writer command, a
temporary `warehouse.duckdb.lock` file may appear next to the active workspace
database and disappear when the command exits.

For legacy raw/parquet run directories created before the canonical layout, use:

```bash
uv run pmtmax inventory-legacy-runs
uv run pmtmax archive-legacy-runs --execute
```

## Backtest Workflow
```bash
scripts/pmtmax-workspace historical_real uv run pmtmax backtest \
  --dataset-path data/workspaces/historical_real/parquet/gold/historical_training_set.parquet \
  --panel-path data/workspaces/historical_real/parquet/gold/historical_backtest_panel.parquet \
  --pricing-source real_history \
  --model-name champion
```

This path requires the official Polymarket price-history panel. Build the panel first:

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax backfill-price-history \
  --markets-path configs/market_inventory/historical_temperature_snapshots.json \
  --only-missing \
  --price-no-cache \
  --limit-markets 25
scripts/pmtmax-workspace historical_real uv run pmtmax materialize-backtest-panel \
  --dataset-path data/workspaces/historical_real/parquet/gold/historical_training_set.parquet \
  --markets-path configs/market_inventory/historical_temperature_snapshots.json \
  --allow-canonical-overwrite
scripts/pmtmax-workspace historical_real uv run pmtmax backtest \
  --dataset-path data/workspaces/historical_real/parquet/gold/historical_training_set.parquet \
  --panel-path data/workspaces/historical_real/parquet/gold/historical_backtest_panel.parquet \
  --pricing-source real_history \
  --model-name champion
```

If you want an execution-aware proxy on top of the same official last-price panel,
switch the pricing source and set an explicit half-spread penalty:

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax backtest \
  --dataset-path data/workspaces/historical_real/parquet/gold/historical_training_set.parquet \
  --panel-path data/workspaces/historical_real/parquet/gold/historical_backtest_panel.parquet \
  --pricing-source quote_proxy \
  --quote-proxy-half-spread 0.02 \
  --model-name champion
```

Official-price runs write
`artifacts/workspaces/historical_real/backtests/v2/backtest_metrics_real_history.json` and
`artifacts/workspaces/historical_real/backtests/v2/backtest_trades_real_history.json`. Quote-proxy runs write
`artifacts/workspaces/historical_real/backtests/v2/backtest_metrics_quote_proxy.json` and
`artifacts/workspaces/historical_real/backtests/v2/backtest_trades_quote_proxy.json` for diagnostics only.

Backtests record `contract_version`, `split_policy`, and `leakage_audit_passed`
in the metrics artifact.

To rerun the current recent `Seoul` / `NYC` / `London` benchmark end-to-end into isolated per-city directories, use:

```bash
scripts/run_recent_core_benchmark_local.sh
```

The runner writes per-city metrics plus diagnostic `city x horizon` real-versus-quote-proxy deltas into `recent_core_benchmark_summary.json`. Promotion gates use `real_history` official-price metrics. It also applies `configs/recent-core-horizon-policy.yaml`, records policy-filtered metrics for the currently recommended horizons, and adds both top-level aggregate profitability fields and nested `cities.<city>.horizons.<horizon>` summaries. For LGBM, pass `--model-name lgbm_emos --variant <variant>` so the benchmarked variant matches the artifact you intend to publish. Use `--prebuilt-dataset-path` and `--prebuilt-panel-path` together when rerunning from trusted historical-real parquet instead of refetching forecasts/prices; `--prebuilt-last-n-market-days 90 --backtest-last-n 60 --retrain-stride 10` keeps an initial 30-market-day training window and avoids one-shot three-row retrains. The runner rejects retrain strides that are larger than the available city test splits. Use `--reuse-existing` only when you are recomputing the same model/variant summary from existing city runs.

## Paper Trading Workflow
```bash
scripts/pmtmax-workspace historical_real uv run pmtmax benchmark-models --retrain-stride 30
scripts/run_recent_core_benchmark_local.sh
uv run pmtmax publish-champion /path/to/workspace/model.pkl --recent-core-summary-path artifacts/workspaces/recent_core_eval/recent_core_benchmark/recent_core_benchmark_summary.json
scripts/pmtmax-workspace ops_daily uv run pmtmax paper-trader --core-recent-only --model-name champion
scripts/pmtmax-workspace ops_daily uv run pmtmax revenue-gate-report
```

Paper trading uses active discovered markets when available. If no active max-temperature markets are currently listed on Polymarket, the command exits cleanly with no fills.
The cron-safe daily wrapper is `scripts/daily_experiment.sh`; on this KST host it
should run at `0 9 * * *` after daily temperature markets open. The wrapper runs
inside `ops_daily`, scans active markets, backfills real truth/forecasts with
strict archive mode, logs Gamma prices, generates scan-edge signals, and passes
the same `discovered_markets.json` snapshot into `paper-trader`.
`paper-trader`, `opportunity-report`, `paper-mm`, `live-trader`, and `live-mm` use real CLOB books by default and report missing books explicitly instead of fabricating liquidity.
`paper-trader`, `live-trader`, `scan-daemon`, `opportunity-report`, and `opportunity-shadow`
also default to the checked-in recent horizon policy. The current recommended
set is `Seoul=market_open+previous_evening+morning_of`,
`NYC=market_open+previous_evening`, and
`London=market_open+previous_evening+morning_of`.
Paper/live/opportunity paths use the v2 forecast contract, write outputs under the active workspace signal root, and reject uncalibrated forecasts as `missing_calibrator`. For revenue-first operation, the preferred loop is `benchmark-models -> recent_core_benchmark -> publish-champion -> paper/opportunity/open-phase shadow -> revenue-gate-report`.
If recent-core remains `NO_GO`, the exploratory follow-up loop is
`hope-hunt-report -> hope-hunt-daemon` on `supported_wu_open_phase` rather than
loosening the live gate.

## Opportunity Workflow
```bash
scripts/pmtmax-workspace ops_daily uv run pmtmax opportunity-report --core-recent-only --model-name champion
```

This writes `artifacts/workspaces/ops_daily/signals/v2/opportunity_report.json` and separates `tradable`,
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
  --model-name champion \
  --interval 60
```

For a home-machine operational entrypoint that first ensures the canonical
dataset/model exist locally and then starts the watcher, use:

```bash
scripts/run_opportunity_shadow_watch.sh --max-cycles 1
scripts/run_opportunity_shadow_watch.sh --city Seoul --interval 60
```

This keeps a near-term (`today` / `tomorrow` in each market timezone) append-only
 audit trail in `artifacts/workspaces/ops_daily/signals/v2/opportunity_shadow.jsonl`, plus latest and summary views
under `artifacts/workspaces/ops_daily/signals/v2/opportunity_shadow_latest.json` and
`artifacts/workspaces/ops_daily/signals/v2/opportunity_shadow_summary.json`.

If you want to watch the hypothesis that taker alpha exists right after listing,
use the open-phase watcher instead:

```bash
uv run pmtmax open-phase-shadow \
  --market-scope supported_wu_open_phase \
  --model-name champion \
  --open-window-hours 24 \
  --interval 60
```

This writes `artifacts/workspaces/ops_daily/signals/v2/open_phase_shadow.jsonl`, `artifacts/workspaces/ops_daily/signals/v2/open_phase_shadow_latest.json`,
and `artifacts/workspaces/ops_daily/signals/v2/open_phase_shadow_summary.json`. The watcher keys off
`componentMarkets[*].acceptingOrdersTimestamp` when available and falls back to
market creation/deploy timestamps, so it can test whether spreads or raw gaps
look different immediately after listing.

For a ranked, no-order search loop over the same market family, use:

```bash
scripts/pmtmax-workspace ops_daily uv run pmtmax hope-hunt-report --model-name champion
scripts/pmtmax-workspace ops_daily uv run pmtmax hope-hunt-daemon --model-name champion --interval 300
```

This keeps a scope-limited candidate board for supported WU-family active
markets and marks alerts only when `after_cost_edge > 0` or a fresh listing is
blocked solely by spread width.

## Dry-Run Live Workflow
```bash
uv run pmtmax live-trader \
  --core-recent-only \
  --model-name champion \
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
- `.agents/skills/` and `.claude/skills/`: repo-local skill entrypoints for Codex and Claude

## Known Limitations
- Historical exact L2 orderbook archives are not publicly available for older markets. The repo therefore separates research backtests from exact execution replay.
- The forecast backfill workflow now uses Open-Meteo's generic archive endpoint `historical-forecast-api.open-meteo.com/v1/forecast` for archived model runs. Research backfills default to strict archive mode and do not silently replace missing forecasts with fixtures.
- Bundled Seoul, NYC, Hong Kong, and Taipei forecast fixtures remain test/demo-only. If fixture rows appear in a canonical warehouse, `trust-check` treats that warehouse as non-canonical.
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
