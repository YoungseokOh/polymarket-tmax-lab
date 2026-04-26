# Research Loop

## Use This When
- `materialize-training-set`
- `train-baseline`, `train-advanced`
- `backtest`
- `paper-trader`
- `paper-multimodel-report`
- `execution-sensitivity-report`
- `market-bottleneck-report`
- `execution-watchlist-playbook`
- `opportunity-report`
- `opportunity-shadow`
- `observation-report`
- `observation-shadow`
- `observation-daemon`
- `approve-live-candidate`
- `open-phase-shadow`
- `hope-hunt-report`
- `hope-hunt-daemon`

## Current Public Champion
- shared registry: `artifacts/public_models/champion.pkl`
- metadata: `artifacts/public_models/champion.json`
- promotion rule: only `uv run pmtmax publish-champion ...` may update the public alias
- current published model family/variant: `lgbm_emos / high_neighbor_oof`
- trusted training inventory is market-level; the latest audit was 1,834 markets
  materialized into 5,478 training rows across supported horizons

## Daily Data Collection
```bash
bash scripts/daily_experiment.sh
```

Runs in the `ops_daily` workspace:
scan-markets → backfill-truth → backfill-forecasts `--strict-archive` →
log_gamma_prices → scan-edge → record_paper_trades → paper-trader →
track_paper_trade_outcomes.

This loop does not retrain the model, mutate `historical_real`, or publish a
champion. It collects active-market operational evidence and forward diagnostics
only. On a KST host, schedule it after daily markets open:

```bash
0 9 * * * bash /home/seok436/projects/polymarket-tmax-lab/scripts/daily_experiment.sh >> /home/seok436/projects/polymarket-tmax-lab/logs/daily_experiment.log 2>&1
```

If the cron host is UTC, use `0 0 * * *` for the same 09:00 KST wall-clock time.

## 2-Hour Price Check (cron every 2 hours)
```bash
bash scripts/run_price_check.sh
```
Runs: log_gamma_prices → track_paper_trade_outcomes

Cron should call the wrapper, not raw `uv run ...` commands, and the redirect
must use an absolute path:

```bash
0 */2 * * * bash /home/seok436/projects/polymarket-tmax-lab/scripts/run_price_check.sh >> /home/seok436/projects/polymarket-tmax-lab/logs/price_check.log 2>&1
```

## scan-edge (signal generation)
```bash
uv run pmtmax scan-edge \
    --model-name champion \
    --min-edge 0.15 \
    --min-model-prob 0.05 \
    --max-model-prob 0.95 \
    --output artifacts/workspaces/ops_daily/signals/v2/scan_edge_latest.json
```
`--min-model-prob 0.05 --max-model-prob 0.95` is required — filters out overconfident 0%/100% predictions.

## Trust Check
Before long-running rebuilds or canonical overwrite, run:

```bash
uv run pmtmax trust-check --markets-path configs/market_inventory/full_training_set_snapshots.json
```

`trust-check` fails closed on synthetic inventories, `synthetic_` market ids,
fixture forecasts, fabricated books, or non-`real_market` dataset profiles.
For weather pretraining, run `scripts/pmtmax-workspace weather_train uv run pmtmax trust-check --workflow weather_training`; that workflow requires `weather_real` and forbids Polymarket inventories.

## Model Training
```bash
# Weather-real pretrain, no Polymarket market ids or prices
scripts/pmtmax-workspace weather_train uv run pmtmax collect-weather-training --city Seoul --date-from 2024-01-01 --date-to 2024-01-07 --http-timeout-seconds 15 --http-retries 1 --missing-only
scripts/pmtmax-workspace weather_train uv run pmtmax train-weather-pretrain --model-name gaussian_emos

# Train a specific lgbm_emos variant
uv run pmtmax train-advanced --model-name lgbm_emos --variant high_neighbor_oof

# Quick comparison (champion baseline + OOF family)
uv run python scripts/quick_eval.py
```

`collect-weather-training` now emits station/date progress lines on stderr by default and keeps the final machine-readable JSON on stdout. When Open-Meteo starts throttling, prefer shorter `--http-timeout-seconds` / `--http-retries` values and resumable date chunks over large silent runs. Keep `--max-consecutive-429 2`: two consecutive `429` responses mean the free-path daily limit is effectively hit, so cancel the remaining chunk and record `rate-limit-cancelled` in checker state.

For repeated older-backfill collection, prefer the queue agent:

```bash
scripts/pmtmax-workspace weather_train uv run python scripts/run_weather_train_queue_agent.py
```

It reads `checker/weather_train_status.md`, advances the next `7`-day queue,
updates `checker/weather_train_status.md` / `checker/weather_train_collection_log.md`
after every chunk, auto-refreshes `gaussian_emos` pretrain when the configured
row-gap threshold is hit, and defaults to the two-consecutive-`429` cancel rule.
Read the checker files for the live row count and current next queue.

This weather queue can run in parallel with
`scripts/pmtmax-workspace historical_real uv run pmtmax backfill-price-history`
because the two jobs use different workspace roots. Inside `historical_real`,
keep `backfill-price-history`, `materialize-backtest-panel`, and benchmark/publish
steps serialized.

Use `checker/weather_train_runbook.md` as the operational checklist and keep
`checker/weather_train_status.md` / `checker/weather_train_collection_log.md`
updated after every collection or pretrain refresh.

For daily official price-history recovery, use the historical price agent:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_historical_price_recovery_agent.py
```

It reads `checker/historical_price_status.md`, advances one missing-price shard
by default, runs `backfill-price-history -> materialize-backtest-panel ->
summarize-price-history-coverage`, updates
`checker/historical_price_status.md` /
`checker/historical_price_collection_log.md`, and keeps the latest
`priced_decision_rows` anchor visible for daily tracking.

For baseline training plus autoresearch queue management, use the model research
agent:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_model_research_agent.py
```

It reuses the current dataset/panel signatures, retrains the baseline only when
needed, auto-creates the next small YAML candidate when the queue is empty,
processes one candidate through `step -> gate -> paper -> promote`, and updates
`checker/model_research_status.md` /
`checker/model_research_log.md`. Public champion publish stays disabled unless
you explicitly pass `--enable-publish` with a candidate-specific recent-core GO
summary path.

`quick_eval.py` sorts by Celsius-normalized CRPS and prints the raw market-unit
CRPS beside it for audit. Promotion still requires benchmark, paper, or shadow evidence.
`train-advanced --pretrained-weather-model <path>` injects
`weather_pretrain_mean`, `weather_pretrain_std`, and
`weather_pretrain_delta_vs_gfs` into the `historical_real` model frame and wraps
the saved model so the same augmentation is applied at prediction time. The
first `high_neighbor_oof` quick eval with active injection worsened `CRPS_C`
from `0.8004` to `0.8241`, and the `delta_only` follow-up still worsened to
`0.8111`; treat these as experiments that need a candidate-specific gate before
any promotion. Public promotion still requires a `real_history` recent-core
`GO` summary.

## Dataset Build — SAFETY RULE
```bash
# ALWAYS pass --markets-path. Never run without it.
scripts/pmtmax-workspace historical_real uv run pmtmax build-dataset \
    --markets-path configs/market_inventory/full_training_set_snapshots.json \
    --forecast-missing-only \
    --allow-canonical-overwrite
```
⚠️ Running without `--markets-path` will rebuild with only example data (12 rows).
The CLI now defaults to `full_training_set_snapshots.json` with a yellow warning,
but explicit is safer. A shrinkage guard (50% threshold) will block silent overwrites.
`full_training_set_snapshots.json` is a checked-in training inventory, not the
auto-refreshed canonical historical backlog; if you need the latest curated
historical collection, use `configs/market_inventory/historical_temperature_snapshots.json`.
If you want to refresh `full_training_set_snapshots.json` itself, do it explicitly
from the refreshed curated historical inventory in the same change; do not assume
that canonical historical collection backfills will rewrite that file for you.
When the warehouse already has most forecast payloads, add `--forecast-missing-only`
to `build-dataset` to avoid refetching existing forecast request keys.
`bootstrap-lab`에도 같은 shortcut이 있고, `scripts/run_full_historical_batch.sh`의
forecast 단계는 기본적으로 missing-only top-off를 사용한다.
For curated multi-source variants, pass repeated `--model` values to
`build-dataset` or `materialize-training-set`; otherwise a config default can
silently materialize a GFS-only dataset with the same row count but fewer NWP
feature columns.
Canonical `historical_training_set*` / `historical_backtest_panel` outputs are immutable
unless you pass `--allow-canonical-overwrite`, and the writer now snapshots the existing
parquet + manifest under `artifacts/recovery/` first.

## Benchmark (slow — use retrain_stride)
```bash
scripts/pmtmax-workspace historical_real uv run pmtmax benchmark-models --retrain-stride 30
```

## Default Workflow
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
scripts/pmtmax-workspace historical_real uv run pmtmax train-advanced --model-name lgbm_emos --variant high_neighbor_oof
uv run python scripts/quick_eval.py
scripts/pmtmax-workspace historical_real uv run pmtmax benchmark-models --retrain-stride 30
```

Non-canonical curated multi-source experiment:

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax materialize-training-set \
    --markets-path configs/market_inventory/historical_temperature_snapshots.json \
    --model ecmwf_ifs025 \
    --model ecmwf_aifs025_single \
    --model kma_gdps \
    --model gfs_seamless \
    --output-name <variant>
```

April 25, 2026 check: `historical_training_set_curated_multisource_20260425`
restored all four NWP source groups and improved canonical baseline holdout
CRPS_C (`0.8004` -> `0.6749`), but on the curated holdout it worsened CRPS_C
versus GFS-only (`2.0559` -> `2.8771`) while improving Brier/DirAcc. The
failure mode was under-dispersed, cold-biased tails, so promotion needs
variance/tail calibration first.

Tail calibration diagnostic:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_tail_calibration_experiment.py
```

The April 26, 2026 run wrote
`artifacts/curated_multisource_tailcal_experiment_20260426.json` and diagnostic
wrapper models under
`artifacts/workspaces/historical_real/models/curated_multisource_tailcal_20260426/`.
The balanced wrapper reduced curated holdout CRPS_C from `2.8771` to `1.7195`
and kept baseline holdout near canonical baseline (`0.8043`), but the root
signal was zero-Celsius sentinel NWP daily-max features (`32F` in F markets,
`0C` in C markets) marked as available. Treat tailcal as diagnostic until that
feature-validity bug is fixed upstream and re-evaluated without holdout-tuned
fallback heuristics.

Sentinel-fix follow-up:
`historical_training_set_curated_multisource_sentinelfix_20260426` removed
all-source zero-C sentinel rows (`1083 -> 0`) and retrained
`lgbm_emos/high_neighbor_oof`, but curated holdout CRPS_C remained worse than
same-holdout GFS-only (`2.7690` vs `2.1103`). Keep this as a diagnostic base,
not a promotion candidate.

Source-gating follow-up:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_source_gating_experiment.py
```

`scripts/run_source_gating_experiment.py` trained binary, fallback-regret
weighted, and absolute-regret weighted gates between the sentinel-fix
multi-source model and GFS-only. Best curated holdout CRPS_C only moved
`2.7690 -> 2.7607`, still worse than same-holdout GFS-only `2.1103`, while
baseline holdout stayed safe/slightly better (`0.4965 -> 0.4932`). The gate
selected fallback for only about `2%` of curated holdout rows because fit-side
labels favored multi-source and the holdout failures are city/region distribution
shift. Keep source-gating artifacts diagnostic; next experiment should target
source-disagreement/region-driven variance calibration or proper OOF stacking.

Disagreement calibration follow-up:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/run_disagreement_calibration_experiment.py
```

The April 26, 2026 run added `DisagreementCalibratedGaussianModel`. Variance-only
source-disagreement calibration improved sentinel-fix curated holdout CRPS_C
`2.7690 -> 2.2342`, but still lost to same-holdout GFS-only `2.1103`. The best
positive primary-vs-GFS disagreement mean-shrink + variance wrapper improved
curated holdout CRPS_C to `1.3153`, Brier to `0.0748`, and ECE to `0.0214`,
but baseline holdout CRPS_C worsened from raw `0.4965` to `0.5627`. Keep these
wrappers diagnostic only. The next serious candidate should make the
disagreement rule non-holdout-tuned, either as explicit `lgbm_emos` features or
as a proper OOF stacker.

## Autoresearch Loop
`karpathy/autoresearch`-style exploration is now a first-class YAML candidate loop around `lgbm_emos`.
Run it through the `historical_real` workspace wrapper so it cannot accidentally
read legacy mixed/synthetic default v2 data.

```bash
scripts/pmtmax-workspace historical_real uv run pmtmax autoresearch-init --baseline-variant high_neighbor_oof
scripts/pmtmax-workspace historical_real uv run pmtmax autoresearch-step --spec-path artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/my_candidate.yaml
scripts/pmtmax-workspace historical_real uv run pmtmax autoresearch-gate --spec-path artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/my_candidate.yaml
scripts/pmtmax-workspace historical_real uv run pmtmax autoresearch-analyze-paper --spec-path artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/my_candidate.yaml
scripts/pmtmax-workspace historical_real uv run pmtmax autoresearch-promote --spec-path artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/my_candidate.yaml
```

- candidate YAML is the only thing the agent should edit inside the loop
- `autoresearch-promote` requires CLI-generated gate leaderboard JSON/CSV, matching dataset/panel signatures, a candidate calibrator, and paper `overall_gate_decision=GO`
- `INCONCLUSIVE` is fail-closed for promotion
- promoted winners are copied to `configs/autoresearch/lgbm_emos/promoted/`
- alias publish remains explicit even after promotion; `autoresearch-promote` no longer mutates the public alias

집에서 canonical dataset/model 존재 여부를 신경 쓰지 않고 바로 recent benchmark나
shadow watcher를 돌리고 싶으면 다음 래퍼를 사용한다.

```bash
scripts/run_recent_core_benchmark_local.sh --model-name lgbm_emos --variant high_neighbor_oof --retrain-stride 1
scripts/pmtmax-workspace recent_core_eval uv run python scripts/run_recent_core_benchmark.py --model-name lgbm_emos --variant high_neighbor_oof --retrain-stride 10 --backtest-last-n 60 --prebuilt-dataset-path data/workspaces/historical_real/parquet/gold/historical_training_set.parquet --prebuilt-panel-path data/workspaces/historical_real/parquet/gold/historical_backtest_panel.parquet --prebuilt-last-n-market-days 90
scripts/run_opportunity_shadow_watch.sh --max-cycles 1
```

`paper-trader`, `live-trader`, `scan-daemon`, `opportunity-report`, and
`opportunity-shadow` default to `--horizon policy`. The policy lives in
`configs/recent-core-horizon-policy.yaml` and filtered rows are emitted as
`reason=policy_filtered`.
기존 backtest artifact에서 recent-core city/horizon coverage와 policy PnL만
빠르게 다시 보고 싶으면 다음 helper를 사용한다.

```bash
uv run python scripts/recent_core_diagnostics.py
```

`recent_core_benchmark_summary.json`의 top-level `decision`만 public `champion`
publish gate로 인정되며 이 decision은 `real_history` official-price metrics를
기준으로 한다. `reduced_core_candidate`는 coverage-thin city를 제외한
축소 코어 진단 결과라서 `decision=GO`여도 publish에는 쓸 수 없다.
LGBM 후보를 검증할 때는 `--model-name lgbm_emos --variant <variant>`로
recent-core benchmark 대상과 `publish-champion`에 넘길 artifact variant를
일치시킨다.
도시별 split이 작은 benchmark에서는 `--retrain-stride 30`을 쓰지 않는다.
runner는 available test split보다 큰 retrain stride를 fail-fast로 막는다.
paper-only sweeps can override that with
`--horizon-policy-path configs/paper-all-supported-horizon-policy.yaml`.
The paper-only sweep grid lives in `configs/paper-exploration.yaml`.
수익화 전용 루프에서는 `--core-recent-only`와 `--model-name champion` 조합을 기본으로 쓰되, `ops_daily` workspace에서만 운영한다.
Forward paper evidence must come from real CLOB books. If `book_source` is
missing or fixture-only, treat the run as a diagnostic and do not record it into
`forward_paper_trades.json`.
Current paper commands reject `book_source=fixture`; CLOB failures are represented
as `missing_book` rows instead of tradable signals.
The daily wrapper passes the just-scanned `discovered_markets.json` into
`paper-trader` so CLOB diagnostics and Gamma scan-edge signals are tied to the
same active-market snapshot.
recent-core 바깥의 fresh listing 탐색은 `--market-scope supported_wu_open_phase`
또는 전용 wrapper인 `hope-hunt-report` / `hope-hunt-daemon`을 쓴다.

zero-fill 진단은 다음 셋으로 돌린다.

```bash
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

- `paper-multimodel-report`는 챔피언과 현재 top challenger pool을 같은 snapshot/books에서 비교한다
- `execution-sensitivity-report`는 paper-only threshold/policy sweep으로 fills가 생기는 조합을 찾는다
- `market-bottleneck-report`는 row-level 결과를 `fee_sensitive_watchlist`, `raw_edge_desert_watchlist`, `policy_blocked_watchlist`로 요약한다
- `execution-watchlist-playbook`는 위 진단 산출물을 `execution_watchlist_playbook.json/.md`로 묶고 Tier A fee-sensitive watch rule을 계산한다

관측 기반 weather-station 루프는 다음 command를 쓴다.

```bash
scripts/pmtmax-workspace ops_daily uv run pmtmax observation-report --model-name champion
scripts/pmtmax-workspace ops_daily uv run pmtmax observation-shadow --model-name champion --max-cycles 1
scripts/pmtmax-workspace ops_daily uv run pmtmax observation-daemon --model-name champion
uv run pmtmax approve-live-candidate <token> --dry-run
uv run pmtmax station-dashboard
uv run pmtmax station-dashboard-daemon
scripts/pmtmax-workspace ops_daily uv run pmtmax station-cycle --model-name champion
scripts/pmtmax-workspace ops_daily uv run pmtmax station-daemon --model-name champion
```

- `observation-report` / `observation-shadow`는 target-day market에서 strongest lower-bound를 골라 이미 불가능해진 하단 outcome을 0으로 만들고 queue를 `tradable`, `manual_review`, `blocked`로 분리한다
- source priority는 `exact_public intraday -> documented research intraday -> METAR fallback`이다. 현재 내장 exact/research intraday는 Hong Kong/HKO, Taipei/CWA, Seoul/AIR_CALP다
- output은 `artifacts/workspaces/ops_daily/signals/v2/observation_shadow_latest.json`, `observation_alerts_latest.json`, `live_pilot_queue.json`에 기록된다
- `opportunity_shadow_summary.json`과 `observation_shadow_summary.json`에는 `by_reason`, `by_city_reason`, `by_horizon_reason`, `top_near_miss_markets`, `top_fee_killed_markets`, `top_spread_blocked_markets`, `top_policy_filtered_markets`가 함께 들어간다
- `observation_shadow_summary.json`에는 추가로 `by_source_family`, `by_observation_source`, `top_after_cost_edges`, `top_price_vs_observation_gaps`가 들어가므로 어떤 소스 계층이 실제 edge를 만드는지 바로 비교할 수 있다
- `approve-live-candidate`는 queue token을 다시 검증하고 preview/post 전에 candidate가 여전히 살아 있는지 fail-closed로 확인한다
- `station-dashboard`는 opportunity / observation / open-phase / revenue-gate 아티팩트와 `execution_watchlist_playbook.json`을 한 화면용 JSON/HTML로 합쳐서 Discovery / Observation / Execution / Watchlist 패널을 만든다
- dashboard의 watchlist alert는 Tier A rule과 현재 `best_ask`를 비교해서만 올리며, live spread/edge guardrail을 자동으로 바꾸지 않는다
- `station-cycle`은 opportunity-report, opportunity-shadow, observation-report, open-phase-shadow, revenue-gate-report, station-dashboard를 순서대로 갱신하는 one-shot orchestrator다
- `station-daemon`은 같은 전체 cycle을 interval 기반으로 반복한다
- `revenue-gate-report`와 `station-cycle`의 기본 benchmark 입력은 `artifacts/workspaces/historical_real/benchmarks/v2/benchmark_summary.json`이다. recent-core 전용 richer summary가 있으면 `--benchmark-summary-path`로 덮어쓴다

## Real Historical Workflow
```bash
scripts/run_historical_refresh_pipeline.sh
scripts/run_full_historical_batch.sh
scripts/pmtmax-workspace historical_real uv run pmtmax train-baseline --model-name gaussian_emos
scripts/pmtmax-workspace historical_real uv run pmtmax backtest --pricing-source real_history --model-name gaussian_emos
```

장기 closed-event refresh는 `run_historical_refresh_pipeline.sh`로 backlog를 계속 정산하고, warehouse rebuild가 필요할 때만 `run_full_historical_batch.sh`를 돌린다.
refresh manifest는 partial progress를 남기므로 source lag나 transient request failure가 있어도 연구 루프 전체를 막지 않는다.

세부 단계를 나눠서 돌리고 싶으면:

```bash
uv run python scripts/refresh_historical_event_urls.py
uv run python scripts/build_historical_market_inventory.py --truth-per-source-limit 1
uv run python scripts/validate_historical_market_inventory.py --truth-per-source-limit 1
scripts/pmtmax-workspace historical_real uv run pmtmax build-dataset --markets-path configs/market_inventory/historical_temperature_snapshots.json --allow-canonical-overwrite
scripts/pmtmax-workspace historical_real uv run pmtmax materialize-backtest-panel --markets-path configs/market_inventory/historical_temperature_snapshots.json --allow-canonical-overwrite
scripts/pmtmax-workspace historical_real uv run pmtmax train-baseline --model-name gaussian_emos
scripts/pmtmax-workspace historical_real uv run pmtmax backtest --pricing-source real_history --model-name gaussian_emos
uv run python scripts/build_active_weather_watchlist.py
```

## Artifacts
- weather-real pretrain dataset: `data/workspaces/weather_train/parquet/gold/weather_training_set.parquet`
- weather pretrain artifacts: `artifacts/workspaces/weather_train/models/v2/`
- historical-real dataset: `data/workspaces/historical_real/parquet/gold/historical_training_set.parquet`
- workspace-local model artifacts: `artifacts/workspaces/<workspace>/models/v2/`
- public champion alias: `artifacts/public_models/champion.pkl`, `artifacts/public_models/champion.json`
- benchmark outputs: `artifacts/workspaces/<workspace>/benchmarks/v2/`
- ablation outputs: `artifacts/workspaces/<workspace>/benchmarks/v2/*_ablation_leaderboard.*`
- backtest outputs: `artifacts/workspaces/<workspace>/backtests/v2/`
- paper/opportunity/observation/open-phase/hope-hunt outputs: `artifacts/workspaces/ops_daily/signals/v2/`
- station dashboard outputs: `artifacts/workspaces/ops_daily/signals/v2/station_dashboard.{json,html}`
- execution watchlist playbook: `artifacts/workspaces/ops_daily/signals/v2/execution_watchlist_playbook.{json,md}`
- station orchestrator state: `artifacts/workspaces/ops_daily/signals/v2/station_cycle_state.json`
- revenue gate output: `artifacts/workspaces/ops_daily/signals/v2/revenue_gate_summary.json`
- closed-event manifests: `data/manifests/historical_event_candidates.json`, `data/manifests/historical_event_page_fetches.json`, `data/manifests/historical_collection_status.json`
- active watchlist: `artifacts/active_weather_watchlist.json`

## Notes
- `backtest`는 최소 2개 row가 필요하다
- 단일 도시 smoke면 horizon을 2개 이상 주는 편이 안전하다
- 모델보다 settlement fidelity와 lookahead 방지가 우선이다
- grouped split이 기본이며 row split은 지원 workflow가 아니다
- `benchmark-ablations`는 grouped one-shot holdout 전용 내부 연구 command이며 champion alias는 publish하지 않는다
- `benchmark-models`는 workspace-local leaderboard만 갱신한다
- public alias promotion은 `publish-champion` + recent-core `GO` summary 조합만 허용된다
- active market 탐색에서는 `missing_book`과 `no_positive_edge`를 구분해서 해석해야 한다
- live/paper/opportunity 경로는 calibrated probability가 없으면 `missing_calibrator`로 fail closed 한다
- `opportunity-shadow`는 주문/알림 없이 `raw_gap`, `after_cost_edge`, reject reason을 시간축으로 쌓아 현재 탐색 로직이 실제로 기회를 잡는지 검증하는 경로다
- `observation-shadow`는 같은 book/fee/slippage guardrail 위에 target-day intraday lower-bound를 덧씌워 관측 기반 dislocation이 실제로 반복되는지 검증하는 경로다
- `open-phase-shadow`는 `componentMarkets[*].acceptingOrdersTimestamp` 기준 최근 상장 시장만 골라 `market_open` horizon으로 평가하는 실험 경로다
- `hope-hunt-report`와 `hope-hunt-daemon`은 `supported_wu_open_phase` 범위에서 fresh listing을 우선순위화하는 no-order candidate loop다
- shadow/open-phase summary에는 `by_city`, `by_horizon`, `by_city_horizon`, `gate_decision`, `gate_reason`이 함께 기록된다
- hope-hunt summary에는 `by_open_phase_age_bucket`, `by_priority_bucket`, `top_candidates`, `gate_decision`, `gate_reason`이 함께 기록된다
- `revenue-gate-report`는 recent-core benchmark와 세 shadow summary를 합쳐 소액 live pilot 전환 가능 여부를 `GO / INCONCLUSIVE / NO_GO`로 출력한다
- active opportunity 진단은 `raw_gap_non_positive`, `fee_killed_edge`, `slippage_killed_edge`, `after_cost_positive_but_spread_too_wide`를 분리해서 봐야 한다
- observation live pilot은 manual approval이 기본이며, `research_public` candidate는 exact-public과 같은 queue에 오르더라도 tier/risk flag를 숨기지 않는다
- 현재 기본 horizon policy는 `Seoul=market_open+previous_evening+morning_of`, `NYC=market_open+previous_evening`, `London=market_open+previous_evening+morning_of`이다
