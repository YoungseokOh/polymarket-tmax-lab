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

## Current Champion (as of 2026-04-04)
- Model: `lgbm_emos`, variant: `recency_neighbor_oof`
- pkl: `artifacts/models/v2/lgbm_emos__recency_neighbor_oof.pkl`
- champion.json: `artifacts/models/v2/champion.json`

## Daily Data Collection (cron 00:00 UTC = 09:00 KST)
```bash
bash scripts/daily_experiment.sh
```
Runs: scan-markets → backfill-truth → backfill-forecasts → log_gamma_prices → scan-edge → track_paper_trade_outcomes

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
    --model-name trading_champion \
    --min-edge 0.15 \
    --min-model-prob 0.05 \
    --max-model-prob 0.95 \
    --output artifacts/signals/v2/scan_edge_latest.json
```
`--min-model-prob 0.05 --max-model-prob 0.95` is required — filters out overconfident 0%/100% predictions.

## Model Training
```bash
# Train a specific lgbm_emos variant
uv run pmtmax train-advanced --model-name lgbm_emos --variant recency_neighbor_oof

# Quick comparison (champion baseline + OOF family)
uv run python scripts/quick_eval.py
```

## Dataset Build — SAFETY RULE
```bash
# ALWAYS pass --markets-path. Never run without it.
uv run pmtmax build-dataset \
    --markets-path configs/market_inventory/full_training_set_snapshots.json \
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
Canonical `historical_training_set*` / `historical_backtest_panel` outputs are immutable
unless you pass `--allow-canonical-overwrite`, and the writer now snapshots the existing
parquet + manifest under `artifacts/recovery/` first.

## Benchmark (slow — use retrain_stride)
```bash
uv run pmtmax benchmark-models --retrain-stride 30
```

## Default Workflow
```bash
uv run pmtmax bootstrap-lab
uv run pmtmax build-dataset \
    --markets-path configs/market_inventory/full_training_set_snapshots.json \
    --allow-canonical-overwrite
uv run pmtmax materialize-backtest-panel --allow-canonical-overwrite
uv run pmtmax train-advanced --model-name lgbm_emos --variant recency_neighbor_oof
uv run python scripts/quick_eval.py
uv run pmtmax benchmark-models --retrain-stride 30
```

## Autoresearch Loop
`karpathy/autoresearch`-style exploration is now a first-class YAML candidate loop around `recency_neighbor_oof`.

```bash
uv run pmtmax autoresearch-init
uv run pmtmax autoresearch-step --spec-path artifacts/autoresearch/<run_tag>/candidates/my_candidate.yaml
uv run pmtmax autoresearch-gate --spec-path artifacts/autoresearch/<run_tag>/candidates/my_candidate.yaml
uv run pmtmax autoresearch-analyze-paper --spec-path artifacts/autoresearch/<run_tag>/candidates/my_candidate.yaml
uv run pmtmax autoresearch-promote --spec-path artifacts/autoresearch/<run_tag>/candidates/my_candidate.yaml
```

- candidate YAML is the only thing the agent should edit inside the loop
- promoted winners are copied to `configs/autoresearch/lgbm_emos/promoted/`
- alias publish remains explicit even after promotion

집에서 canonical dataset/model 존재 여부를 신경 쓰지 않고 바로 recent benchmark나
shadow watcher를 돌리고 싶으면 다음 래퍼를 사용한다.

```bash
scripts/run_recent_core_benchmark_local.sh
scripts/run_opportunity_shadow_watch.sh --max-cycles 1
```

`paper-trader`, `live-trader`, `scan-daemon`, `opportunity-report`, and
`opportunity-shadow` default to `--horizon policy`. The policy lives in
`configs/recent-core-horizon-policy.yaml` and filtered rows are emitted as
`reason=policy_filtered`.
paper-only sweeps can override that with
`--horizon-policy-path configs/paper-all-supported-horizon-policy.yaml`.
The paper-only sweep grid lives in `configs/paper-exploration.yaml`.
수익화 전용 루프에서는 `--core-recent-only`와 `--model-name trading_champion` 조합을 기본으로 쓴다.
recent-core 바깥의 fresh listing 탐색은 `--market-scope supported_wu_open_phase`
또는 전용 wrapper인 `hope-hunt-report` / `hope-hunt-daemon`을 쓴다.

zero-fill 진단은 다음 셋으로 돌린다.

```bash
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

- `paper-multimodel-report`는 챔피언과 현재 top challenger pool을 같은 snapshot/books에서 비교한다
- `execution-sensitivity-report`는 paper-only threshold/policy sweep으로 fills가 생기는 조합을 찾는다
- `market-bottleneck-report`는 row-level 결과를 `fee_sensitive_watchlist`, `raw_edge_desert_watchlist`, `policy_blocked_watchlist`로 요약한다
- `execution-watchlist-playbook`는 위 진단 산출물을 `execution_watchlist_playbook.json/.md`로 묶고 Tier A fee-sensitive watch rule을 계산한다

관측 기반 weather-station 루프는 다음 command를 쓴다.

```bash
uv run pmtmax observation-report --model-name trading_champion
uv run pmtmax observation-shadow --model-name trading_champion --max-cycles 1
uv run pmtmax observation-daemon --model-name trading_champion
uv run pmtmax approve-live-candidate <token> --dry-run
uv run pmtmax station-dashboard
uv run pmtmax station-dashboard-daemon
uv run pmtmax station-cycle --model-name trading_champion
uv run pmtmax station-daemon --model-name trading_champion
```

- `observation-report` / `observation-shadow`는 target-day market에서 strongest lower-bound를 골라 이미 불가능해진 하단 outcome을 0으로 만들고 queue를 `tradable`, `manual_review`, `blocked`로 분리한다
- source priority는 `exact_public intraday -> documented research intraday -> METAR fallback`이다. 현재 내장 exact/research intraday는 Hong Kong/HKO, Taipei/CWA, Seoul/AIR_CALP다
- output은 `artifacts/signals/v2/observation_shadow_latest.json`, `observation_alerts_latest.json`, `live_pilot_queue.json`에 기록된다
- `opportunity_shadow_summary.json`과 `observation_shadow_summary.json`에는 `by_reason`, `by_city_reason`, `by_horizon_reason`, `top_near_miss_markets`, `top_fee_killed_markets`, `top_spread_blocked_markets`, `top_policy_filtered_markets`가 함께 들어간다
- `observation_shadow_summary.json`에는 추가로 `by_source_family`, `by_observation_source`, `top_after_cost_edges`, `top_price_vs_observation_gaps`가 들어가므로 어떤 소스 계층이 실제 edge를 만드는지 바로 비교할 수 있다
- `approve-live-candidate`는 queue token을 다시 검증하고 preview/post 전에 candidate가 여전히 살아 있는지 fail-closed로 확인한다
- `station-dashboard`는 opportunity / observation / open-phase / revenue-gate 아티팩트와 `execution_watchlist_playbook.json`을 한 화면용 JSON/HTML로 합쳐서 Discovery / Observation / Execution / Watchlist 패널을 만든다
- dashboard의 watchlist alert는 Tier A rule과 현재 `best_ask`를 비교해서만 올리며, live spread/edge guardrail을 자동으로 바꾸지 않는다
- `station-cycle`은 opportunity-report, opportunity-shadow, observation-report, open-phase-shadow, revenue-gate-report, station-dashboard를 순서대로 갱신하는 one-shot orchestrator다
- `station-daemon`은 같은 전체 cycle을 interval 기반으로 반복한다
- `revenue-gate-report`와 `station-cycle`의 기본 benchmark 입력은 `artifacts/benchmarks/v2/benchmark_summary.json`이다. recent-core 전용 richer summary가 있으면 `--benchmark-summary-path`로 덮어쓴다

## Real Historical Workflow
```bash
scripts/run_historical_refresh_pipeline.sh
scripts/run_full_historical_batch.sh
uv run pmtmax train-baseline --model-name gaussian_emos
uv run pmtmax backtest --model-name gaussian_emos
```

장기 closed-event refresh는 `run_historical_refresh_pipeline.sh`로 backlog를 계속 정산하고, warehouse rebuild가 필요할 때만 `run_full_historical_batch.sh`를 돌린다.
refresh manifest는 partial progress를 남기므로 source lag나 transient request failure가 있어도 연구 루프 전체를 막지 않는다.

세부 단계를 나눠서 돌리고 싶으면:

```bash
uv run python scripts/refresh_historical_event_urls.py
uv run python scripts/build_historical_market_inventory.py --truth-per-source-limit 1
uv run python scripts/validate_historical_market_inventory.py --truth-per-source-limit 1
uv run pmtmax build-dataset --markets-path configs/market_inventory/historical_temperature_snapshots.json --allow-canonical-overwrite
uv run pmtmax train-baseline --model-name gaussian_emos
uv run pmtmax backtest --model-name gaussian_emos
uv run python scripts/build_active_weather_watchlist.py
```

## Artifacts
- gold dataset: `data/parquet/gold/v2/historical_training_set.parquet`
- sequence dataset: `data/parquet/gold/v2/historical_training_set_sequence.parquet`
- model artifacts: `artifacts/models/v2/`
- trading alias: `artifacts/models/v2/trading_champion.pkl`, `artifacts/models/v2/trading_champion.json`
- benchmark outputs: `artifacts/benchmarks/v2/leaderboard.json`, `artifacts/benchmarks/v2/leaderboard.csv`
- ablation outputs: `artifacts/benchmarks/v2/*_ablation_leaderboard.json`, `artifacts/benchmarks/v2/*_ablation_leaderboard.csv`
- champion alias: `artifacts/models/v2/champion.pkl`, `artifacts/models/v2/champion.json`
- backtest outputs: `artifacts/backtests/v2/`
- paper outputs: `artifacts/signals/v2/paper_signals.json`
- opportunity outputs: `artifacts/signals/v2/opportunity_report.json`
- shadow validation outputs: `artifacts/signals/v2/opportunity_shadow.jsonl`, `artifacts/signals/v2/opportunity_shadow_latest.json`, `artifacts/signals/v2/opportunity_shadow_summary.json`
- observation-station outputs: `artifacts/signals/v2/observation_shadow.jsonl`, `artifacts/signals/v2/observation_shadow_latest.json`, `artifacts/signals/v2/observation_shadow_summary.json`, `artifacts/signals/v2/observation_alerts_latest.json`, `artifacts/signals/v2/live_pilot_queue.json`
- station dashboard outputs: `artifacts/signals/v2/station_dashboard.json`, `artifacts/signals/v2/station_dashboard.html`, `artifacts/signals/v2/station_dashboard_state.json`
- execution watchlist playbook: `artifacts/signals/v2/execution_watchlist_playbook.json`, `artifacts/signals/v2/execution_watchlist_playbook.md`
- station orchestrator state: `artifacts/signals/v2/station_cycle_state.json`
- open-phase validation outputs: `artifacts/signals/v2/open_phase_shadow.jsonl`, `artifacts/signals/v2/open_phase_shadow_latest.json`, `artifacts/signals/v2/open_phase_shadow_summary.json`
- hope-hunt outputs: `artifacts/signals/v2/hope_hunt_history.jsonl`, `artifacts/signals/v2/hope_hunt_latest.json`, `artifacts/signals/v2/hope_hunt_summary.json`
- revenue gate output: `artifacts/signals/v2/revenue_gate_summary.json`
- closed-event manifests: `data/manifests/historical_event_candidates.json`, `data/manifests/historical_event_page_fetches.json`, `data/manifests/historical_collection_status.json`
- active watchlist: `artifacts/active_weather_watchlist.json`

## Notes
- `backtest`는 최소 2개 row가 필요하다
- 단일 도시 smoke면 horizon을 2개 이상 주는 편이 안전하다
- 모델보다 settlement fidelity와 lookahead 방지가 우선이다
- grouped split이 기본이며 row split은 지원 workflow가 아니다
- `benchmark-ablations`는 grouped one-shot holdout 전용 내부 연구 command이며 champion alias는 publish하지 않는다
- `benchmark-models`는 research `champion`과 execution-oriented `trading_champion`을 함께 publish한다
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
- 현재 기본 horizon policy는 `Seoul=market_open+previous_evening+morning_of`, `NYC=market_open+previous_evening`, `London=previous_evening`이다
