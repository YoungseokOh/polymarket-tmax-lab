# Research Loop

## Use This When
- `materialize-training-set`
- `train-baseline`, `train-advanced`
- `backtest`
- `paper-trader`
- `opportunity-report`
- `opportunity-shadow`
- `open-phase-shadow`
- `hope-hunt-report`
- `hope-hunt-daemon`

## Current Champion (as of 2026-04-03)
- Model: `lgbm_emos`, variant: `recency_neighbor_fast`, CRPS: 0.4769
- pkl: `artifacts/models/v2/lgbm_emos__recency_neighbor_fast.pkl`
- champion.json: `artifacts/models/v2/champion.json`

## Daily Data Collection (cron 00:00 UTC = 09:00 KST)
```bash
bash scripts/daily_experiment.sh
```
Runs: scan-markets → backfill-truth → backfill-forecasts → log_gamma_prices → scan-edge → track_paper_trade_outcomes

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
uv run pmtmax train-advanced --model-name lgbm_emos --variant recency_neighbor_fast

# Publish as champion
uv run pmtmax publish-champion --model-name lgbm_emos --variant recency_neighbor_fast

# Quick comparison (top variants only, fast)
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
uv run pmtmax train-advanced --model-name lgbm_emos --variant recency_neighbor_fast
uv run python scripts/quick_eval.py
uv run pmtmax benchmark-models --retrain-stride 30
```

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
수익화 전용 루프에서는 `--core-recent-only`와 `--model-name trading_champion` 조합을 기본으로 쓴다.
recent-core 바깥의 fresh listing 탐색은 `--market-scope supported_wu_open_phase`
또는 전용 wrapper인 `hope-hunt-report` / `hope-hunt-daemon`을 쓴다.

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
- `open-phase-shadow`는 `componentMarkets[*].acceptingOrdersTimestamp` 기준 최근 상장 시장만 골라 `market_open` horizon으로 평가하는 실험 경로다
- `hope-hunt-report`와 `hope-hunt-daemon`은 `supported_wu_open_phase` 범위에서 fresh listing을 우선순위화하는 no-order candidate loop다
- shadow/open-phase summary에는 `by_city`, `by_horizon`, `by_city_horizon`, `gate_decision`, `gate_reason`이 함께 기록된다
- hope-hunt summary에는 `by_open_phase_age_bucket`, `by_priority_bucket`, `top_candidates`, `gate_decision`, `gate_reason`이 함께 기록된다
- `revenue-gate-report`는 recent-core benchmark와 두 shadow summary를 합쳐 소액 live pilot 전환 가능 여부를 `GO / INCONCLUSIVE / NO_GO`로 출력한다
- active opportunity 진단은 `raw_gap_non_positive`, `fee_killed_edge`, `slippage_killed_edge`, `after_cost_positive_but_spread_too_wide`를 분리해서 봐야 한다
- 현재 기본 horizon policy는 `Seoul=market_open+previous_evening+morning_of`, `NYC=market_open+previous_evening`, `London=previous_evening`이다
