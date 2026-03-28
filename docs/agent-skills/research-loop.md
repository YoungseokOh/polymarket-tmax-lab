# Research Loop

## Use This When
- `materialize-training-set`
- `train-baseline`, `train-advanced`
- `backtest`
- `paper-trader`
- `opportunity-report`
- `opportunity-shadow`
- `open-phase-shadow`

## Default Workflow
```bash
uv run pmtmax bootstrap-lab
uv run pmtmax build-dataset
uv run pmtmax materialize-backtest-panel
uv run pmtmax train-baseline --model-name gaussian_emos
uv run pmtmax benchmark-models
uv run pmtmax benchmark-ablations --model-name tuned_ensemble
uv run pmtmax backtest --model-name champion
uv run pmtmax paper-trader --core-recent-only --model-name trading_champion
uv run pmtmax opportunity-report --core-recent-only --model-name trading_champion
uv run pmtmax opportunity-shadow --core-recent-only --model-name trading_champion --max-cycles 1
uv run pmtmax open-phase-shadow --core-recent-only --model-name trading_champion --max-cycles 1
uv run pmtmax revenue-gate-report
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
uv run python scripts/build_historical_market_inventory.py
uv run python scripts/validate_historical_market_inventory.py
uv run pmtmax build-dataset --markets-path configs/market_inventory/historical_temperature_snapshots.json
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
- shadow/open-phase summary에는 `by_city`, `by_horizon`, `by_city_horizon`, `gate_decision`, `gate_reason`이 함께 기록된다
- `revenue-gate-report`는 recent-core benchmark와 두 shadow summary를 합쳐 소액 live pilot 전환 가능 여부를 `GO / INCONCLUSIVE / NO_GO`로 출력한다
- active opportunity 진단은 `raw_gap_non_positive`, `fee_killed_edge`, `slippage_killed_edge`, `after_cost_positive_but_spread_too_wide`를 분리해서 봐야 한다
- 현재 기본 horizon policy는 `Seoul=market_open+previous_evening+morning_of`, `NYC=market_open+previous_evening`, `London=previous_evening`이다
