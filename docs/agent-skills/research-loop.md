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
uv run pmtmax train-baseline --model-name gaussian_emos
uv run pmtmax backtest --model-name gaussian_emos
uv run pmtmax paper-trader --model-name gaussian_emos
uv run pmtmax opportunity-report --model-name gaussian_emos
uv run pmtmax opportunity-shadow --model-name gaussian_emos --max-cycles 1
uv run pmtmax open-phase-shadow --model-name gaussian_emos --max-cycles 1
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
- gold dataset: `data/parquet/gold/historical_training_set.parquet`
- sequence dataset: `data/parquet/gold/historical_training_set_sequence.parquet`
- model artifacts: `artifacts/models/`
- backtest outputs: `artifacts/backtest_metrics.json`, `artifacts/backtest_trades.json`
- paper outputs: `artifacts/paper_signals.json`
- opportunity outputs: `artifacts/opportunity_report.json`
- shadow validation outputs: `artifacts/opportunity_shadow.jsonl`, `artifacts/opportunity_shadow_latest.json`, `artifacts/opportunity_shadow_summary.json`
- open-phase validation outputs: `artifacts/open_phase_shadow.jsonl`, `artifacts/open_phase_shadow_latest.json`, `artifacts/open_phase_shadow_summary.json`
- closed-event manifests: `data/manifests/historical_event_candidates.json`, `data/manifests/historical_event_page_fetches.json`, `data/manifests/historical_collection_status.json`
- active watchlist: `artifacts/active_weather_watchlist.json`

## Notes
- `backtest`는 최소 2개 row가 필요하다
- 단일 도시 smoke면 horizon을 2개 이상 주는 편이 안전하다
- 모델보다 settlement fidelity와 lookahead 방지가 우선이다
- active market 탐색에서는 `missing_book`과 `no_positive_edge`를 구분해서 해석해야 한다
- `opportunity-shadow`는 주문/알림 없이 `raw_gap`, `after_cost_edge`, reject reason을 시간축으로 쌓아 현재 탐색 로직이 실제로 기회를 잡는지 검증하는 경로다
- `open-phase-shadow`는 `componentMarkets[*].acceptingOrdersTimestamp` 기준 최근 상장 시장만 골라 `market_open` horizon으로 평가하는 실험 경로다
- active opportunity 진단은 `raw_gap_non_positive`, `fee_killed_edge`, `slippage_killed_edge`, `after_cost_positive_but_spread_too_wide`를 분리해서 봐야 한다
- 현재 기본 horizon policy는 `Seoul=market_open+previous_evening+morning_of`, `NYC=market_open+previous_evening`, `London=previous_evening`이다
