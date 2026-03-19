# Research Loop

## Use This When
- `materialize-training-set`
- `train-baseline`, `train-advanced`
- `backtest`
- `paper-trader`

## Default Workflow
```bash
uv run pmtmax bootstrap-lab
uv run pmtmax train-baseline --model-name gaussian_emos
uv run pmtmax backtest --model-name gaussian_emos
uv run pmtmax paper-trader --model-name gaussian_emos
```

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
- closed-event manifests: `data/manifests/historical_event_candidates.json`, `data/manifests/historical_event_page_fetches.json`, `data/manifests/historical_collection_status.json`
- active watchlist: `artifacts/active_weather_watchlist.json`

## Notes
- `backtest`는 최소 2개 row가 필요하다
- 단일 도시 smoke면 horizon을 2개 이상 주는 편이 안전하다
- 모델보다 settlement fidelity와 lookahead 방지가 우선이다
