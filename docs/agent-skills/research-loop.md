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

## Artifacts
- gold dataset: `data/parquet/gold/historical_training_set.parquet`
- sequence dataset: `data/parquet/gold/historical_training_set_sequence.parquet`
- model artifacts: `artifacts/models/`
- backtest outputs: `artifacts/backtest_metrics.json`, `artifacts/backtest_trades.json`
- paper outputs: `artifacts/paper_signals.json`

## Notes
- `backtest`는 최소 2개 row가 필요하다
- 단일 도시 smoke면 horizon을 2개 이상 주는 편이 안전하다
- 모델보다 settlement fidelity와 lookahead 방지가 우선이다
