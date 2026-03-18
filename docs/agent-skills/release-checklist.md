# Release Checklist

## Use This When
- one-shot bootstrap 이후 연구 환경 점검
- `bootstrap-lab -> train-baseline -> backtest` 운영 점검
- 새 머신에서 실험 시작 전 체크리스트가 필요할 때

## Standard Research Checklist
1. `uv sync --all-extras`
2. `uv run pmtmax bootstrap-lab`
3. `uv run pmtmax train-baseline --model-name gaussian_emos`
4. `uv run pmtmax backtest --model-name gaussian_emos`
5. 필요하면 `uv run pmtmax export-seed`

## Smoke Rules
- `backtest`는 최소 2개 row가 필요하다.
- 단일 도시 smoke면 horizon을 2개 이상 주는 편이 안전하다.
- strict archive가 기본이고, fixture fallback은 데모에서만 연다.

## Expected Outputs
- `data/duckdb/warehouse.duckdb`
- `data/parquet/gold/historical_training_set.parquet`
- `data/parquet/gold/historical_training_set_sequence.parquet`
- `artifacts/models/*.pkl`
- `artifacts/backtest_metrics.json`
- `artifacts/bootstrap/bootstrap_manifest.json`

## Machine Bootstrap Variant
- source machine: `uv run pmtmax export-seed`
- target machine: `uv run pmtmax restore-seed && uv run pmtmax bootstrap-lab`

## Guardrails
- live trading은 release checklist에 포함하지 않는다.
- seed는 raw/parquet/manifests만 옮기고 `warehouse.duckdb`는 옮기지 않는다.
- canonical writer는 한 번에 하나만 허용한다.
