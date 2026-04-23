# Release Checklist

## Use This When
- one-shot bootstrap 이후 연구 환경 점검
- `bootstrap-lab -> train-baseline -> backtest` 운영 점검
- 새 머신에서 실험 시작 전 체크리스트가 필요할 때

## Standard Research Checklist
1. `uv sync --all-extras`
2. `scripts/pmtmax-workspace historical_real uv run pmtmax trust-check --markets-path configs/market_inventory/full_training_set_snapshots.json`
3. `scripts/pmtmax-workspace historical_real uv run pmtmax build-dataset --markets-path configs/market_inventory/full_training_set_snapshots.json --forecast-missing-only --allow-canonical-overwrite`
4. `scripts/pmtmax-workspace historical_real uv run pmtmax materialize-backtest-panel --markets-path configs/market_inventory/full_training_set_snapshots.json --allow-canonical-overwrite`
5. `scripts/pmtmax-workspace historical_real uv run pmtmax train-baseline --model-name gaussian_emos`
6. `scripts/pmtmax-workspace historical_real uv run pmtmax backtest --pricing-source real_history --model-name gaussian_emos`
7. 필요하면 `scripts/pmtmax-workspace historical_real uv run pmtmax export-seed`

## Smoke Rules
- `backtest`는 최소 2개 row가 필요하다.
- 단일 도시 smoke면 horizon을 2개 이상 주는 편이 안전하다.
- strict archive가 기본이고, real-only dataset/backfill commands는 fixture fallback을 거부한다.
- backtest smoke는 official price-history panel이 필요하므로 `materialize-backtest-panel`을 먼저 통과해야 한다.
- `quote_proxy` 결과는 diagnostic only라 release readiness나 champion promotion 기준으로 쓰지 않는다.

## Expected Outputs
- `data/workspaces/historical_real/duckdb/warehouse.duckdb`
- `data/workspaces/historical_real/parquet/gold/historical_training_set.parquet`
- `data/workspaces/historical_real/parquet/gold/historical_backtest_panel.parquet`
- `artifacts/workspaces/historical_real/models/v2/*.pkl`
- `artifacts/workspaces/historical_real/backtests/v2/backtest_metrics_real_history.json`
- `artifacts/bootstrap/bootstrap_manifest.json`

## Machine Bootstrap Variant
- source machine: `uv run pmtmax export-seed`
- target machine: `uv run pmtmax restore-seed && uv run pmtmax bootstrap-lab`

## Guardrails
- live trading은 release checklist에 포함하지 않는다.
- seed는 raw/parquet/manifests만 옮기고 `warehouse.duckdb`는 옮기지 않는다.
- canonical writer는 한 번에 하나만 허용한다.
- synthetic inventories, `synthetic_` market ids, fixture forecasts, and fabricated books are release blockers.
