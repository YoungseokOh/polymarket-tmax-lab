# Data Ops

## Use This When
- `bootstrap-lab`
- `export-seed`, `restore-seed`
- `init-warehouse`, `compact-warehouse`
- `inventory-legacy-runs`, `archive-legacy-runs`
- `data/raw`, `data/parquet`, `data/manifests`, `warehouse.duckdb`

## Canonical Paths
- canonical DB: `data/duckdb/warehouse.duckdb`
- canonical raw: `data/raw/bronze/`
- canonical parquet: `data/parquet/{bronze,silver,gold}/`
- manifests: `data/manifests/`
- archived legacy runs: `data/archive/legacy-runs/`

## One-Shot Flow
```bash
uv run pmtmax bootstrap-lab
```

이 명령은 기본적으로:
- legacy raw/parquet run 정리
- seed가 있으면 restore
- bundled historical snapshots backfill
- gold tabular/sequence dataset materialize
- forecast availability summary 생성
- warehouse manifest refresh

`bootstrap-lab`은 seed/demo 워크플로우다. 실제 historical Polymarket 시장을 canonical warehouse로 모을 때는 curated inventory 경로를 사용한다.

## Curated Historical Flow
```bash
scripts/run_historical_refresh_pipeline.sh
scripts/run_full_historical_batch.sh
```

`run_historical_refresh_pipeline.sh`는 closed-event refresh 전용 장기 배치다.
이 wrapper는 `discover -> fetch-pages -> classify -> publish`를 단계별로 나눠 실행하고, 기본 `--resume`로 candidate/page-fetch/status manifest를 이어서 갱신한다.
`--checkpoint-every`는 fetch/classify를 배치 단위로 checkpoint하고, `--fill-gaps-only`는 discover를 건너뛰고 이미 저장된 backlog에서 pending gap만 이어서 채운다.
`run_full_historical_batch.sh`는 `refresh -> build/validate -> backfill -> compact -> optional smoke -> watchlist`를 한 번에 묶는 warehouse rebuild 경로다.
기본은 incremental canonical update이고, 실행 로그는 `artifacts/batch_logs/`에 남는다.
`--city`를 주면 refresh/watchlist discovery와 backfill/materialization 입력을 같은 도시 집합으로 제한한다.

세부 단계를 직접 제어하고 싶으면:

```bash
uv run python scripts/refresh_historical_event_urls.py
uv run python scripts/build_historical_market_inventory.py
uv run python scripts/validate_historical_market_inventory.py
uv run pmtmax init-warehouse
uv run pmtmax backfill-markets --markets-path configs/market_inventory/historical_temperature_snapshots.json
uv run pmtmax backfill-forecasts --markets-path configs/market_inventory/historical_temperature_snapshots.json --strict-archive --single-run-horizon market_open --single-run-horizon previous_evening --single-run-horizon morning_of
uv run pmtmax backfill-truth --markets-path configs/market_inventory/historical_temperature_snapshots.json
uv run pmtmax materialize-training-set --markets-path configs/market_inventory/historical_temperature_snapshots.json --decision-horizon market_open --decision-horizon previous_evening --decision-horizon morning_of
uv run pmtmax summarize-forecast-availability
uv run pmtmax compact-warehouse
```

- source URL 목록: `configs/market_inventory/historical_temperature_event_urls.json`
- curated snapshot inventory: `configs/market_inventory/historical_temperature_snapshots.json`
- closed-event candidates: `data/manifests/historical_event_candidates.json`
- closed-event page fetches: `data/manifests/historical_event_page_fetches.json`
- closed-event collection status: `data/manifests/historical_collection_status.json`
- inventory report: `data/manifests/historical_inventory_report.json`
- active supported-city watchlist: `artifacts/active_weather_watchlist.json`

`refresh_historical_event_urls.py`는 Gamma grouped weather events에서 supported closed backlog를 찾아 candidate/page-fetch/status manifest를 남기고, `collected`만 append-only URL manifest에 publish한다.
retryable 상태는 `truth_source_lag`, `truth_request_failed`로 남기고 다음 batch에서 다시 classify할 수 있다.
`build_active_weather_watchlist.py`는 supported active grouped events를 parse-ready 상태로 정리하지만 canonical warehouse는 건드리지 않는다.
`run_historical_refresh_pipeline.sh`는 refresh 전용 shell wrapper고, `run_full_historical_batch.sh`는 refresh 이후 warehouse rebuild까지 포함하는 shell wrapper다.

## Portable Seed Flow
```bash
uv run pmtmax export-seed
uv run pmtmax restore-seed
```

- seed에는 `warehouse.duckdb`를 넣지 않는다
- raw/parquet/manifests만 옮기고 local warehouse는 다시 생성한다
- machine 간 ongoing sync가 아니라 cold bootstrap용이다

## Guardrails
- canonical writer는 한 번에 하나만 허용
- `warehouse.duckdb.lock`은 실행 중 일시 파일
- unsupported forecast coverage는 결측으로 남기고 숨기지 않는다
