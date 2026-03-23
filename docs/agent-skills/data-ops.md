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
uv run pmtmax collection-preflight --markets-path configs/market_inventory/historical_temperature_snapshots.json
uv run pmtmax init-warehouse
uv run pmtmax backfill-markets --markets-path configs/market_inventory/historical_temperature_snapshots.json
uv run pmtmax backfill-forecasts --markets-path configs/market_inventory/historical_temperature_snapshots.json --strict-archive --single-run-horizon market_open --single-run-horizon previous_evening --single-run-horizon morning_of
uv run pmtmax backfill-truth --markets-path configs/market_inventory/historical_temperature_snapshots.json
uv run pmtmax summarize-truth-coverage
uv run pmtmax summarize-dataset-readiness --markets-path configs/market_inventory/historical_temperature_snapshots.json
uv run pmtmax materialize-training-set --markets-path configs/market_inventory/historical_temperature_snapshots.json --decision-horizon market_open --decision-horizon previous_evening --decision-horizon morning_of
uv run pmtmax backfill-price-history --markets-path configs/market_inventory/historical_temperature_snapshots.json
uv run pmtmax materialize-backtest-panel --dataset-path data/parquet/gold/historical_training_set.parquet --markets-path configs/market_inventory/historical_temperature_snapshots.json
uv run pmtmax summarize-price-history-coverage --markets-path configs/market_inventory/historical_temperature_snapshots.json
uv run pmtmax summarize-forecast-availability
uv run pmtmax compact-warehouse
```

- source URL 목록: `configs/market_inventory/historical_temperature_event_urls.json`
- curated snapshot inventory: `configs/market_inventory/historical_temperature_snapshots.json`
- closed-event candidates: `data/manifests/historical_event_candidates.json`
- closed-event page fetches: `data/manifests/historical_event_page_fetches.json`
- closed-event collection status: `data/manifests/historical_collection_status.json`
- inventory build report: `data/manifests/historical_inventory_build_report.json`
- inventory validate report: `data/manifests/historical_inventory_validate_report.json`
- active supported-city watchlist: `configs/market_inventory/active_temperature_watchlist.json`
- recent core benchmark URLs: `configs/market_inventory/recent_core_temperature_event_urls.json`
- recent core benchmark snapshots: `configs/market_inventory/recent_core_temperature_snapshots.json`
- recent core benchmark config: `configs/recent-core-benchmark.yaml`
- recent core horizon policy: `configs/recent-core-horizon-policy.yaml`
- recent core benchmark runner: `scripts/run_recent_core_benchmark.py`

`refresh_historical_event_urls.py`는 Gamma grouped weather/temperature events에서 supported closed backlog를 찾아 candidate/page-fetch/status manifest를 남기고, `collected`만 append-only URL manifest에 publish한다.
retryable 상태는 `truth_source_lag`, `truth_request_failed`로 남기고 다음 batch에서 다시 classify할 수 있다.
`collection-preflight`는 curated snapshot 집합의 truth track과 수동 env 요구사항을 함께 요약한다. Wunderground-family markets는 기본적으로 같은 공항의 documented public truth를 사용하므로 `PMTMAX_WU_API_KEY`는 optional audit env로만 표시된다. 현재 Seoul / RKSI는 AMO `AIR_CALP`, London / EGLC와 NYC / KLGA는 Wunderground public historical API를 research truth 기본값으로 사용한다.
`build_historical_market_inventory.py`는 canonical snapshot output을 truth-ready subset으로만 만든다. `truth_source_lag`, `truth_blocked`, `truth_request_failed`는 snapshot JSON에 들어가지 않고 inventory report issue로 남는다. canonical 경로로 실행하면 `historical_collection_status.json`도 함께 sync해서 `collected` count가 현재 curated snapshot inventory와 맞도록 갱신한다.
`backfill-truth`에 `lag` 상태가 보이면 public archive가 target date보다 뒤처진 것이다. 이때 `summarize-truth-coverage`가 station별 최신 archive 일자를 JSON으로 내보내므로 truth-ready subset을 따로 고르거나 재시도 시점을 판단할 수 있다.
`summarize-dataset-readiness`는 snapshot 수집 성공과 warehouse materialization 성공을 분리해서 보여준다. 도시별 `snapshot_count`, `forecast_ready_count`, `truth_ok_count`, `truth_lag_count`, `gold_market_count`, `gold_row_count`와 market-level readiness detail을 함께 확인할 수 있다.
`backfill-price-history`는 official CLOB token history를 `bronze_price_history_requests`와 `silver_price_timeseries`에 적재한다. 성공 응답이 비어도 `empty` 상태로 남겨서 coverage gap을 숨기지 않는다.
`materialize-backtest-panel`은 gold training set의 각 decision timestamp마다 outcome token별 최신 공식 가격을 붙여 `gold_backtest_panel`을 만든다. `coverage_status=ok|stale|missing`이 모두 기록되고, real-history backtest는 `ok`만 사용한다.
`run_recent_core_benchmark.py`는 recent `Seoul` / `NYC` / `London` benchmark를 city별 격리 디렉터리로 재실행하고, summary JSON에 `real_history`, `quote_proxy`, `city x horizon` delta, 그리고 `configs/recent-core-horizon-policy.yaml` 기준 policy-filtered metrics까지 같이 적는다. 기존 산출물이 있으면 `--reuse-existing`로 summary만 다시 만들 수 있다.
`summarize-price-history-coverage`는 request-level 상태와 decision-time panel coverage를 함께 JSON으로 내보낸다. 따라서 “공식 가격이 실제로 남아 있는 market/date” subset을 바로 확인할 수 있다.
기본 research CLI는 더 이상 `tests/fixtures/truth`를 자동 주입하지 않는다. fixture truth는 테스트나 명시적 demo wiring에서만 사용한다.
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
