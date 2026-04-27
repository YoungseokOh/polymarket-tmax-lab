# Data Ops

## Use This When
- `bootstrap-lab`
- `export-seed`, `restore-seed`
- `init-warehouse`, `compact-warehouse`
- `inventory-legacy-runs`, `archive-legacy-runs`
- `data/raw`, `data/parquet`, `data/manifests`, `warehouse.duckdb`

## Canonical Paths
- weather pretrain workspace: `data/workspaces/weather_train/...`
- weather pretrain gold: `data/workspaces/weather_train/parquet/gold/weather_training_set.parquet`
- historical research DB: `data/workspaces/historical_real/duckdb/warehouse.duckdb`
- historical research raw: `data/workspaces/historical_real/raw/bronze/`
- historical research parquet: `data/workspaces/historical_real/parquet/{bronze,silver,gold}/`
- historical research manifests: `data/workspaces/historical_real/manifests/`
- ops daily workspace: `data/workspaces/ops_daily/...`
- recent-core gate workspace: `data/workspaces/recent_core_eval/...`
- root `data/duckdb`, `data/raw`, `data/parquet`, and `data/manifests` are legacy/default paths when no workspace wrapper is used
- archived legacy runs: `data/archive/legacy-runs/`

## One-Shot Flow
```bash
uv run pmtmax bootstrap-lab
```

기존 workspace warehouse나 seed restore 결과를 재사용하면서 forecast만 top-off하고
싶으면 `--forecast-missing-only`를 같이 준다.

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
`--checkpoint-every`는 fetch/classify를 배치 단위로 checkpoint하고, `--fill-gaps-only`는 discover를 건너뛰고 이미 저장된 backlog에서 retryable gap만 다시 classify한다.
lag recovery나 Wunderground-family repair에는 `--truth-no-cache`와 `--truth-per-source-limit 1`을 우선 사용한다.
`run_full_historical_batch.sh`는 `refresh -> build/validate -> backfill -> compact -> optional smoke -> watchlist`를 한 번에 묶는 warehouse rebuild 경로다.
기본은 incremental canonical update이고, 실행 로그는 `artifacts/batch_logs/`에 남는다.
`--city`를 주면 refresh/watchlist discovery와 backfill/materialization 입력을 같은 도시 집합으로 제한한다.

세부 단계를 직접 제어하고 싶으면:

```bash
scripts/pmtmax-workspace historical_real uv run python scripts/refresh_historical_event_urls.py
scripts/pmtmax-workspace historical_real uv run python scripts/build_historical_market_inventory.py
scripts/pmtmax-workspace historical_real uv run python scripts/validate_historical_market_inventory.py
scripts/pmtmax-workspace historical_real uv run pmtmax collection-preflight --markets-path configs/market_inventory/historical_temperature_snapshots.json
scripts/pmtmax-workspace historical_real uv run pmtmax trust-check --markets-path configs/market_inventory/historical_temperature_snapshots.json
scripts/pmtmax-workspace historical_real uv run pmtmax init-warehouse
scripts/pmtmax-workspace historical_real uv run pmtmax backfill-markets --markets-path configs/market_inventory/historical_temperature_snapshots.json
scripts/pmtmax-workspace historical_real uv run pmtmax backfill-forecasts --markets-path configs/market_inventory/historical_temperature_snapshots.json --model ecmwf_ifs025 --strict-archive --missing-only --single-run-horizon market_open --single-run-horizon previous_evening --single-run-horizon morning_of --max-consecutive-429 2
scripts/pmtmax-workspace historical_real uv run pmtmax backfill-truth --markets-path configs/market_inventory/historical_temperature_snapshots.json --truth-no-cache
scripts/pmtmax-workspace historical_real uv run pmtmax summarize-truth-coverage
scripts/pmtmax-workspace historical_real uv run pmtmax summarize-dataset-readiness --markets-path configs/market_inventory/historical_temperature_snapshots.json
scripts/pmtmax-workspace historical_real uv run pmtmax materialize-training-set --markets-path configs/market_inventory/historical_temperature_snapshots.json --decision-horizon market_open --decision-horizon previous_evening --decision-horizon morning_of --allow-canonical-overwrite
scripts/pmtmax-workspace historical_real uv run pmtmax backfill-price-history --markets-path configs/market_inventory/historical_temperature_snapshots.json --only-missing --price-no-cache --limit-markets 25
scripts/pmtmax-workspace historical_real uv run pmtmax materialize-backtest-panel --dataset-path data/workspaces/historical_real/parquet/gold/historical_training_set.parquet --markets-path configs/market_inventory/historical_temperature_snapshots.json --allow-canonical-overwrite
scripts/pmtmax-workspace historical_real uv run pmtmax summarize-price-history-coverage --markets-path configs/market_inventory/historical_temperature_snapshots.json
scripts/pmtmax-workspace historical_real uv run pmtmax summarize-forecast-availability
scripts/pmtmax-workspace historical_real uv run pmtmax compact-warehouse
```

기존 canonical warehouse를 이어서 채우는 incremental run이면
`backfill-forecasts`에 `--missing-only`를 추가해서
`bronze_forecast_requests`에 없는 key만 수집할 수 있다.
historical forecast backfill도 `--max-consecutive-429 2`를 유지한다. 두 번
연속 `429`가 나오면 이미 적재한 행을 flush한 뒤 취소하고 checker에
`rate-limit-cancelled`로 기록한다.
multi-source variant를 만들 때는 `backfill-forecasts`, `build-dataset`,
`materialize-training-set`에 `--model`을 반복해서 명시한다. base config가
`gfs_seamless`만 가리키면 row count는 같아도 feature schema가 GFS-only로
축소될 수 있다.
`scripts/run_full_historical_batch.sh`는 이 동작을 기본 forecast backfill로 사용한다.
`bootstrap-lab`도 `--forecast-missing-only`로 같은 top-off semantics를 쓸 수 있다.

`configs/market_inventory/historical_temperature_snapshots.json`은 canonical curated
historical collection inventory다. 반면
`configs/market_inventory/full_training_set_snapshots.json`은 checked-in training
inventory라서 자동으로 sync되지 않는다. training inventory를 갱신해야 하면
먼저 `historical_temperature_snapshots.json`을 refresh/build/validate한 뒤,
의도적으로 `full_training_set_snapshots.json`을 다시 curate해서 같은 change에
체크인한다.

## Current Inventory Semantics
- `weather_train` is a `weather_real` profile for station/date real weather
  pretraining. It is not a Polymarket dataset and must not contain market ids,
  market rule JSON, CLOB books, price history, or champion publish evidence.
- `full_training_set_snapshots.json` currently represents the trusted checked-in
  training market inventory. The latest audit count was 1,834 market snapshots.
- `historical_training_set.parquet` materializes model rows from that inventory
  and supported decision horizons. The latest trusted build had 5,478 training rows.
- `historical_temperature_snapshots.json` is the curated collection backlog, not
  the automatically-used training set. Its latest local audit count was `2,602`
  snapshots after the April 27, 2026 discovery120 expansion. Follow-up
  discovery160 appended `0` URLs, so the latest collection turn stopped on the
  no-new-truth-ready-backlog condition.
- More real data can be collected. The hard rule is that new markets must be
  closed, parseable, truth-ready, forecast-backed, and intentionally curated
  before replacing the checked-in training inventory.
- Daily `ops_daily` collection records active-market evidence for the future; it
  does not automatically append to `full_training_set_snapshots.json` or rebuild
  `historical_real`.
- Open-Meteo bulk collection belongs in `weather_train`. Start with small,
  resumable date chunks, `--missing-only`, workers=1, and the free rate-limit
  profile unless an API-key/paid profile is intentionally added. The collector
  prints station/date progress to stderr by default; keep shorter
  `--http-timeout-seconds` / `--http-retries` settings when monitoring long
  free-tier runs. Keep `--max-consecutive-429 2`: two consecutive Open-Meteo
  `429` responses are treated as a free-path daily-limit hit, not a signal to
  keep burning requests.
- For repeated older-gap backfill, prefer
  `scripts/pmtmax-workspace weather_train uv run python scripts/run_weather_train_queue_agent.py`.
  It reads the checker state, advances the next `7`-day queue, updates both
  checker markdown files after every chunk, auto-refreshes weather pretrain
  when the row-gap threshold is hit, and cancels the remaining chunk after two
  consecutive `429` responses. Record that as `rate-limit-cancelled` and retry
  only after cooldown/reset or an API-key path.
- For daily official price-history recovery, prefer
  `scripts/pmtmax-workspace historical_real uv run python scripts/run_historical_price_recovery_agent.py`.
  It reads `checker/historical_price_status.md`, advances the next missing-price
  shard, rebuilds the canonical panel, refreshes the latest coverage JSON, and
  updates both historical-price checker markdown files.
- `weather_train` queue collection can run in parallel with
  `historical_real backfill-price-history` because those jobs use separate
  workspace roots and different upstream APIs. Do not overlap multiple mutating
  `historical_real` jobs in the same workspace.
- Keep `checker/weather_train_status.md` and
  `checker/weather_train_collection_log.md` in sync after every weather-train
  collection attempt so the next turn can resume from verified state.
- Keep `checker/historical_price_status.md` and
  `checker/historical_price_collection_log.md` in sync after every daily price
  recovery run so shard rotation and decision-row coverage are resumable.
- Current observed Open-Meteo state should be read from
  `checker/weather_train_status.md`; do not duplicate row counts in this doc.
- Old official CLOB `/prices-history` gaps may be retention-limited. Preserve
  `empty` / `missing` coverage explicitly instead of filling with synthetic prices.
- Forecast materialization must preserve feature validity. All-zero target-day
  temperature aggregates from Open-Meteo are treated as invalid source features,
  not real `0C` / `32F` forecasts. Re-materialize a non-canonical variant after
  changing this logic and compare source availability plus sentinel counts before
  retraining.
- To improve `priced_decision_rows`, use this order inside `historical_real`:
  `summarize-price-history-coverage` -> `backfill-price-history --only-missing --price-no-cache` ->
  `materialize-backtest-panel`. Keep those three steps serialized even if
  weather collection is running elsewhere.
- Use `checker/historical_price_runbook.md` as the operational checklist for the
  daily price agent. The checker board should track the next shard, the current
  token coverage, `panel-ready decision rows`, and the latest backtest
  `priced_decision_rows` anchor.

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
retryable 상태는 `not_closed`, `not_historical`, `truth_source_lag`, `truth_request_failed`다. `not_closed`/`not_historical`는 미래 시장이 나중에 historical closed subset으로 넘어올 때를 위한 상태고, `--fill-gaps-only`는 한 invocation 안에서 각 URL을 한 번만 다시 classify한다.
`collection-preflight`는 curated snapshot 집합의 truth track과 수동 env 요구사항을 함께 요약한다. Wunderground-family markets는 기본적으로 같은 공항의 documented public truth를 사용하므로 `PMTMAX_WU_API_KEY`는 optional audit env로만 표시된다. 현재 Seoul / RKSI는 AMO `AIR_CALP`, London / EGLC와 NYC / KLGA는 Wunderground public historical API를 research truth 기본값으로 사용한다.
`build_historical_market_inventory.py`는 canonical snapshot output을 truth-ready subset으로만 만든다. `truth_source_lag`, `truth_blocked`, `truth_request_failed`는 snapshot JSON에 들어가지 않고 inventory report issue로 남는다. canonical 경로로 실행하면 `historical_collection_status.json`도 함께 sync해서 `collected` count가 현재 curated snapshot inventory와 맞도록 갱신한다.
Wunderground-family truth probe는 source-family 동시성에 민감하다. 기본 `truth_per_source_limit=1`을 유지하고, cached truth payload가 의심될 때만 `--truth-no-cache`를 추가한다.
`backfill-truth`에 `lag` 상태가 보이면 public archive가 target date보다 뒤처진 것이다. 이때 `summarize-truth-coverage`가 station별 최신 archive 일자를 JSON으로 내보내므로 truth-ready subset을 따로 고르거나 재시도 시점을 판단할 수 있다.
`summarize-dataset-readiness`는 snapshot 수집 성공과 warehouse materialization 성공을 분리해서 보여준다. 도시별 `snapshot_count`, `forecast_ready_count`, `truth_ok_count`, `truth_lag_count`, `gold_market_count`, `gold_row_count`와 market-level readiness detail을 함께 확인할 수 있다.
`backfill-price-history`는 official CLOB token history를 `bronze_price_history_requests`와 `silver_price_timeseries`에 적재한다. 성공 응답이 비어도 `empty` 상태로 남겨서 coverage gap을 숨기지 않는다. gap recovery는 `--only-missing --price-no-cache`를 사용해서 이미 official price point가 있는 token은 건너뛰고, 오래된 empty cache만 우회한다. 큰 backlog는 `--limit-markets` / `--offset-markets`나 `--target-date-from` / `--target-date-to`로 나눠 실행한다.
public CLOB `/prices-history`는 오래된 시장에서 retention-limited하게 동작할 수 있다. 이미 raw/bronze/silver에 저장된 official history가 있으면 `materialize-backtest-panel`과 `summarize-price-history-coverage` 재실행을 먼저 고려한다.
`materialize-backtest-panel`은 gold training set의 각 decision timestamp마다 outcome token별 최신 공식 가격을 붙여 `gold_backtest_panel`을 만든다. `coverage_status=ok|stale|missing`이 모두 기록되고, real-history backtest는 `ok`만 사용한다.
`run_recent_core_benchmark.py`는 recent `Seoul` / `NYC` / `London` benchmark를 city별 격리 디렉터리로 재실행하고, summary JSON에 `real_history`, 진단용 `quote_proxy`, `city x horizon` delta, `configs/recent-core-horizon-policy.yaml` 기준 policy-filtered metrics, 그리고 top-level aggregate profitability fields (`aggregate_*`, `aggregate_panel_coverage`, `city_gate_details`, `reduced_core_candidate`, `decision`, `decision_reason`, `sample_adequacy`)까지 같이 적는다. promotion gate와 city gate는 `real_history`를 기준으로 판단하며, `quote_proxy`는 diagnostic field일 뿐이다. `reduced_core_candidate`는 coverage-thin 도시를 제외한 진단용 후보이고 publish gate로는 사용할 수 없다. LGBM 후보는 `--model-name lgbm_emos --variant <variant>`로 benchmark 대상과 publish artifact를 명시적으로 맞춘다. 이미 검증된 historical-real parquet에서 재실행할 때는 `--prebuilt-dataset-path`와 `--prebuilt-panel-path`를 함께 넘겨 외부 forecast/price refetch 없이 city별 parquet를 slice하고, `--prebuilt-last-n-market-days`, `--backtest-last-n`, `--retrain-stride`로 training/test window를 명시한다. 도시별 split이 작은 recent-core run에서는 `--retrain-stride 30`을 쓰지 않는다. runner가 available test split보다 큰 retrain stride를 fail-fast로 막는다. 일반 실행은 `scripts/run_recent_core_benchmark_local.sh`로 `recent_core_eval` workspace를 잡아서 돌린다. 기존 산출물이 있으면 같은 model/variant summary 재계산에 한해서 `--reuse-existing`를 쓴다.
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
- `restore-seed`는 parquet mirror를 DuckDB에서 직접 읽어 warehouse를 재구성한다
- machine 간 ongoing sync가 아니라 cold bootstrap용이다

## Guardrails
- canonical writer는 한 번에 하나만 허용
- `warehouse.duckdb.lock`은 실행 중 일시 파일
- unsupported forecast coverage는 결측으로 남기고 숨기지 않는다
- canonical `gold/v2/historical_training_set*`와 `gold/v2/historical_backtest_panel`은 기본 immutable이다
- canonical overwrite는 `--allow-canonical-overwrite`가 있어야만 가능하고, 실행 전 `artifacts/recovery/<timestamp>/`에 backup이 생성된다
- 실험/ablation은 canonical 경로를 덮지 말고 variant `--output-name`을 사용한다
- use `uv run pmtmax trust-check --markets-path <inventory>` before overwrite or long collection
- synthetic inventories, `synthetic_` market ids, fixture forecasts, and fabricated books make a warehouse non-canonical
- legacy mixed data should be quarantined and audited, not deleted silently
- physical legacy cleanup must write an `artifacts/recovery/data_cleanup_<timestamp>.json`
  manifest before deletion, with path, reason, size, file count, and checksums where practical
