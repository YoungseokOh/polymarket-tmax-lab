# storage and configs

## configs
- `configs/base.yaml`: shared defaults
- `configs/research.yaml`: research-oriented overrides
- `configs/paper.yaml`: paper-trading defaults
- `configs/live.example.yaml`: example live-trading configuration only
- `configs/market_inventory/`: checked-in Polymarket event URL manifests and generated curated historical `MarketSnapshot[]` inventories
- `src/pmtmax/config/settings.py`: loads YAML plus environment variables into typed settings, including workspace-aware data/artifact roots, the shared public model registry, and Firebase mirror settings

## storage
- `storage/duckdb_store.py`: writes research tables to DuckDB
- `storage/parquet_store.py`: writes portable Parquet artifacts
- `storage/raw_store.py`: archives raw JSON/HTML source payloads
- `storage/schemas.py`: shared typed records such as `ProbForecast`, `TradeSignal`, and model artifacts
- `storage/warehouse.py`: manages the canonical warehouse, writer lock, ingest runs, parquet mirrors, manifests, and legacy migration
- `storage/lab_bootstrap.py`: one-shot lab bootstrap, portable seed export/restore, and legacy raw/parquet cleanup helpers
- `storage/firebase_mirror.py`: builds Firebase Storage backup manifests and performs optional mirror syncs

## Runtime Paths
- workspace launcher: `scripts/pmtmax-workspace <weather_train|ops_daily|historical_real|recent_core_eval> <command...>`
- preflight: `uv run pmtmax trust-check --markets-path <inventory>` validates workspace/inventory/canonical overwrite safety without mutating data
- weather preflight: `scripts/pmtmax-workspace weather_train uv run pmtmax trust-check --workflow weather_training`
- shared public alias registry: `artifacts/public_models/champion.pkl`, `artifacts/public_models/champion.json`
- workspace data roots: `data/workspaces/<workspace>/...`
- workspace artifact roots: `artifacts/workspaces/<workspace>/...`
- `data/workspaces/weather_train/parquet/gold/weather_training_set.parquet`: weather-real pretrain rows keyed by station/date/model, with no Polymarket market ids or prices
- `data/cache/`: legacy/default cached HTTP payloads when no workspace wrapper is used
- `data/raw/bronze/`: legacy/default raw source payloads keyed by market/source/date
- `data/duckdb/warehouse.duckdb`: legacy/default warehouse when no workspace wrapper is used
- `data/parquet/bronze`, `data/parquet/silver`, `data/parquet/gold`: legacy/default parquet mirrors and materialized datasets
- `data/manifests/`: legacy/default generated warehouse manifest files
- `data/manifests/historical_event_candidates.json`: persisted supported-city closed-event backlog discovered from Gamma grouped events
- `data/manifests/historical_event_page_fetches.json`: persisted grouped event page fetch state with `pending`, `fetched`, and `fetch_failed`
- `data/manifests/historical_collection_status.json`: grouped closed-event collection audit synced with the current canonical curated snapshot inventory
- `data/manifests/historical_inventory_build_report.json`: curated historical inventory build report
- `data/manifests/historical_inventory_validate_report.json`: curated historical inventory validation report
- `data/archive/`: archived legacy databases after migration
- `artifacts/`: legacy/default trained models, metrics, and paper-trading outputs
- `artifacts/active_weather_watchlist.json`: supported-city active grouped-event watchlist
- `artifacts/dataset_readiness.json`: city-level and market-level forecast/truth/gold readiness summary
- `artifacts/price_history_coverage.json`: official price-history request and decision-time coverage summary
- `data/workspaces/historical_real/parquet/gold/historical_backtest_panel.parquet`: decision-time official-price panel for real-history backtests
- `data/workspaces/<workspace>/duckdb/warehouse.duckdb.lock`: transient writer lock file only while workspace backfill/materialization commands are running
- `artifacts/bootstrap/pmtmax_seed.tar.gz`: portable seed archive for moving canonical data between machines

## What To Keep Stable
- Config names used in docs and scripts should stay consistent with the CLI.
- Storage schemas should only change with a matching downstream compatibility review, because the canonical warehouse is now the source of truth for long-history experiments.
- Example live config should remain instructional, not operational by default.
- Research CLI defaults should read live/public truth adapters, not `tests/fixtures/*`, unless a demo/test path opts into fixtures explicitly.
- Workspace boundaries should remain explicit: `weather_train`, `ops_daily`, `historical_real`, and `recent_core_eval` must never share DuckDB/parquet/artifact roots.
- Canonical storage is real-only; market workflows require `real_market`, weather pretrain requires `weather_real`, and trust-check fails closed on synthetic inventories, `synthetic_` market ids, fixture forecasts, fabricated books, and profile/workflow mismatches.
- Legacy mixed/synthetic roots should be quarantined and audited, not deleted silently.

## Change Checklist
- Settings changes should be checked against `.env.example`, README, and `docs/operations/live-trading.md`.
- Schema changes should be checked against CLI JSON outputs, migration tests, and the generated warehouse manifest.
