# storage and configs

## configs
- `configs/base.yaml`: shared defaults
- `configs/research.yaml`: research-oriented overrides
- `configs/paper.yaml`: paper-trading defaults
- `configs/live.example.yaml`: example live-trading configuration only
- `src/pmtmax/config/settings.py`: loads YAML plus environment variables into typed settings, including canonical warehouse paths and Firebase mirror settings

## storage
- `storage/duckdb_store.py`: writes research tables to DuckDB
- `storage/parquet_store.py`: writes portable Parquet artifacts
- `storage/raw_store.py`: archives raw JSON/HTML source payloads
- `storage/schemas.py`: shared typed records such as `ProbForecast`, `TradeSignal`, and model artifacts
- `storage/warehouse.py`: manages the canonical warehouse, writer lock, ingest runs, parquet mirrors, manifests, and legacy migration
- `storage/lab_bootstrap.py`: one-shot lab bootstrap, portable seed export/restore, and legacy raw/parquet cleanup helpers
- `storage/firebase_mirror.py`: builds Firebase Storage backup manifests and performs optional mirror syncs

## Runtime Paths
- `data/cache/`: cached HTTP payloads
- `data/raw/bronze/`: canonical raw source payloads keyed by market/source/date
- `data/duckdb/warehouse.duckdb`: canonical warehouse
- `data/parquet/bronze`, `data/parquet/silver`, `data/parquet/gold`: parquet mirrors and materialized datasets
- `data/manifests/`: generated warehouse manifest files
- `data/archive/`: archived legacy databases after migration
- `artifacts/`: trained models, metrics, and paper-trading outputs
- `data/duckdb/warehouse.duckdb.lock`: transient writer lock file only while canonical backfill/materialization commands are running
- `artifacts/bootstrap/pmtmax_seed.tar.gz`: portable seed archive for moving canonical data between machines

## What To Keep Stable
- Config names used in docs and scripts should stay consistent with the CLI.
- Storage schemas should only change with a matching downstream compatibility review, because the canonical warehouse is now the source of truth for long-history experiments.
- Example live config should remain instructional, not operational by default.

## Change Checklist
- Settings changes should be checked against `.env.example`, README, and `docs/live-trading.md`.
- Schema changes should be checked against CLI JSON outputs, migration tests, and the generated warehouse manifest.
