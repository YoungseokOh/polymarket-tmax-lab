# Data Layout

- `data/cache/`: cached HTTP responses and normalized fixture downloads.
- `data/raw/bronze/`: canonical bronze raw market, forecast, and truth payload archives.
- `data/duckdb/`: canonical DuckDB warehouse and temporary smoke or migration databases.
- `data/parquet/bronze/`, `data/parquet/silver/`, `data/parquet/gold/`: warehouse parquet mirrors and materialized datasets.
- `data/manifests/`: generated warehouse manifests and sync manifests.
- `data/runs/`: optional per-run scratch outputs when backfills are sharded or parallelized outside the canonical writer.
- `data/archive/`: archived legacy DuckDB files after migration into the canonical warehouse.
- `artifacts/`: trained model bundles, reports, and paper-trading logs.
- `artifacts/bootstrap/`: portable seed archives and one-shot bootstrap manifests.

Operational expectation:

- Active canonical DB: `data/duckdb/warehouse.duckdb`
- Legacy DuckDB after cutover: `data/archive/legacy-duckdb/*.duckdb`
- Legacy raw/parquet runs after cleanup: `data/archive/legacy-runs/{raw,parquet}/...`
- Cutover manifests: `data/manifests/legacy_inventory.json`, `data/manifests/migration_report.json`, `data/manifests/warehouse_manifest.json`
- Bootstrap manifests: `data/manifests/seed_manifest.json`, `artifacts/bootstrap/bootstrap_manifest.json`
- A temporary `data/duckdb/warehouse.duckdb.lock` file can appear while a canonical writer is running. It is not a second warehouse.
