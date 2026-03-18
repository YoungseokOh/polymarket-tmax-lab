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
