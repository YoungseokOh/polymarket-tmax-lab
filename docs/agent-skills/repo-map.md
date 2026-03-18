# Repo Map

## Start Here
- `AGENTS.md`: 공용 운영 계약
- `README.md`: 빠른 시작과 CLI 흐름
- `docs/codebase/index.md`: 폴더별 읽기 순서

## Major Runtime Areas
- `src/pmtmax/markets`: Gamma/CLOB, discovery, filtering, rule parsing, `MarketSpec`
- `src/pmtmax/weather`: Open-Meteo 입력, archived forecast, truth adapters
- `src/pmtmax/modeling`: baselines, advanced models, calibration, bin mapping
- `src/pmtmax/backfill`: bronze/silver/gold backfill과 materialization
- `src/pmtmax/backtest`: dataset builder, replay, metrics, PnL
- `src/pmtmax/execution`: edge, fees, slippage, sizing, brokers
- `src/pmtmax/storage`: DuckDB, Parquet, raw archive, warehouse, seed/bootstrap helpers

## Read Paths By Task
- 시장/규칙: `docs/codebase/markets.md`, `docs/market-rules.md`
- 날씨/진실값: `docs/codebase/weather.md`
- 모델/평가: `docs/codebase/modeling.md`, `docs/backtesting.md`
- 저장소/운영: `docs/codebase/storage-and-configs.md`, `data/README.md`
