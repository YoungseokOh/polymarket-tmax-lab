# Repo Map

## Start Here
- `AGENTS.md`: 공용 운영 계약
- `README.md`: 빠른 시작과 CLI 흐름
- `docs/overview/README.md`: 문서 폴더 구조
- `docs/codebase/index.md`: 폴더별 읽기 순서
- `checker/README.md`: ongoing dataset checker status/log/runbook

## Major Runtime Areas
- `src/pmtmax/markets`: Gamma/CLOB, discovery, filtering, rule parsing, `MarketSpec`
- `src/pmtmax/weather`: Open-Meteo 입력, archived forecast, truth adapters
- `src/pmtmax/modeling`: baselines, advanced models, calibration, bin mapping
- `src/pmtmax/backfill`: bronze/silver/gold backfill과 materialization
- `src/pmtmax/backtest`: dataset builder, replay, metrics, PnL
- `src/pmtmax/execution`: edge, fees, slippage, sizing, brokers
- `src/pmtmax/storage`: DuckDB, Parquet, raw archive, warehouse, seed/bootstrap helpers

## Read Paths By Task
- 시장/규칙: `docs/codebase/markets.md`, `docs/markets/market-rules.md`
- 날씨/진실값: `docs/codebase/weather.md`
- 모델/평가: `docs/codebase/modeling.md`, `docs/research/backtesting.md`
- 저장소/운영: `docs/codebase/storage-and-configs.md`, `docs/agent-skills/data-ops.md`
- 반복 데이터셋 체크:
  `checker/weather_train_status.md`, `checker/weather_train_collection_log.md`, `checker/weather_train_runbook.md`,
  `checker/historical_price_status.md`, `checker/historical_price_collection_log.md`, `checker/historical_price_runbook.md`,
  `checker/model_research_status.md`, `checker/model_research_log.md`, `checker/model_research_runbook.md`
- weather queue orchestrator: `scripts/run_weather_train_queue_agent.py`
- historical price recovery orchestrator: `scripts/run_historical_price_recovery_agent.py`
- model research orchestrator: `scripts/run_model_research_agent.py`
- daily 운영 루프: `docs/agent-skills/research-loop.md`, `scripts/daily_experiment.sh`
- release/검증: `docs/agent-skills/release-checklist.md`

## Workspace Boundaries
- `historical_real`: canonical real-only research data and model experiments.
- `ops_daily`: active collection, signals, paper diagnostics, and observation queues.
- `recent_core_eval`: isolated publish-gate benchmark runs.
- root `data/*` and non-public root `artifacts/*` paths are legacy/default unless a doc explicitly names them as public aliases.
