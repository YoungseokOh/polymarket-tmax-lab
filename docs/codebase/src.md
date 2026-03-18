# src/pmtmax

## Responsibility
`src/pmtmax` is the runtime package. It keeps market ingestion, weather data,
probabilistic modeling, backtesting, execution, and storage separated by concern.

## Package Layout
- `config/`: loads YAML and environment settings
- `backfill/`: bronze/silver/gold backfill orchestration and dataset materialization
- `markets/`: discovery, filtering, parsing, and structured settlement specs
- `weather/`: forecast inputs, archived forecast access, and official truth adapters
- `modeling/`: baselines, advanced models, calibration, and market-probability mapping
- `backtest/`: dataset construction, replay logic, metrics, and PnL
- `execution/`: edge, fees, slippage, sizing, guardrails, and brokers
- `storage/`: DuckDB/Parquet writes and shared schema objects
- `cli/`: Typer entrypoints

## Cross-Cutting Modules
- `examples.py`: bundled market templates used by tests and smoke workflows
- `http.py`: cached HTTP client shared across API adapters
- `logging_utils.py`: structured logger setup
- `utils.py`: small shared helpers

## Typical Read Order
- Start at `cli/main.py` to see public workflows.
- Follow the subsystem guide that matches the command you are changing.
- Check `storage/schemas.py` and `markets/market_spec.py` before changing shared data contracts.

## Change Checklist
- If you add a new major subpackage, document its boundary here and in `docs/codebase/index.md`.
- If you change shared contracts, check downstream use in dataset building, prediction, and execution.
