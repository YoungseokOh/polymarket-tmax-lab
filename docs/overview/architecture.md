# Architecture

## Pipeline
1. `markets.gamma_client` fetches raw market/event metadata.
2. `markets.market_filter` keeps only recurring max-temperature contracts.
3. `markets.rule_parser` converts rule text into `MarketSpec`.
4. `weather.openmeteo_client` reconstructs admissible forecast inputs.
5. `weather.truth_sources.*` retrieves official settlement truth.
6. `backtest.dataset_builder` aligns predictors, target date, and realized value.
7. `modeling.*` fits probabilistic postprocessing models.
8. `modeling.daily_max` and `modeling.bin_mapper` transform forecasts into market outcome probabilities.
9. `execution.edge`, `fees`, `slippage`, and `guardrails` generate trade signals.
10. `execution.paper_broker` simulates conservative fills; `execution.live_broker` remains gated.

## Data Flow
- Gamma -> raw market metadata -> filtered temperature markets
- Rule text -> structured source/station/date/unit/outcomes
- Open-Meteo forecast archives -> station-aligned features
- Official source adapter -> realized daily max
- Dataset -> train/eval -> champion selection
- Champion forecast + live/public prices -> edge -> paper/live broker

## Design Principles
- settlement fidelity before model complexity
- no silent station substitution
- no lookahead in dataset building
- paper trading before live execution
- exact-source failure over fake completeness

## See Also
- `docs/codebase/index.md` for folder ownership and edit impact
- `docs/markets/market-rules.md` for supported rule families and parser expectations
