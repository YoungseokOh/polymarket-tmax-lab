# scripts, tests, and notebooks

## scripts
- `scan_markets.py`, `train_baseline.py`, `train_advanced.py`, `run_backtest.py`, and `run_paper_trader.py` mirror the CLI flows for shell-first usage.
- `bootstrap_data.py` is the operational helper for local data bootstrapping.
- Backfill and archive inspection are CLI-first now. Use `pmtmax init-warehouse`, `pmtmax backfill-*`, `pmtmax materialize-training-set`, `pmtmax summarize-forecast-availability`, `pmtmax migrate-legacy-warehouse`, and `pmtmax compact-warehouse` as the canonical entrypoints.
- Scripts should stay thin. Business logic belongs under `src/pmtmax/`.

## tests
- Unit coverage lives in `tests/test_rule_parser.py`, `test_market_spec.py`, `test_bin_mapper.py`, `test_daily_max.py`, `test_edge.py`, and `test_paper_broker.py`.
- `tests/test_backfill_pipeline.py` covers the bronze/silver/gold backfill smoke path, strict-archive behavior, and single-run horizon selection.
- `tests/test_integration_pipeline.py` covers the example model train/predict smoke path.
- Fixtures live under `tests/fixtures/` and back the repo's deterministic examples.

## notebooks
- Numbered notebooks mirror the standard workflow from rule parsing through backtesting.
- Notebooks are exploratory and explanatory. They should not become the only place where logic exists.

## Change Checklist
- If a script grows domain logic, move that logic into `src/pmtmax/` and keep the script as a wrapper.
- If a fixture changes behavior, check the matching unit or integration tests.
- If a notebook becomes stale after a workflow change, update it in the same change or document the gap.
