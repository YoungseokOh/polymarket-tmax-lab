# scripts, tests, and notebooks

## scripts
- `scan_markets.py`, `train_baseline.py`, `train_advanced.py`, `run_backtest.py`, and `run_paper_trader.py` mirror the CLI flows for shell-first usage.
- `bootstrap_data.py` is the operational helper for local data bootstrapping.
- `refresh_historical_event_urls.py` is the staged, resumable closed-event refresh runner. It persists candidate, page-fetch, and collection-status manifests before publishing collected URLs.
- `run_historical_refresh_pipeline.sh` is the shell wrapper for long-running closed-event refresh jobs and refresh-only smoke runs.
- `build_historical_market_inventory.py` and `validate_historical_market_inventory.py` manage the curated grouped-event inventory used for real historical collection after URL manifest publication.
- `build_active_weather_watchlist.py` emits the supported-city active grouped-event watchlist without touching the canonical warehouse.
- `run_full_historical_batch.sh` is the long-running shell wrapper for supported-city historical expansion and research smoke.
- Backfill and archive inspection are CLI-first now. Use `pmtmax init-warehouse`, `pmtmax backfill-*`, `pmtmax materialize-training-set`, `pmtmax summarize-forecast-availability`, `pmtmax migrate-legacy-warehouse`, and `pmtmax compact-warehouse` as the canonical entrypoints.
- Scripts should stay thin. Business logic belongs under `src/pmtmax/`.

## tests
- Unit coverage lives in `tests/test_rule_parser.py`, `test_market_spec.py`, `test_bin_mapper.py`, `test_daily_max.py`, `test_edge.py`, and `test_paper_broker.py`.
- `tests/test_backfill_pipeline.py` covers the bronze/silver/gold backfill smoke path, strict-archive behavior, and single-run horizon selection.
- `tests/test_historical_inventory.py` covers event-page aggregation, resumable refresh manifests, and curated inventory validation.
- `tests/test_integration_pipeline.py` covers the example model train/predict smoke path.
- Fixtures live under `tests/fixtures/` and back the repo's deterministic examples.

## notebooks
- Numbered notebooks mirror the standard workflow from rule parsing through backtesting.
- Notebooks are exploratory and explanatory. They should not become the only place where logic exists.

## Change Checklist
- If a script grows domain logic, move that logic into `src/pmtmax/` and keep the script as a wrapper.
- If a fixture changes behavior, check the matching unit or integration tests.
- If a notebook becomes stale after a workflow change, update it in the same change or document the gap.
