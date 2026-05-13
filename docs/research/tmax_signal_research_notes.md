# TMAX Signal Research Notes

Updated: 2026-05-13 21:55 KST

## Current Dataset / Quality State

- Workspace: `historical_real`
- Canonical gold dataset: **7,806 rows** / **2,602 markets** / **30 cities** / 3 horizons per market.
- Dataset quality score: **8.5/10** (`docs/research/historical_dataset_card.md`) after removing the constant `neighbor_spread` feature and adding explicit city sample tiering / exposure weighting policy.
- Readiness is now clean for canonical inventory: **2,602 ready**, **0 non-ready**.
- Official price panel: **68,682 rows** with **57,733 ok**, **10,435 missing**, **514 stale**.
- Price coverage is no longer Seoul-only. All 30 cities have at least some `ok` rows after the 2026-05-13 recovery pass.

Top official price `ok` city coverage:

| city | ok rows |
| --- | ---: |
| NYC | 7,021 |
| London | 6,175 |
| Dallas | 3,508 |
| Atlanta | 3,493 |
| Buenos Aires | 3,477 |
| Seattle | 3,223 |
| Toronto | 3,214 |
| Chicago | 2,557 |
| Miami | 2,384 |
| Seoul | 2,125 |

Remaining non-ok blockers are concentrated in older/liquidity-limited CLOB history:

| city | missing rows | note |
| --- | ---: | --- |
| London | 3,305 | no-cache retry returned 0 rows; likely history unavailable / retention-limited |
| NYC | 1,759 | no-cache retry returned 0 rows; likely history unavailable / retention-limited |
| Seoul | 891 | mostly market-open decision-time gaps |
| Wellington | 604 | mostly market-open decision-time gaps |
| Ankara | 503 | mostly market-open decision-time gaps |

## Current-Panel Model Benchmark

Artifacts:

- `artifacts/workspaces/historical_real/benchmarks/v2/model_leaderboard_20260513_current_panel.csv`
- `artifacts/workspaces/historical_real/benchmarks/v2/model_leaderboard_20260513_current_panel.json`
- `artifacts/workspaces/historical_real/benchmarks/v2/model_benchmark_summary_20260513_current_panel.json`

Benchmark settings: `retrain_stride=30`, `seed=42`, `split_policy=market_day`, `--no-publish-champion`.

| model | champion score | MAE | RMSE | avg CRPS | real-history PnL | hit rate | trades | quote-proxy PnL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gaussian_emos` | **1.55** | **1.5457** | **2.4107** | **1.2048** | 2,695.10 | 26.78% | 6,224 | 478.41 |
| `tuned_ensemble` | 1.65 | 1.6111 | 2.5732 | 1.2404 | 3,540.07 | **30.54%** | 6,152 | **1,066.24** |
| `lgbm_emos` | 2.65 | 2.6640 | 6.0198 | 2.1214 | **3,965.45** | 25.94% | 6,283 | 896.82 |

Interpretation:

- **Champion remains `gaussian_emos`** because it has the best forecast-quality composite: lowest MAE/RMSE/CRPS and best champion score.
- **Execution candidate is `tuned_ensemble`** because it has materially better PnL/hit-rate than champion while forecast degradation is modest.
- **Do not publish `lgbm_emos` as champion** despite highest PnL. Its RMSE and CRPS are much worse, suggesting unstable or overfit tails.

## Trade-Level Diagnostics

Artifacts:

- `artifacts/workspaces/historical_real/quality/backtest_trade_diagnostics_current_panel.{json,md}`
- `artifacts/workspaces/historical_real/quality/backtest_trade_diagnostics_quote_proxy.{json,md}`

Current-panel real-history all-trades baseline:

- Trades: **6,201**
- PnL: **2,670.38**
- PnL/trade: **0.4306**
- Hit rate: **26.79%**
- Avg price: **0.2131**
- Avg edge: **0.3780**

Quote-proxy conservative diagnostic is still thin and much weaker:

- Trades: **242**
- PnL: **-86.29**
- Hit rate: **11.98%**
- Best robust-ish region remains mid-price contracts:
  - `min_price=0.20`, `max_price=0.60`: +3.40 over 55 trades
  - `min_edge=0.15`, `min_price=0.15`, `max_price=0.60`: +3.14 over 67 trades

## Research Conclusion

Current TMAX is now **stronger than the 2026-05-12 state** because official price coverage expanded from Seoul-only to all 30 cities and the canonical inventory is fully ready.

Still not production-ready:

1. Truth fidelity is still mostly `research_public` rather than high-confidence exact settlement truth.
2. Missing/stale price rows remain large enough to bias execution evaluation, especially London/NYC.
3. Small-sample cities are now explicitly tiered as exploratory and city exposure weighting is documented, but model-level city/time split robustness still needs to be run before broad claims.
4. `tuned_ensemble` looks promising for execution, but needs time/city split robustness checks before publishing or using it as a live execution candidate.
5. Quote-proxy diagnostics remain much weaker than real-history PnL, so execution friction can still erase apparent edge.

## Recommended Next Steps

1. Keep `gaussian_emos` as unpublished champion for now; do not publish new champion yet.
2. Run city/time split model benchmarks comparing `gaussian_emos` vs `tuned_ensemble` with conservative quote-proxy scoring.
3. Add a dedicated execution-policy benchmark for mid-price filters (`15–60c`, `20–60c`) across time folds and city groups.
4. Classify remaining non-ok coverage into:
   - retention-limited / unavailable (`London`, `NYC` confirmed by no-cache zero-row retries),
   - decision-time gaps (`market_open` heavy groups),
   - genuinely recoverable gaps.
5. Keep `neighbor_spread` excluded unless a future multi-model/spatial source can populate it with non-constant values.
