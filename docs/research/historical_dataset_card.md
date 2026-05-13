# Historical Real Dataset Card

Generated: `2026-05-13T12:55:48.327923+00:00`

## Verdict

- Quality score: **8.5/10** (research-grade)
- Check statuses: `{'pass': 11, 'warn': 1}`
- Rows: **7806** / Markets: **2602** / Cities: **30**
- Target date range: **2025-05-30 → 2026-04-26**

## Core Coverage

- Horizon counts: `{'market_open': 2602, 'previous_evening': 2602, 'morning_of': 2602}`
- Truth tracks: `{'research_public': 7791, 'exact_public': 15}`
- Settlement eligible: `{'False': 7791, 'True': 15}`
- Forecast source kind: `{'historical_forecast': 6658, 'single_run': 1148}`
- Readiness statuses: `{'ready': 2602}`
- City sample tiering: `{'enabled': True, 'primary_min_rows': 90, 'primary_city_count': 25, 'exploratory_city_count': 5, 'exploratory_cities': {'Shenzhen': 84, 'Wuhan': 84, 'Tel Aviv': 39, 'Taipei': 15, 'Hong Kong': 6}, 'policy': 'Cities below the minimum row threshold remain in the raw dataset but are treated as exploratory and excluded from broad cross-city quality claims.', 'raw_max_city_ratio': 0.1618}`
- City exposure weighting: `{'enabled': True, 'city_exposure_cap': 0.15, 'raw_max_city_ratio': 0.1618, 'capped_cities': {'London': 0.927078, 'NYC': 0.995663}, 'policy': 'Evaluation reports should apply city-balanced sample weights when making aggregate claims; raw rows stay unchanged for reproducibility.'}`

## Baseline Forecast Quality

`model_daily_max` vs `realized_daily_max`:

- rows: `7806`
- bias: `-0.4974`
- MAE: `1.5662`
- RMSE: `2.3271`
- p95 abs error: `4.38`
- p99 abs error: `8.12`

## Time-Forward Baseline Benchmark

Macro fold MAE: `1.4697`; MAE range: `[1.3038, 1.6257]`

| fold | start | end | rows | mae | bias |
| --- | --- | --- | --- | --- | --- |
| 1 | 2025-08-04 | 2025-10-08 | 396 | 1.3038 | 0.6862 |
| 2 | 2025-10-09 | 2025-12-14 | 588 | 1.3656 | -0.3065 |
| 3 | 2025-12-15 | 2026-02-18 | 2103 | 1.5837 | -0.6789 |
| 4 | 2026-02-19 | 2026-04-26 | 4323 | 1.6257 | -0.6337 |

## City Holdout Diagnostic

Macro city MAE: `1.5081` across `30` cities.

Worst 10 cities by baseline MAE:

| city | rows | mae | bias |
| --- | --- | --- | --- |
| Dallas | 426 | 4.5896 | -3.7318 |
| Hong Kong | 6 | 3.4 | -3.4 |
| Seoul | 369 | 2.6114 | -1.5415 |
| Lucknow | 153 | 2.249 | 2.249 |
| Chicago | 285 | 2.1324 | 0.2777 |
| Chengdu | 108 | 1.8972 | -1.6306 |
| Wellington | 258 | 1.6911 | -1.5469 |
| Chongqing | 108 | 1.6889 | -1.4389 |
| Munich | 129 | 1.4791 | -1.3349 |
| Singapore | 105 | 1.46 | -1.2429 |

Small-sample cities: `['Shenzhen', 'Wuhan', 'Tel Aviv', 'Taipei', 'Hong Kong']`

City tiering policy: `Cities below the minimum row threshold remain in the raw dataset but are treated as exploratory and excluded from broad cross-city quality claims.`

City weighting policy: `Evaluation reports should apply city-balanced sample weights when making aggregate claims; raw rows stay unchanged for reproducibility.`

Top city row counts:

| city | rows |
| --- | --- |
| London | 1263 |
| NYC | 1176 |
| Dallas | 426 |
| Atlanta | 423 |
| Buenos Aires | 420 |
| Seattle | 396 |
| Toronto | 396 |
| Seoul | 369 |
| Chicago | 285 |
| Ankara | 279 |
| Miami | 267 |
| Wellington | 258 |
| Paris | 186 |
| Sao Paulo | 186 |
| Lucknow | 153 |

## Leakage / Feature Audit Checks

| status | check | detail |
| --- | --- | --- |
| pass | dataset_exists | data/workspaces/historical_real/parquet/gold/historical_training_set.parquet |
| pass | non_empty | rows=7806 |
| pass | duplicate_rows | duplicate_rows=0 |
| pass | null_cells | null_cells=0 |
| pass | three_horizons_per_market | markets_without_3_horizons=0 |
| pass | issue_time_not_after_decision_time | invalid_rows=0 |
| pass | no_obvious_leaky_feature_names | leaky_named_features=[] |
| pass | no_constant_numeric_features | constant_features=[] |
| pass | truth_tier_explicit | truth_track_counts={'research_public': 7791, 'exact_public': 15} |
| warn | exact_truth_coverage | Most rows are research_public; keep as research-grade or tier exact/proxy datasets. |
| pass | readiness_non_ready_small | non_ready=0, status_counts={'ready': 2602} |
| pass | panel_has_official_price_rows | ok_rows=57733 |

## Current 9/10 Blockers

- Increase high-confidence/exact truth-track coverage or explicitly tier training/evaluation sets.
- Run model-level city/time split backtests, not only deterministic forecast audits.
- Keep city sample tiers and capped evaluation weights active so small/large cities do not dominate claims.
