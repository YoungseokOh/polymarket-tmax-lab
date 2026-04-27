# Historical Real Collection Status

Updated: 2026-04-27 KST

## Current Snapshot
- workspace: `historical_real`
- dataset profile: `real_market`
- curated inventory: `configs/market_inventory/historical_temperature_snapshots.json`
- curated snapshots: `2,602`
- event URL manifest: `configs/market_inventory/historical_temperature_event_urls.json` (`2,604` URLs)
- checked-in training baseline inventory: `configs/market_inventory/full_training_set_snapshots.json` (`1,834` snapshots)
- readiness artifact: `artifacts/targeted_historical_refresh_20260427/discovery120_dataset_readiness.json`
- forecast availability artifact: `artifacts/forecast_availability.json`

## Latest Collection
- Discovery120 expansion on 2026-04-27 scanned `2,934` supported closed events, fetched `2` new candidate pages, classified `72` fetched events, and appended `1` truth-ready URL: Tokyo `1`.
- Follow-up discovery160 expansion scanned the same `2,934` supported closed events, fetched `0` new candidate pages, classified `71` retryable/source-lag fetched events, and appended `0` URLs. This satisfies the no-new-truth-ready-backlog stop condition.
- Target-only inventory validation: `1` curated snapshot, issues `0`; merged local curated inventory now has `2,602` snapshots.
- Trust check on merged curated inventory: `ok=true`, `inventory_rows=2602`.
- `backfill-markets`: `bronze_market_snapshots=6496`, `silver_market_specs=2689`.
- `backfill-truth`: `bronze_truth_snapshots=2731`, `silver_observations_daily=2610`, status counts `ok=2610`, `error=97`, `lag=24`.
- Targeted `backfill-forecasts` with `ecmwf_ifs025` and `gfs_seamless` across `market_open`, `previous_evening`, and `morning_of`: `single_run_ok=6`, `single_run_err=0`, no two-consecutive-`429` cancellation, `bronze_forecast_requests=40444`, `silver_forecast_runs_hourly=1330156`.

## Verified Readiness
- `artifacts/targeted_historical_refresh_20260427/discovery120_dataset_readiness.json`: detail rows `2,602`, summary rows `30`.
- `forecast_ready=true`: `2,602`; `truth_ready=true`: `2,602`; `gold_ready=true`: `2,598`; `gold_ready=false`: `4`.
- `readiness_status`: `ready=2,598`, `gold_missing=4`.
- `settlement_eligible=true`: `5`; `settlement_eligible=false`: `2,597`.
- `truth_status`: `ok=2,602`.
- Current non-canonical gold row sum: `7,794` rows from `2,598` markets; market ids `282535`, `285758`, `289186`, and `412000` currently materialize no gold rows.
- `artifacts/forecast_availability.json` summary rows: `645`; recommended rows: `318`; min/max availability ratio `0.0/1.0`, with the `0.0` rows limited to old `kma_ldps` London/NYC availability records.

## Current Judgment
- The curated `2,602`-market surface is fully forecast/truth-ready for research-public modeling. The current non-canonical gold variant is ready for `2,598` markets after gold materialization skips four no-row markets.
- The canonical checked-in training set and panel are not automatically replaced. Use variant `--output-name` values first and only promote canonical after evaluation.
- Discovery120 added one truth-ready URL after expansion; follow-up discovery160 appended `0` URLs, so historical collection should stop here until Gamma exposes new closed truth-ready markets or source-lag entries become ready.
- `weather_train` remains separate and should wait for the Open-Meteo free-path limit cooldown before continuing.

## Latest Discovery120 Expansion
- Target URLs:
  `artifacts/targeted_historical_refresh_20260427/discovery120_event_urls.json`.
- Target snapshots:
  `artifacts/targeted_historical_refresh_20260427/discovery120_snapshots.json`.
- Target training set:
  `data/workspaces/historical_real/parquet/gold/v2/targeted_discovery120_20260427.parquet`
  has `3` rows, `1` market, all three supported horizons, `ecmwf_ifs025` + `gfs_seamless` features, and `0` all-zero daily-max rows.
- Full local backlog variant:
  `data/workspaces/historical_real/parquet/gold/v2/historical_training_set_curated_multisource_discovery120_20260427.parquet`
  has `7,794` rows, `2,598` markets, all three supported horizons, all four configured source feature groups, and `0` all-zero daily-max rows.
- Target price history:
  `backfill-price-history --only-missing --price-no-cache` selected `1` market, wrote `11` ok requests and `565` official price points.
- Target panel:
  `data/workspaces/historical_real/parquet/gold/v2/targeted_discovery120_backtest_panel_20260427.parquet`
  has `33` token rows with coverage `ok=22`, `missing=11`, `stale=0`.
- Full local backlog panel:
  `data/workspaces/historical_real/parquet/gold/v2/historical_backtest_panel_curated_multisource_discovery120_20260427.parquet`
  has `68,550` token rows with coverage `ok=30,615`, `missing=37,501`, `stale=434`.

## Latest Targeted Beijing/Chengdu/Chongqing/Madrid Expansion
- Target URLs:
  `artifacts/targeted_historical_refresh_20260427/east_madrid_event_urls.json`.
- Target snapshots:
  `artifacts/targeted_historical_refresh_20260427/east_madrid_snapshots.json`.
- Target training set:
  `data/workspaces/historical_real/parquet/gold/v2/targeted_east_madrid_20260427.parquet`
  has `201` rows, `67` markets, all three supported horizons, `ecmwf_ifs025` + `gfs_seamless` features, and `0` all-zero daily-max rows.
- Full local backlog variant:
  `data/workspaces/historical_real/parquet/gold/v2/historical_training_set_curated_multisource_targeted_east_madrid_20260427.parquet`
  has `6,495` rows, `2,165` markets, all three supported horizons, and `0` all-zero daily-max rows.
- Target price history:
  `backfill-price-history --only-missing --price-no-cache` selected `68` markets, wrote `748` ok requests and `36,030` official price points.
- Target panel:
  `data/workspaces/historical_real/parquet/gold/v2/targeted_east_madrid_backtest_panel_20260427.parquet`
  has `2,211` token rows with coverage `ok=1,868`, `missing=302`, `stale=41`.
- Full local backlog panel:
  `data/workspaces/historical_real/parquet/gold/v2/historical_backtest_panel_curated_multisource_targeted_east_madrid_20260427.parquet`
  has `54,261` token rows with coverage `ok=17,820`, `missing=36,349`, `stale=92`.

## Latest Targeted Dallas/Atlanta/Miami Expansion
- Target URLs:
  `artifacts/targeted_historical_refresh_20260426/dallas_atlanta_miami_event_urls.json`.
- Target snapshots:
  `artifacts/targeted_historical_refresh_20260426/dallas_atlanta_miami_snapshots.json`.
- Target training set:
  `data/workspaces/historical_real/parquet/gold/v2/targeted_dallas_atlanta_miami_20260426.parquet`
  has `180` rows, `60` markets, all three supported horizons, `ecmwf_ifs025` + `gfs_seamless` features, and `0` all-zero daily-max rows.
- Full local backlog variant:
  `data/workspaces/historical_real/parquet/gold/v2/historical_training_set_curated_multisource_targeted_south_20260426.parquet`
  has `6,294` rows, `2,098` markets, all three supported horizons, and `0` all-zero daily-max rows.
- Target price history:
  `backfill-price-history --only-missing --price-no-cache` selected `60` markets, wrote `660` ok requests and `35,365` official price points.
- Target panel:
  `data/workspaces/historical_real/parquet/gold/v2/targeted_dallas_atlanta_miami_backtest_panel_20260426.parquet`
  has `1,980` token rows with coverage `ok=1,941`, `missing=11`, `stale=28`.
- Full local backlog panel:
  `data/workspaces/historical_real/parquet/gold/v2/historical_backtest_panel_curated_multisource_targeted_south_20260426.parquet`
  has `52,050` token rows with coverage `ok=15,952`, `missing=36,047`, `stale=51`.

## Latest Targeted Ankara Expansion
- Target URLs:
  `artifacts/targeted_historical_refresh_20260426/ankara_event_urls.json`.
- Target snapshots:
  `artifacts/targeted_historical_refresh_20260426/ankara_snapshots.json`.
- Target training set:
  `data/workspaces/historical_real/parquet/gold/v2/targeted_ankara_20260426.parquet`
  has `60` rows, `20` markets, all three supported horizons, `ecmwf_ifs025` + `gfs_seamless` features, and `0` all-zero daily-max rows.
- Full local backlog variant:
  `data/workspaces/historical_real/parquet/gold/v2/historical_training_set_curated_multisource_targeted_ankara_20260426.parquet`
  has `6,114` rows, `2,038` markets, all three supported horizons, and `0` all-zero daily-max rows.
- Target price history:
  `backfill-price-history --only-missing --price-no-cache` selected `20` markets, wrote `220` ok requests and `11,036` official price points.
- Target panel:
  `data/workspaces/historical_real/parquet/gold/v2/targeted_ankara_backtest_panel_20260426.parquet`
  has `660` token rows with coverage `ok=641`, `missing=14`, `stale=5`.
- Full local backlog panel:
  `data/workspaces/historical_real/parquet/gold/v2/historical_backtest_panel_curated_multisource_targeted_ankara_20260426.parquet`
  has `50,070` token rows with coverage `ok=14,011`, `missing=36,036`, `stale=23`.

## Latest Dataset/Model Experiment
- GFS-only curated variant:
  `data/workspaces/historical_real/parquet/gold/v2/historical_training_set_curated_20260425.parquet`
  has `6,063` rows, `2,021` markets, `38` columns, and only `gfs_seamless` NWP features.
- Multi-source curated variant:
  `data/workspaces/historical_real/parquet/gold/v2/historical_training_set_curated_multisource_20260425.parquet`
  has `6,063` rows, `2,021` markets, `68` columns, and `40` numeric NWP feature columns
  (`10` each for `ecmwf_ifs025`, `ecmwf_aifs025_single`, `kma_gdps`, `gfs_seamless`).
- Matching panel:
  `data/workspaces/historical_real/parquet/gold/v2/historical_backtest_panel_curated_multisource_20260425.parquet`
  has `49,509` token rows.
- Multi-source model:
  `artifacts/workspaces/historical_real/models/curated_multisource_20260425/lgbm_emos__high_neighbor_oof.pkl`.
- Quick-eval artifacts:
  `artifacts/curated_multisource_comparison_20260425.json`,
  `artifacts/curated_multisource_quick_eval_20260425.json`,
  `artifacts/curated_multisource_error_analysis_20260425.json`.
- Baseline holdout: canonical baseline CRPS_C `0.8004`, Brier `0.0719`, DirAcc `0.6169`;
  curated multi-source CRPS_C `0.6749`, Brier `0.0714`, DirAcc `0.4746`;
  curated GFS-only CRPS_C `1.5532`.
- Curated multi-source holdout: curated GFS-only CRPS_C `2.0559`, Brier `0.0929`, DirAcc `0.2709`;
  curated multi-source CRPS_C `2.8771`, Brier `0.0840`, DirAcc `0.4966`.
- Error decomposition: curated multi-source improves median CRPS (`0.5154` vs GFS-only `1.2858`) but loses the tail
  (`p90` CRPS `10.8405` vs `4.4642`) because it is under-biased (`-2.57C`) and under-dispersed
  (`mean std 2.19C` vs GFS-only `3.94C`). Do not promote this model without variance/tail calibration.

## Latest Tail Calibration Experiment
- Script:
  `scripts/pmtmax-workspace historical_real uv run python scripts/run_tail_calibration_experiment.py`.
- Report:
  `artifacts/curated_multisource_tailcal_experiment_20260426.json`.
- Tail calibration wrapper:
  `src/pmtmax/modeling/tail_calibration.py`.
- Selected diagnostic model artifacts:
  `artifacts/workspaces/historical_real/models/curated_multisource_tailcal_20260426/lgbm_emos__high_neighbor_oof_tailcal_balanced.pkl`
  and
  `artifacts/workspaces/historical_real/models/curated_multisource_tailcal_20260426/lgbm_emos__high_neighbor_oof_tailcal_aggressive.pkl`.
- Root cause signal: several tail-loss rows have unit-aware zero-Celsius sentinels in NWP daily-max features
  (`32F` in F markets, `0C` in C markets) while `feature_availability_json` still says the source is available.
  This is a data-quality/feature-validity issue, not just a global sigma issue.
- Balanced config: on rows with primary std `<1.5C` or all four source daily-max sentinels,
  blend mean `50%` toward the GFS-only model and blend std `50%` toward the GFS-only std.
- Balanced quick eval:
  curated holdout CRPS_C `2.8771 -> 1.7195`, Brier `0.0840 -> 0.0832`, DirAcc `0.4966 -> 0.4212`;
  baseline holdout CRPS_C `0.6749 -> 0.8043`, Brier `0.0714 -> 0.0754`, DirAcc `0.4746 -> 0.4346`.
- Aggressive quick eval:
  curated holdout CRPS_C `1.4565`, Brier `0.0825`, DirAcc `0.3969`;
  baseline holdout CRPS_C `0.9154`. It is too damaging to baseline behavior for promotion.
- Judgment: balanced tailcal is a useful diagnostic fallback and beats GFS-only on curated CRPS
  (`1.7195` vs `2.0559`) while staying near canonical baseline CRPS (`0.8043` vs `0.8004`),
  but it is not promotable until sentinel feature validity is fixed upstream and evaluated without holdout-tuned heuristics.

## Latest Sentinel-Fix Experiment
- Code fix: `BackfillPipeline.materialize_training_set` now treats all-zero target-day temperature aggregates
  as invalid forecast features instead of materializing them as `0C` / `32F`. Invalid single-run features fall
  back to valid `historical_forecast` or fixture features when available; otherwise the market/horizon is skipped.
- Dataset:
  `data/workspaces/historical_real/parquet/gold/v2/historical_training_set_curated_multisource_sentinelfix_20260426.parquet`.
- Panel:
  `data/workspaces/historical_real/parquet/gold/v2/historical_backtest_panel_curated_multisource_sentinelfix_20260426.parquet`.
- Model:
  `artifacts/workspaces/historical_real/models/curated_multisource_sentinelfix_20260426/lgbm_emos__high_neighbor_oof.pkl`.
- Reports:
  `artifacts/curated_multisource_sentinelfix_comparison_20260426.json`,
  `artifacts/curated_multisource_sentinelfix_tailcal_experiment_20260426.json`.
- Dataset delta: `6,063 -> 6,054` rows and `2,021 -> 2,018` markets; dropped market ids
  `282535`, `285758`, `289186` because all three horizons lacked valid forecast features.
- Sentinel delta: all-source sentinel rows `1,083 -> 0`; max sentinel count per row `4 -> 2`.
- Raw sentinel-fix quick eval:
  baseline holdout CRPS_C `0.4965`, Brier `0.0644`, DirAcc `0.6879`;
  curated holdout CRPS_C `2.7690`, Brier `0.0803`, DirAcc `0.5754`.
- GFS-only on the same sentinel-fix curated holdout: CRPS_C `2.1103`, Brier `0.0938`, DirAcc `0.2738`.
- Sentinel-fix tailcal balanced: curated holdout CRPS_C `2.3429`, baseline holdout CRPS_C `0.6418`.
- Judgment: upstream sentinel validity is fixed, but the retrained multi-source model is still not promotable.
  Curated holdout CRPS remains worse than GFS-only and city tails now include hot-bias failures
  in `Chongqing`, `Chengdu`, `Beijing`, and `Madrid`.

## Latest Source-Gating Experiment
- Code:
  `src/pmtmax/modeling/source_gating.py`,
  `scripts/run_source_gating_experiment.py`,
  `tests/test_source_gating.py`.
- Report:
  `artifacts/curated_multisource_sourcegate_experiment_20260426.json`.
- Diagnostic artifacts:
  `artifacts/workspaces/historical_real/models/curated_multisource_sourcegate_20260426/lgbm_emos__high_neighbor_oof_sourcegate_binary.pkl`,
  `..._sourcegate_regret_weighted.pkl`,
  and
  `..._sourcegate_absregret_weighted.pkl`.
- Method: train-side classifier gate between sentinel-fix multi-source and GFS-only, with binary,
  fallback-regret-weighted, and absolute-regret-weighted variants. Holdouts were not used for gate fitting.
- Gate fit split: fallback beat primary on only `8.91%` of rows; primary fit CRPS_C `0.6690`
  vs fallback fit CRPS_C `1.1596`.
- Curated holdout quick eval: raw primary CRPS_C `2.7690`; best gate
  (`sourcegate_regret_weighted`) CRPS_C `2.7607`; GFS-only CRPS_C `2.1103`.
- Baseline holdout quick eval: raw primary CRPS_C `0.4965`; best gate
  (`sourcegate_absregret_weighted`) CRPS_C `0.4932`; GFS-only CRPS_C `1.5532`.
- Diagnostic: oracle row-wise primary/fallback choice on curated holdout is `0.8340` CRPS_C,
  but learned gates selected fallback on only about `2%` of curated holdout rows. Large tail losses
  are concentrated in city/region distribution shift (`Chengdu`, `Chongqing`, `Beijing`, `Madrid`,
  `Ankara`, `Dallas`, `Atlanta`) and are not solved by a train-side binary source classifier.
- Judgment: source-gating is safe on baseline but not effective enough for promotion. Next experiment
  should model source-disagreement/region-driven variance inflation or a proper out-of-fold stacking
  gate, not a simple binary fallback classifier.

## Latest Disagreement Calibration Experiment
- Code:
  `src/pmtmax/modeling/disagreement_calibration.py`,
  `scripts/run_disagreement_calibration_experiment.py`,
  `tests/test_disagreement_calibration.py`.
- Report:
  `artifacts/curated_multisource_disagreementcal_experiment_20260426.json`.
- Diagnostic artifacts:
  `artifacts/workspaces/historical_real/models/curated_multisource_disagreementcal_20260426/lgbm_emos__high_neighbor_oof_variance_only_disagreement.pkl`,
  `..._positive_shrink_light.pkl`,
  `..._positive_shrink_minvar.pkl`,
  and
  `..._positive_shrink_balanced.pkl`.
- Method: post-hoc source-disagreement calibration on the sentinel-fix multi-source model. The
  variance-only candidate keeps the primary mean and adds variance from primary-vs-GFS disagreement.
  The positive-shrink candidates additionally blend the mean toward GFS only when multi-source is
  hotter than GFS, matching the observed hot-bias tail.
- Variance-only result: curated holdout CRPS_C `2.7690 -> 2.2342`, p90 CRPS_C `12.98 -> 9.28`,
  but still worse than GFS-only CRPS_C `2.1103`; baseline holdout worsened to `0.6635`.
- Best positive-shrink result (`positive_shrink_balanced`): curated holdout CRPS_C `1.3153`,
  Brier `0.0748`, ECE `0.0214`, DirAcc `0.5361`; baseline holdout CRPS_C worsened
  from `0.4965` to `0.5627`.
- City diagnostics: the best candidate dramatically reduced hot-bias losses in `Chongqing`,
  `Chengdu`, `Beijing`, `Madrid`, and `Ankara`, but cold-bias groups such as `Dallas`,
  `Atlanta`, and `Miami` still trail GFS-only.
- Judgment: this is the strongest diagnostic multi-source post-hoc result so far, but it is still
  holdout-informed and damages baseline behavior. Do not promote. The next serious candidate should
  turn the positive-disagreement rule into an explicit model variant or OOF stacking candidate and
  validate on recent-core/backtest gates.

## Next Strategy
1. Build a non-holdout-tuned candidate around positive primary-vs-GFS disagreement: either add
   disagreement-derived mean/scale features to `lgbm_emos` or implement proper OOF stacking with
   primary/fallback predictions generated out-of-fold.
2. Add a symmetric cold-bias branch only if it improves `Dallas`, `Atlanta`, and `Miami` without
   damaging primary-win cities like `Lucknow`, `Paris`, and `Wellington`.
3. Validate any candidate with a recent-core/backtest gate before promotion; all current wrappers are
   diagnostic artifacts only.
4. Keep price-history recovery serialized; the price checker still shows stagnant shard recovery and
   must not overlap forecast backfill jobs.
