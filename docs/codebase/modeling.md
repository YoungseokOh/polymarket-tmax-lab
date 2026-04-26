# modeling

## Responsibility
`src/pmtmax/modeling` turns station-aligned forecast features into probabilistic
daily-max forecasts, calibrates them, and maps them onto Polymarket outcomes.

## Key Modules
- `baselines/`: public stable baseline is `gaussian_emos`
- `advanced/`: public stable candidates include `tuned_ensemble` (contextual mixture-of-experts), `det2prob_nn` (mixture-density NN), and `lgbm_emos`; older scaffolds may remain in-tree but are not part of the supported registry
- `train.py`: model registry, feature selection, artifact serialization
- `predict.py`: model loading, forecast generation, and outcome-probability mapping
- `design_matrix.py`: shared contextual featurization for availability, missingness, city, horizon, and seasonality
- `daily_max.py`: hourly-to-daily-max transforms
- `bin_mapper.py`: exact Polymarket outcome mapping for parametric or sample forecasts
- `tail_calibration.py`: post-hoc Gaussian mean/std wrappers for diagnostic tail fallback experiments
- `source_gating.py`: post-hoc train-side Gaussian source blend wrappers for diagnostic
  multi-source vs GFS-only gate experiments
- `disagreement_calibration.py`: post-hoc Gaussian wrappers for diagnostic
  source-disagreement mean/variance calibration experiments
- `calibration.py`, `evaluation.py`, `sampling.py`, `champion.py`: supporting evaluation and selection utilities

## Data Contract
- Training expects a tabular dataset from `backtest/dataset_builder.py`.
- Weather pretraining expects `data/workspaces/weather_train/parquet/gold/weather_training_set.parquet`, produced by `collect-weather-training`; this dataset is station/date real weather only and must not contain Polymarket market ids, rule JSON, prices, or CLOB history.
- Shared targets are `realized_daily_max` and, when available, `winning_outcome`.
- Model outputs feed `ProbForecast` and then execution logic.
- The canonical contract adds `contract_version`, `group_id`, `split_group`, `feature_availability_json`, grouped split metadata, and optional sibling calibrator artifacts.
- `ProbForecast` carries raw and calibrated outcome probabilities, distribution family/payload, and feature-availability metadata.

## What To Keep Aligned
- Baseline and advanced model names must stay in sync with CLI defaults and docs.
- `train-weather-pretrain` runs only under the `weather_real` profile; `train-advanced` remains a `real_market` Polymarket adaptation/fine-tune path and can record `--pretrained-weather-model` lineage.
- The public registry must stay aligned with `configs/base.yaml` benchmark candidates and champion publishing.
- Current public champion docs should name the exact model family and variant; the current alias is `lgbm_emos / high_neighbor_oof`.
- Probability mapping logic must remain consistent with `MarketSpec.outcome_schema`.
- If feature names or contextual derived columns change, check `design_matrix.py`, `train.py`, `predict.py`, calibrator persistence, and the dataset builder together.
- Experimental or historical model files should not silently re-enter the public registry.

## Change Checklist
- Changes to market outcome mapping require `tests/test_bin_mapper.py`.
- Changes to daily-max transforms require `tests/test_daily_max.py`.
- If a model becomes the preferred champion, update README workflow examples only if the default user path changes.
