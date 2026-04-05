# Autoresearch

## Use This When
- `autoresearch-init`
- `autoresearch-step`
- `autoresearch-gate`
- `autoresearch-analyze-paper`
- `autoresearch-promote`
- `recency_neighbor_oof` 기준 LGBM variant 탐색

## Baseline
- current baseline variant: `recency_neighbor_oof`
- model family: `lgbm_emos`
- canonical aliases are off-limits during the loop unless promotion explicitly asks for publish

## Workflow
```bash
uv run pmtmax autoresearch-init

# edit one YAML candidate under artifacts/autoresearch/<run_tag>/candidates/
uv run pmtmax autoresearch-step --spec-path artifacts/autoresearch/<run_tag>/candidates/my_candidate.yaml
uv run pmtmax autoresearch-gate --spec-path artifacts/autoresearch/<run_tag>/candidates/my_candidate.yaml
uv run pmtmax autoresearch-analyze-paper --spec-path artifacts/autoresearch/<run_tag>/candidates/my_candidate.yaml
uv run pmtmax autoresearch-promote --spec-path artifacts/autoresearch/<run_tag>/candidates/my_candidate.yaml
```

`scripts/autoresearch.sh` is a thin wrapper around the same CLI:

```bash
bash scripts/autoresearch.sh init
bash scripts/autoresearch.sh step --spec-path artifacts/autoresearch/<run_tag>/candidates/my_candidate.yaml
```

## Candidate Spec
- YAML-backed and agent-editable
- one candidate at a time
- fields:
  - `run_tag`
  - `candidate_name`
  - `base_variant`
  - `description`
  - `params`
- `params` may override only editable `LgbmEMOSVariantConfig` knobs such as:
  - `n_estimators`
  - `num_leaves`
  - `max_depth`
  - `learning_rate`
  - `min_child_samples`
  - `use_recency_weights`
  - `recency_half_life_days`
  - `use_oof_scale`
  - `use_neighbor_delta`
  - `use_quantile_loss`
  - `subsample_freq`
  - `fixed_std`

## Gate Rules
- quick step decides `keep / discard / crash` by lexicographic `CRPS -> Brier -> MAE`
- benchmark gate compares candidate vs baseline on grouped holdout under both `market_day` and `target_day`
- paper analysis uses direct candidate `--model-path`, not public aliases
- promotion copies the winning YAML into `configs/autoresearch/lgbm_emos/promoted/`
- alias publish is explicit and optional

## Safety Rules
- do not rewrite canonical `historical_training_set*` / `historical_backtest_panel`
- do not rewrite `champion` / `trading_champion` during quick research loops
- prefer one small candidate YAML change over broad registry edits
- if a promoted YAML exists, it becomes a supported `--variant` for later runs
