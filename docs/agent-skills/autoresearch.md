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

## Champion Selection Rules (updated 2026-04-15)

Champion is selected by **paper-trading win rate + PnL**, NOT CRPS alone.

### New Process
1. **Quick step** (2 min): check CRPS + DirAcc + ECE. Only proceed if DirAcc improves or CRPS improves by >0.001.
2. **Shadow scan-edge** (same day): run scan-edge with `--model-path` pointing to candidate (not published alias). Compare signals vs current champion.
3. **Paper trade** (3–7 days): record candidate signals in separate log. After settlement, compare win rate + PnL vs champion baseline.
4. **Promote** if candidate shows better win rate or PnL with comparable CRPS.
5. **Gate** (5h benchmark): only for large CRPS jumps (>0.005) or major architectural changes. Skip for incremental tuning.

### Quick Step Decision Criteria
- `CRPS -> DirAcc (desc) -> Brier -> ECE (asc) -> MAE`
- DirAcc = fraction of rows where top-1 predicted bin == actual winning bin (higher = better)
- ECE = Expected Calibration Error (lower = better; well-calibrated model should have ECE < 0.05)

### Legacy Gate Rules
- benchmark gate compares candidate vs baseline on grouped holdout under both `market_day` and `target_day`
- paper analysis uses direct candidate `--model-path`, not public aliases
- promotion copies the winning YAML into `configs/autoresearch/lgbm_emos/promoted/`
- alias publish is explicit and optional

## Safety Rules
- do not rewrite canonical `historical_training_set*` / `historical_backtest_panel`
- do not rewrite `champion` / `trading_champion` during quick research loops
- prefer one small candidate YAML change over broad registry edits
- if a promoted YAML exists, it becomes a supported `--variant` for later runs
