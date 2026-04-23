# Autoresearch

## Use This When
- `autoresearch-init`
- `autoresearch-step`
- `autoresearch-gate`
- `autoresearch-analyze-paper`
- `autoresearch-promote`
- current LGBM champion-adjacent variant 탐색

## Baseline
- current public champion variant: `high_neighbor_oof`
- new champion-adjacent runs should pass `--baseline-variant high_neighbor_oof`
- existing run manifests are authoritative; do not silently rewrite older
  `recency_neighbor_oof` runs
- model family: `lgbm_emos`
- canonical aliases are off-limits during the loop unless promotion explicitly asks for publish

## Workflow
```bash
scripts/pmtmax-workspace historical_real uv run pmtmax autoresearch-init --baseline-variant high_neighbor_oof

# edit one YAML candidate under artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/
scripts/pmtmax-workspace historical_real uv run pmtmax autoresearch-step --spec-path artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/my_candidate.yaml
scripts/pmtmax-workspace historical_real uv run pmtmax autoresearch-gate --spec-path artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/my_candidate.yaml
scripts/pmtmax-workspace historical_real uv run pmtmax autoresearch-analyze-paper --spec-path artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/my_candidate.yaml
scripts/pmtmax-workspace historical_real uv run pmtmax autoresearch-promote --spec-path artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/my_candidate.yaml
```

`scripts/autoresearch.sh` is a thin wrapper around the same CLI:

```bash
bash scripts/autoresearch.sh init
bash scripts/autoresearch.sh step --spec-path artifacts/workspaces/historical_real/autoresearch/<run_tag>/candidates/my_candidate.yaml
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
Quick evaluation reports both `crps_celsius_normalized` and `crps_market_unit`.
Use the Celsius-normalized value for mixed-unit model comparisons and keep the raw
market-unit value only as an audit field.

### New Process
1. **Quick step** (2 min): check CRPS + DirAcc + ECE. Only proceed if DirAcc improves or CRPS improves by >0.001.
2. **Shadow scan-edge** (same day): run scan-edge with `--model-path` pointing to candidate (not published alias). Compare signals vs current champion.
3. **Gate**: run `autoresearch-gate` and require real-history sample adequacy, calibration, CRPS improvement, and non-worse real-history PnL versus baseline.
4. **Paper/live-shadow analysis**: run `autoresearch-analyze-paper`; `overall_gate_decision` must be `GO`.
5. **Promote YAML only**: `autoresearch-promote` is fail-closed. `INCONCLUSIVE`, missing leaderboard artifacts, missing calibrator, signature mismatch, or manually bypassed summaries block promotion.

### Quick Step Decision Criteria
- `CRPS -> DirAcc (desc) -> Brier -> ECE (asc) -> MAE`
- DirAcc = fraction of rows where top-1 predicted bin == actual winning bin (higher = better)
- ECE = Expected Calibration Error (lower = better; well-calibrated model should have ECE < 0.05)

### Legacy Gate Rules
- benchmark gate compares candidate vs baseline on grouped holdout under both `market_day` and `target_day`
- benchmark gate summaries must reference existing leaderboard JSON/CSV artifacts
- paper analysis uses direct candidate `--model-path`, not public aliases
- paper analysis must be complete and `overall_gate_decision=GO`; `INCONCLUSIVE` is not promotable
- promotion copies the winning YAML into `configs/autoresearch/lgbm_emos/promoted/`
- public alias publish is explicit and handled by `publish-champion` only

## Safety Rules
- do not rewrite canonical `historical_training_set*` / `historical_backtest_panel`
- use real-only `historical_real` data; synthetic augmentation is not allowed
- for new research near the public champion, base candidates on `high_neighbor_oof`
- do not rewrite the public `champion` alias during quick research loops
- prefer one small candidate YAML change over broad registry edits
- if a promoted YAML exists, it becomes a supported `--variant` for later runs
