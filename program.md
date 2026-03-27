# AutoResearch Program: det2prob_nn Architecture Search

## Goal
Improve the `det2prob_nn` neural network so it produces better-calibrated
temperature probability forecasts, measured by **CRPS** (continuous ranked
probability score, lower is better).

## Context
This is a Polymarket temperature forecasting project. The model predicts the
daily maximum temperature for cities worldwide. Forecasts are converted to
probabilities over temperature outcome brackets and used to find edge in
prediction markets.

The rolling-origin backtest simulates real-world conditions: the model trains
on all data seen so far, makes a prediction for the next observation, and the
training set grows by one group per step. Starting from very small datasets.

## Target file
`src/pmtmax/modeling/advanced/det2prob_nn.py`

This file contains:
- `Det2ProbVariantConfig` — dataclass with all hyperparameters for a variant
- `DET2PROB_VARIANTS` — dict of named configurations to try
- `FlexibleProbNet` — the PyTorch model (MLP trunk + Gaussian/MDN head)
- `Det2ProbNNModel` — sklearn-style wrapper (fit/predict)

## Eval command (run after each modification)
```bash
uv run pmtmax backtest \
  --model-name det2prob_nn \
  --pricing-source real_history \
  --variant <VARIANT_NAME> \
  --last-n 100
```
This runs the last 100 market-day groups as test points (~12 minutes).
Parse `avg_crps` from the JSON output. **Lower is better.**

## Primary metric
`avg_crps` — averaged CRPS over all test predictions.

Secondary metrics (also worth tracking):
- `mae` — mean absolute error of point forecast
- `pnl` — simulated paper trading profit (not primary due to market noise)
- `hit_rate` — fraction of winning trades

## Baseline
`legacy_gaussian` variant: CRPS ≈ 1.472, MAE ≈ 1.895 (last 100 groups)
Full backtest: CRPS = 1.40, MAE = 1.82, PnL = +124

## Current champion (autoresearch rounds 1–5)
`champion_v1`: CRPS = 1.295, MAE = 1.646 (last 100 groups) — **+12% over baseline**
Config: `legacy_raw` features, `hidden_dims=(128,64)`, `activation=silu`,
`use_recency_weights=True`, `loss_name=gaussian_nll_mean`, `use_val_split=False`,
`min_train_rows_override=30`

Key findings:
- wider network (128,64 > 64,64): +0.055 CRPS
- recency weights: +0.118 CRPS total
- SiLU activation (with recency): +0.177 CRPS total
- layernorm: catastrophically bad (CRPS 2.67!)
- contextual features: worse than raw+recency (CRPS 1.53 vs 1.29)
- depth (3 layers) or going too wide (256,128): both hurt
- min_train_rows 25/30/40: all equivalent

## Experiment loop (autoresearch style)
1. Create git branch `autoresearch/<tag>`
2. Read current `det2prob_nn.py` for full context
3. Establish baseline by running eval on `legacy_gaussian`
4. Loop:
   a. Add or modify a variant in `DET2PROB_VARIANTS`
      - Give the new variant a descriptive name (e.g. `exp_layernorm_gaussian`)
      - Keep changes focused: one variable at a time
   b. Commit the change
   c. Run eval on the new variant
   d. If CRPS improved → keep commit, note result in `results.tsv`
   e. If CRPS equal or worse → `git reset HEAD~1 --hard`, discard

## Key levers to explore (in roughly increasing complexity)
1. `min_train_rows_override`: try 20, 25, 30, 40 — earlier training start may help
2. `hidden_dims`: try (32,32), (64,32), (128,64), (64,64,32)
3. `activation`: relu vs silu
4. `use_layernorm`: True may help with deeper networks, hurts small ones
5. `dropout`: 0.05–0.1 for deeper nets only
6. `loss_name`: `gaussian_nll` vs `gaussian_nll_mean` — does mean anchor help?
7. `use_val_split=False` + small `min_train_rows_override` (v1-style training)
8. `use_recency_weights`: True adds exponential decay weighting on training rows
9. `feature_mode`: `legacy_raw` vs `contextual` — contextual adds seasonality

## Hard constraints
- DO NOT modify `FlexibleProbNet` forward pass or loss functions — these are
  architecturally sound. Only add new entries to `DET2PROB_VARIANTS`.
- DO NOT change `_MIN_TRAIN_ROWS` (the global default) — use `min_train_rows_override`
- Each new variant must be a new key in `DET2PROB_VARIANTS`
- Always evaluate with `--pricing-source real_history --last-n 100`

## Results log
Append to `results.tsv` (untracked, created if absent):
```
commit_hash\tvariant_name\tavg_crps\tmae\tpnl\tstatus\tdescription
```

## Decision heuristic
- CRPS improved by > 0.02 → strong keep
- CRPS improved by 0.005–0.02 → keep, but note small gain
- CRPS unchanged or worse → revert
- MAE improved even if CRPS flat → borderline, keep and note
