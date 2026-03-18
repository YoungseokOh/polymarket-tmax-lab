# Papers

## A. Review / foundations

### Vannitsem et al. (2021)
- Problem: broad review of statistical postprocessing, calibration, dependence, and operational scaling.
- Why it matters: max-temperature markets need calibrated station-level predictive distributions, not raw model outputs.
- Implemented here: benchmark ladder, calibration layer, separation between deterministic and ensemble-aware postprocessing.
- Simplified/omitted: full multivariate dependence modeling and large-scale operational big-data systems.

## B. Operational / classical baselines

### Ylinen et al. (2020)
- Problem: operational station-specific temperature postprocessing using ML predictors.
- Why it matters: supports practical station-aware baselines for airport-based settlement targets.
- Implemented here: station features, deterministic predictors, heteroscedastic linear/Gaussian baselines.
- Simplified/omitted: original operational feature breadth and proprietary forecast inputs.

### Jobst et al. (2024)
- Problem: time-series based ensemble MOS for temperature postprocessing.
- Why it matters: lead time and recent temporal context matter for recurring daily weather markets.
- Implemented here: `ts_emos`-inspired lag/lead-aware polynomial baseline.
- Simplified/omitted: exact paper architecture and richer ensemble state.

### Wessel et al. (2024)
- Problem: lead-time-continuous postprocessing across horizons.
- Why it matters: these markets can be priced at market open, previous evening, morning-of, and closer to settlement.
- Implemented here: `leadtime_continuous` baseline with continuous lead-hour features.
- Simplified/omitted: full continuous parameterization across all operational lead regimes.

## C. Strong modern methods

### Landry et al. (2024)
- Problem: convert deterministic forecast inputs into in-situ probabilistic temperature predictions with deep learning.
- Why it matters: public deterministic models are accessible even when full ensembles are not.
- Implemented here: `advanced/det2prob_nn.py`, the most complete advanced v1 model.
- Simplified/omitted: paper-faithful architecture details and large training corpora.

### Mlakar et al. (2024 / 2023)
- Problem: flexible probabilistic neural-network postprocessing.
- Why it matters: richer uncertainty shapes can matter near bin boundaries.
- Implemented here: mixture-density approximation in `flexible_flow_nn.py`.
- Simplified/omitted: original flexible distribution family and reference-code specifics.

### Höhlein et al. (2024)
- Problem: permutation-invariant neural processing of ensemble members.
- Why it matters: pseudo-ensemble and true ensemble members should be order-invariant.
- Implemented here: Deep-Sets style approximation in `pinn_postproc.py`.
- Simplified/omitted: paper-faithful architecture details and richer member metadata.

### Feik et al. (2024)
- Problem: spatial learning for postprocessing across station graphs.
- Why it matters: neighboring stations can help when source stations are sparse or noisy.
- Implemented here: GNN-inspired neighbor aggregation scaffold in `spatial_gnn.py`.
- Simplified/omitted: full graph neural network message passing and large station graphs.

### Van Poecke et al. (2025)
- Problem: transformer-based fast postprocessing for temperature and wind.
- Why it matters: hourly trajectory modeling benefits from sequence-aware encoders.
- Implemented here: transformer approximation in `transformer_postproc.py`.
- Simplified/omitted: paper-faithful architecture depth, tokenization, and training regime.

## D. AI/NWP blending

### Trotta et al. (2025)
- Problem: accurate probabilistic forecasts from AI weather models via statistical postprocessing.
- Why it matters: AIFS-style AI models can complement IFS/KMA in recurring market settings.
- Implemented here: `aifs_nwp_blend.py` with NWP-only, AI-only, and learned blend support.
- Simplified/omitted: original model suite and operational evaluation framework.

## Mapping To Repo Strategy
- Tier 0: climatology, raw best-model, multimodel average
- Tier 1: Gaussian EMOS, heteroscedastic calibration
- Tier 2: tsEMOS-inspired, lead-time continuous
- Tier 3: det2prob neural model
- Tier 4: permutation-invariant, flexible probabilistic NN, transformer
- Tier 5: spatial GNN scaffold

