---
name: pmtmax-phase-chain
description: Use when planning or running a multi-candidate autoresearch phase in polymarket-tmax-lab — creating YAML candidates, running sequential steps, comparing CRPS, promoting winners, and validating via paper trading.
---

# pmtmax-phase-chain

Use this skill when designing and running a research phase: creating candidate YAMLs, running steps sequentially, comparing results, promoting the best candidate, and validating with paper trading.

## First reads
1. Read `AGENTS.md`
2. Read `docs/agent-skills/autoresearch.md`
3. Read `docs/agent-skills/research-loop.md`

## Phase Design Rules
- Each phase targets one clear hypothesis (e.g. "debias YES direction", "reduce warm bias").
- Create 2-4 candidate YAMLs under `artifacts/autoresearch/<run_tag>/candidates/`.
- Base all candidates on `recency_neighbor_oof` unless explicitly overriding.
- Champion parameters are the baseline — change only the knobs that test the hypothesis.

## Current Run Tag
`20260410-lgbm-275k-expansion`

## Step-Only Workflow (no gate)
Gate is slow (~4h). Skip it unless two candidates are within 0.005 CRPS.

```bash
# 1. Run step for each candidate (sequential)
uv run pmtmax autoresearch-step artifacts/autoresearch/<run_tag>/candidates/<name>.yaml

# 2. Compare step CRPS — lower is better
# Champion step CRPS baseline: 0.7887 (debias_short_halflife)

# 3. Promote best if it beats champion
uv run pmtmax autoresearch-promote \
  artifacts/autoresearch/<run_tag>/candidates/<best>.yaml \
  --publish-champion --force

# 4. Validate with paper trading
uv run pmtmax scan-markets
uv run pmtmax scan-edge --min-model-prob 0.05 --max-model-prob 0.95
uv run pmtmax paper-trader --price-source gamma
```

## Chain Script Pattern
For overnight runs, use `/tmp/p7_chain.sh` pattern:
- run_step() skips if quick_eval already exists (reboot-safe)
- compares all step CRPSs, promotes best automatically
- runs scan-edge + paper-trader after promotion
- sends Telegram notification on completion

## Available Debias Knobs (Phase 7+)
| Param | Default | Effect |
|-------|---------|--------|
| `quantile_center_alpha` | 0.5 | 0.4 → shift all YES probs down ~5-10% |
| `use_city_month` | false | city×month dummies → per-city seasonal bias |
| `use_city_lat` | false | latitude feature → hemisphere correction |
| `recency_half_life_days` | 30 | 15 → emphasise recent spring data |
| `drop_dead_features` | false | drops xmod_*/kma_gdps_*/gfs_seamless_* |

## Performance Notes
- Quantile models (q10/qCenter/q90) now train **in parallel** via ThreadPoolExecutor(3)
- Step time: ~50 min (was ~2h before parallel training)
- Gate time: ~4h (market_day CV) — skip unless candidates are very close
- Memory per step: ~3-4GB peak

## Validation Metric Priority
1. Paper trading YES win rate (target: >70%, currently ~53%)
2. Paper trading overall win rate (target: >80%)
3. Step CRPS (proxy for probabilistic quality, lower is better)
4. Gate CRPS (only if candidates are close in step)

## Current Champion
- Name: `debias_short_halflife`
- Step CRPS: 0.7887
- Gate CRPS: 0.8068
- Key params: halflife=15d, quantile loss, use_neighbor_delta=true

## Paper Trade Stats (as of Apr 13 2026)
- Overall win rate: 77% (56 settled)
- NO direction: 89% (strong)
- YES direction: 53% (target of Phase 7 fix)
- Losses concentrated in: Singapore/HK April, Chengdu, Milan, Seattle
