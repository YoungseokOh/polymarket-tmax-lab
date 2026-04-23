# Codebase Guide

This section explains the repository by major folder. Use it together with the
thematic docs in `docs/`.

## Reading Order
1. [src.md](src.md)
2. [markets.md](markets.md)
3. [weather.md](weather.md)
4. [modeling.md](modeling.md)
5. [backtest-execution.md](backtest-execution.md)
6. [storage-and-configs.md](storage-and-configs.md)
7. [scripts-tests-notebooks.md](scripts-tests-notebooks.md)

## How To Use This Guide
- Use these files when you need folder ownership, data flow, or edit impact.
- Use `docs/overview/README.md` for the top-level documentation map.
- Use `docs/overview/architecture.md` for the high-level pipeline.
- Use `docs/markets/market-rules.md`, `docs/research/modeling.md`, and `docs/operations/live-trading.md` for domain policy.

## Folder Map
- `src/pmtmax/`: runtime package and subsystem boundaries
- `configs/`: layered config defaults
- `scripts/`: CLI wrappers and operational entrypoints
- `checker/`: agent-maintained markdown status boards, logs, and runbooks for recurring dataset checks
- `tests/`: unit and integration validation
- `notebooks/`: exploratory walkthroughs
- `docs/`: thematic docs plus this codebase guide

## Canonical Runtime Roots
- `weather_train`: weather-real station/date training rows and pretrain artifacts; no Polymarket ids, rules, prices, or CLOB history.
- `historical_real`: trusted research warehouse, training set, official-price panel, and model experiments.
- `ops_daily`: active-market scanning, Gamma price logs, scan-edge outputs, paper diagnostics, and observation queues.
- `recent_core_eval`: isolated recent-core benchmark and publish-gate runs.
- `artifacts/public_models/champion.*`: the only public champion alias consumed by operational commands.

Root `data/*` and `artifacts/*` paths are legacy/default paths unless a specific
doc calls them out as public aliases. Use `scripts/pmtmax-workspace` for research
or operations so these roots do not mix.

## When To Update
- Add or remove a major subsystem: update the matching file here and `AGENTS.md`.
- Change a workflow entrypoint: update this guide, `README.md`, and agent assets.
- Move responsibilities between folders: update the affected folder guides in the same change.
