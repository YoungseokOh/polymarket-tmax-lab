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
- Use `docs/architecture.md` for the high-level pipeline.
- Use `docs/market-rules.md`, `docs/modeling.md`, and `docs/live-trading.md` for domain policy.

## Folder Map
- `src/pmtmax/`: runtime package and subsystem boundaries
- `configs/`: layered config defaults
- `scripts/`: CLI wrappers and operational entrypoints
- `tests/`: unit and integration validation
- `notebooks/`: exploratory walkthroughs
- `docs/`: thematic docs plus this codebase guide

## When To Update
- Add or remove a major subsystem: update the matching file here and `AGENTS.md`.
- Change a workflow entrypoint: update this guide, `README.md`, and agent assets.
- Move responsibilities between folders: update the affected folder guides in the same change.
